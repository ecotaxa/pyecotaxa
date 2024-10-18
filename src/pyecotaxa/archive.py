"""Read and write EcoTaxa archives and individual EcoTaxa TSV files."""

import collections
import csv
import fnmatch
import io
import pathlib
import posixpath
import shutil
import tarfile
import zipfile
from io import BufferedReader, BytesIO, IOBase
from typing import (
    IO,
    Any,
    BinaryIO,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd
from tqdm.auto import tqdm

__all__ = ["read_tsv", "write_tsv"]


DEFAULT_DTYPES = {
    "img_file_name": str,
    "img_rank": int,
    "object_id": str,
    "object_link": str,
    "object_lat": float,
    "object_lon": float,
    "object_date": str,
    "object_time": str,
    "object_annotation_date": str,
    "object_annotation_time": str,
    "object_annotation_category": str,
    "object_annotation_category_id": "Int64",
    "object_annotation_person_name": str,
    "object_annotation_person_email": str,
    "object_annotation_status": str,
    "process_id": str,
    "acq_id": str,
    "sample_id": str,
}

VALID_PREFIXES = {"object", "sample", "acq", "process", "img"}


def _parse_tsv_header(
    f: IOBase, encoding: str
) -> Tuple[Optional[Sequence[str]], Dict[str, Any], int]:
    skiprows = 0

    header: List[str] = []
    while len(header) < 2:
        line = f.readline()

        if not line:
            break

        skiprows += 1

        if isinstance(line, bytes):
            line = line.decode(encoding)

        if not line.startswith("#"):
            header.append(line)

    if not header:
        return None, {}, 0

    csv_reader = csv.reader(header, delimiter="\t")

    names = next(csv_reader)

    try:
        maybe_types = next(csv_reader)
    except StopIteration:
        # No second line
        return names, {}, skiprows

    if len(names) != len(maybe_types):
        raise ValueError("Number of names does not match number of types")

    # Second line *might* contain types
    if all(t in ("[t]", "[f]") for t in maybe_types):
        # Infer dtype
        dtype = {n: str for n, t in zip(names, maybe_types) if t == "[t]"}
    else:
        # This wasn't a type row after all
        dtype = {}
        skiprows -= 1

    return names, dtype, skiprows


def read_tsv(
    fn_or_f: Union[str, pathlib.Path, IOBase],
    encoding: str = "utf-8-sig",
    dtype=None,
    enforce_types=True,
    **kwargs,
):
    """
    We just use the type header (if provided) to make sure that the appropriate columns are treated as strings.
    """
    must_close = False
    f: BinaryIO

    if dtype is None:
        dtype = DEFAULT_DTYPES

    if isinstance(fn_or_f, str):
        fn_or_f = pathlib.Path(fn_or_f)

    if hasattr(fn_or_f, "open"):
        f = fn_or_f.open("r", encoding=encoding)  # type: ignore
        must_close = True
    else:
        f = fn_or_f  # type: ignore

    try:
        if f.seekable():
            # We can just rewind after inspecting the header
            names, header_dtype, skiprows = _parse_tsv_header(f, encoding)
            f.seek(0)
        else:
            # Make sure that we can peek into the file
            if not hasattr(f, "peek"):
                f = BufferedReader(f)  # type: ignore

            # Peek the first 8kb and inspect
            header_f = BytesIO(f.peek(8 * 1024))  # type: ignore
            names, header_dtype, skiprows = _parse_tsv_header(header_f, encoding)

        if enforce_types:
            dtype = {**dtype, **header_dtype}

        # Detect duplicate names
        duplicate_names = [
            f"'{name}' ({count}x)"
            for name, count in collections.Counter(names).items()
            if count > 1
        ]
        if duplicate_names:
            raise ValueError(
                "TSV file contains duplicate column names: "
                + (", ".join(duplicate_names))
            )

        dataframe = pd.read_csv(f, sep="\t", names=names, dtype=dtype, skiprows=skiprows, **kwargs)  # type: ignore

        for c, dt in dataframe.dtypes.items():
            if pd.api.types.is_string_dtype(dt):
                dataframe[c] = dataframe[c].fillna("")

        return dataframe
    finally:
        if must_close:
            f.close()


def _dtype_to_ecotaxa(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return "[f]"

    return "[t]"


def write_tsv(
    dataframe: pd.DataFrame,
    path_or_buf=None,
    encoding="utf-8",
    type_header=True,
    formatters: Optional[Mapping] = None,
    **kwargs,
):
    """
    Write an individual EcoTaxa TSV file.

    Args:
        dataframe: A pandas DataFrame.
        path_or_buf (str, path object or file-like object): ...
        encoding: Encoding of the TSV file.
            With the default "utf-8", both UTF8 and signed UTF8 readers can read the file.
        enforce_types: Enforce the column dtypes provided in the header.
            Usually, it is desirable to allow pandas to infer the column dtypes.
        type_header (bool, default true): Include the type header ([t]/[f]).
            This is required for a successful import into EcoTaxa.

    Return:
        None or str

            If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
    """

    if formatters is None:
        formatters = {}

    dataframe = dataframe.copy(deep=False)

    # Calculate type header before formatting values
    ecotaxa_types = [_dtype_to_ecotaxa(dt) for dt in dataframe.dtypes]

    # Apply formatting
    for col in dataframe.columns:
        fmt = formatters.get(col)

        if fmt is None:
            continue

        dataframe[col] = dataframe[col].apply(fmt)

    if type_header:
        # Inject types into header
        dataframe.columns = pd.MultiIndex.from_tuples(
            list(zip(dataframe.columns, ecotaxa_types))
        )

    return dataframe.to_csv(
        path_or_buf, sep="\t", encoding=encoding, index=False, **kwargs
    )


class MemberNotFoundError(Exception):
    pass


class UnknownArchiveError(Exception):
    pass


class _TSVIterator:
    def __init__(self, archive: "Archive", tsv_fns, kwargs) -> None:
        self.archive = archive
        self.tsv_fns = tsv_fns
        self.kwargs = kwargs

    def __iter__(self):
        for tsv_fn in self.tsv_fns:
            with self.archive.open(tsv_fn) as f:
                yield tsv_fn, read_tsv(f, **self.kwargs)

    def __len__(self):
        return len(self.tsv_fns)


class ArchivePath:
    def __init__(self, archive: "Archive", filename) -> None:
        self.archive = archive
        self.filename = filename

    def open(self, mode="r", compress_hint=True) -> IO:
        return self.archive.open(self.filename, mode, compress_hint)

    def __truediv__(self, filename):
        return ArchivePath(self.archive, posixpath.join(self.filename, filename))


class ValidationError(Exception):
    pass


class Archive:
    """
    A generic archive reader and writer for ZIP and TAR archives.
    """

    extensions: List[str] = []

    def __new__(cls, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        archive_fn = str(archive_fn)

        if mode[0] == "r":
            for subclass in cls.__subclasses__():
                if subclass.is_readable(archive_fn):
                    return super(Archive, subclass).__new__(subclass)

            raise UnknownArchiveError(f"No handler found to read {archive_fn}")

        if mode[0] in ("a", "w", "x"):
            for subclass in cls.__subclasses__():
                if any(archive_fn.endswith(ext) for ext in subclass.extensions):
                    return super(Archive, subclass).__new__(subclass)

            raise UnknownArchiveError(f"No handler found to write {archive_fn}")

        raise ValueError("Unknown mode: {mode}")

    @staticmethod
    def is_readable(archive_fn) -> bool:
        raise NotImplementedError()  # pragma: no cover

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        raise NotImplementedError()  # pragma: no cover

    def open(self, member_fn, mode="r", compress_hint=True) -> IO:
        """
        Open an archive member.

        Raises:
            MemberNotFoundError if mode=="r" and the member was not found.
        """

        raise NotImplementedError()  # pragma: no cover

    def find(self, pattern) -> List[str]:
        return fnmatch.filter(self.members(), pattern)

    def members(self) -> List[str]:
        raise NotImplementedError()  # pragma: no cover

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        self.close()

    def iter_tsv(self, **kwargs):
        """
        Yield tuples of all (tsv_fn, data) in the archive.

        Example:
            with Archive("export.zip") as archive:
                for tsv_fn, tsv in archive.iter_tsv():
                    # tsv_fn is the location of the tsv file (img_file_name is relative to that)
                    # tsv is a pandas DataFrame containing all data
                    ...

        """
        return _TSVIterator(self, self.find("*.tsv"), kwargs)

    def __truediv__(self, key):
        return ArchivePath(self, key)

    def add_images(
        self, df: pd.DataFrame, src: Union[str, "Archive", pathlib.Path], progress=False
    ):
        """Add images referenced in df from src."""

        if isinstance(src, str):
            src = pathlib.Path(src)

        for img_file_name in tqdm(df["img_file_name"], disable=not progress):
            with (src / img_file_name).open() as f_src, self.open(
                img_file_name, "w"
            ) as f_dst:
                shutil.copyfileobj(f_src, f_dst)

    def validate(self):
        """Mimic the validation done by EcoTaxa."""

        # ecotaxa_back/py/BO/Bundle.py:43
        MAX_FILES = 2000

        tsv: pd.DataFrame

        for i, (tsv_fn, tsv) in enumerate(self.iter_tsv(), start=1):
            if i > MAX_FILES:
                raise ValidationError(
                    f"Archive contains too many files, max. is {MAX_FILES}"
                )

            errors = []

            # Validate columns (validate_structure)
            # ecotaxa_back/py/BO/TSVFile.py:873
            for c in tsv.columns:
                if c in DEFAULT_DTYPES:
                    # This is a known field
                    # TODO: Check dtype
                    continue

                try:
                    prefix, name = c.split("_", 1)
                except ValueError:
                    errors.append(
                        f"Invalid field '{c}', format must be '<prefix>_<name>'"
                    )
                    continue

                if prefix not in VALID_PREFIXES:
                    errors.append(f"Invalid prefix '{prefix}' for column '{c}'")
                    continue

            # Ensure that each used prefix contains at least an ID
            for prefix in ["object", "acq", "process", "sample"]:
                expected_id = f"{prefix}_id"
                prefix_columns = [c for c in tsv.columns if c.startswith(prefix)]
                if prefix_columns and expected_id not in tsv.columns:
                    errors.append(
                        f"Field {expected_id} is mandatory as there are some '{prefix}' columns: {sorted(prefix_columns)}."
                    )

            if errors:
                raise ValidationError(
                    f"Invalid structure in {tsv_fn}:\n" + ("\n".join(errors))
                )

            # TODO: Validate contents (validate_content)
            # ecotaxa_back/py/BO/TSVFile.py:967
            ...


class _TarIO(io.BytesIO):
    def __init__(self, archive: "TarArchive", member_fn) -> None:
        super().__init__()
        self.archive = archive
        self.member_fn = member_fn

    def close(self) -> None:
        self.seek(0)
        self.archive.write_member(self.member_fn, self)
        super().close()


class TarArchive(Archive):
    extensions = [
        ".tar",
        ".tar.bz2",
        ".tb2",
        ".tbz",
        ".tbz2",
        ".tz2",
        ".tar.gz",
        ".taz",
        ".tgz",
        ".tar.lzma",
        ".tlz",
    ]

    @staticmethod
    def is_readable(archive_fn):
        return tarfile.is_tarfile(archive_fn)

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        self._tar = tarfile.open(archive_fn, mode)
        self.__members = None

    def close(self):
        self._tar.close()

    def open(self, member_fn, mode="r", compress_hint=True) -> IO:
        # tar does not compress files individually
        del compress_hint

        if mode == "r":
            try:
                fp = self._tar.extractfile(self._resolve_member(member_fn))
            except KeyError as exc:
                raise MemberNotFoundError(
                    f"{member_fn} not in {self._tar.name}"
                ) from exc

            if fp is None:
                raise IOError("There's no data associated with this member")

            return fp

        if mode == "w":
            return _TarIO(self, member_fn)

        raise ValueError(f"Unrecognized mode: {mode}")

    @property
    def _members(self):
        if self.__members is not None:
            return self.__members
        self.__members = {
            tar_info.name: tar_info for tar_info in self._tar.getmembers()
        }
        return self.__members

    def _resolve_member(self, member):
        return self._members[member]

    def write_member(
        self, member_fn: str, fileobj_or_bytes: Union[IO, bytes], compress_hint=True
    ):
        # tar does not compress files individually
        del compress_hint

        if isinstance(fileobj_or_bytes, bytes):
            fileobj_or_bytes = io.BytesIO(fileobj_or_bytes)

        if isinstance(fileobj_or_bytes, io.BytesIO):
            tar_info = tarfile.TarInfo(member_fn)
            tar_info.size = len(fileobj_or_bytes.getbuffer())
        else:
            tar_info = self._tar.gettarinfo(arcname=member_fn, fileobj=fileobj_or_bytes)

        self._tar.addfile(tar_info, fileobj=fileobj_or_bytes)
        self._members[tar_info.name] = tar_info

    def members(self):
        return self._tar.getnames()


class ZipArchive(Archive):
    extensions = [".zip"]

    @staticmethod
    def is_readable(archive_fn):
        return zipfile.is_zipfile(archive_fn)

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        self._zip = zipfile.ZipFile(archive_fn, mode)

    def members(self):
        return self._zip.namelist()

    def open(self, member_fn: str, mode="r", compress_hint=True) -> IO:
        if mode == "w" and not compress_hint:
            # Disable compression
            member = zipfile.ZipInfo(member_fn)
            member.compress_type = zipfile.ZIP_STORED
        else:
            # Let ZipFile.open select compression and compression level
            member = member_fn

        try:
            return self._zip.open(member, mode)
        except KeyError as exc:
            raise MemberNotFoundError(
                f"{member_fn} not in {self._zip.filename}"
            ) from exc

    def write_member(
        self, member_fn: str, fileobj_or_bytes: Union[IO, bytes], compress_hint=True
    ):
        compress_type = zipfile.ZIP_DEFLATED if compress_hint else zipfile.ZIP_STORED
        # TODO: Optimize for on-disk files and BytesIO (.getvalue())
        if isinstance(fileobj_or_bytes, bytes):
            return self._zip.writestr(
                member_fn, fileobj_or_bytes, compress_type=compress_type
            )

        self._zip.writestr(
            member_fn, fileobj_or_bytes.read(), compress_type=compress_type
        )

    def close(self):
        self._zip.close()
