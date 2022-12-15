"""Read and write EcoTaxa archives and individual EcoTaxa TSV files."""

import fnmatch
import io
import pathlib
import tarfile
import warnings
import zipfile
from typing import IO, Callable, List, Union

import numpy as np
import pandas as pd

__all__ = ["read_tsv", "write_tsv"]


def _fix_types(dataframe, enforce_types):
    header = dataframe.columns.get_level_values(0)
    types = dataframe.columns.get_level_values(1)

    dataframe.columns = header

    float_cols = []
    text_cols = []
    for c, t in zip(header, types):
        if t == "[f]":
            float_cols.append(c)
        elif t == "[t]":
            text_cols.append(c)
        else:
            # If the first row contains other values than [f] or [t],
            # it is not a type header but a normal line of values and has to be inserted into the dataframe.
            # This is the case for "General export".

            # Clean up empty fields
            types = [None if t.startswith("Unnamed") else t for t in types]

            # Prepend the current "types" to the dataframe
            row0 = pd.DataFrame([types], columns=header).astype(dataframe.dtypes)

            if enforce_types:
                warnings.warn(
                    "enforce_types=True, but no type header was found.", stacklevel=3
                )

            return pd.concat((row0, dataframe), ignore_index=True)

    if enforce_types:
        # Enforce [f] types
        dataframe[float_cols] = dataframe[float_cols].astype(float)
        dataframe[text_cols] = dataframe[text_cols].fillna("").astype(str)

    return dataframe


def _apply_usecols(
    df: pd.DataFrame, usecols: Union[Callable, List[str]]
) -> pd.DataFrame:
    if callable(usecols):
        columns = [c for c in df.columns.get_level_values(0) if usecols(c)]
    else:
        columns = [c for c in df.columns.get_level_values(0) if c in usecols]

    return df[columns]


def read_tsv(
    filepath_or_buffer,
    encoding: str = "utf-8-sig",
    enforce_types=False,
    usecols: Union[None, Callable, List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read an individual EcoTaxa TSV file.

    Args:
        filepath_or_buffer (str, path object or file-like object): ...
        encoding: Encoding of the TSV file.
            With the default "utf-8-sig", both UTF8 and signed UTF8 can be read.
        enforce_types: Enforce the column dtypes provided in the header.
            Usually, it is desirable to allow pandas to infer the column dtypes.
        usecols: List of strings or callable.
        **kwargs: Additional kwargs are passed to :func:`pandas:pandas.read_csv`.

    Returns:
        A Pandas :class:`~pandas:pandas.DataFrame`.
    """

    if usecols is not None:
        chunksize = kwargs.pop("chunksize", 10000)

        # Read a few rows a time
        dataframe: pd.DataFrame = pd.concat(
            [
                _apply_usecols(chunk, usecols)
                for chunk in pd.read_csv(
                    filepath_or_buffer,
                    sep="\t",
                    encoding=encoding,
                    header=[0, 1],
                    chunksize=chunksize,
                    **kwargs,
                )
            ]
        )  # type: ignore
    else:
        if kwargs.pop("chunksize", None) is not None:
            warnings.warn("Parameter chunksize is ignored.")

        dataframe: pd.DataFrame = pd.read_csv(
            filepath_or_buffer, sep="\t", encoding=encoding, header=[0, 1], **kwargs
        )  # type: ignore

    return _fix_types(dataframe, enforce_types)


def _dtype_to_ecotaxa(dtype):
    if np.issubdtype(dtype, np.number):
        return "[f]"

    return "[t]"


def write_tsv(
    dataframe: pd.DataFrame,
    path_or_buf=None,
    encoding="utf-8",
    type_header=True,
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

    if type_header:
        # Make a copy before changing the index
        dataframe = dataframe.copy()

        # Inject types into header
        type_header = [_dtype_to_ecotaxa(dt) for dt in dataframe.dtypes]
        dataframe.columns = pd.MultiIndex.from_tuples(
            list(zip(dataframe.columns, type_header))
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

    @staticmethod
    def is_readable(archive_fn) -> bool:
        raise NotImplementedError()  # pragma: no cover

    def __init__(self, archive_fn: Union[str, pathlib.Path], mode: str = "r"):
        raise NotImplementedError()  # pragma: no cover

    def open(self, member_fn, mode="r") -> IO:
        """
        Raises:
            MemberNotFoundError if a member was not found
        """
        raise NotImplementedError()  # pragma: no cover

    def write_member(
        self, member_fn, fileobj_or_bytes: Union[IO, bytes], compress_hint=True
    ):
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

    def open(self, member_fn, mode="r") -> IO:
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
        if isinstance(fileobj_or_bytes, bytes):
            fileobj_or_bytes = io.BytesIO(fileobj_or_bytes)

        if isinstance(fileobj_or_bytes, io.BytesIO):
            tar_info = tarfile.TarInfo(member_fn)
            tar_info.size = len(fileobj_or_bytes.getbuffer())
        else:
            tar_info = self._tar.gettarinfo(arcname=member_fn, fileobj=fileobj_or_bytes)

        self._tar.addfile(tar_info, fileobj=fileobj_or_bytes)

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

    def open(self, member_fn, mode="r") -> IO:
        try:
            return self._zip.open(member_fn, mode)
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
