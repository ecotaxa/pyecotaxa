"""Read and write EcoTaxa archives and individual EcoTaxa TSV files."""

import warnings

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


def read_tsv(
    filepath_or_buffer, encoding=None, enforce_types=False, **kwargs
) -> pd.DataFrame:
    """
    Read an individual EcoTaxa TSV file.

    Args:
        filepath_or_buffer (str, path object or file-like object): ...
        encoding: ...
        enforce_types: Enforce the column dtypes provided in the header.
            Usually, it is desirable to allow pandas to infer the column dtypes.

    Return:
        A Pandas dataframe.
    """

    if encoding is None:
        encoding = "ascii"

    dataframe = pd.read_csv(
        filepath_or_buffer, sep="\t", encoding=encoding, header=[0, 1], **kwargs
    )

    return _fix_types(dataframe, enforce_types)


def _dtype_to_ecotaxa(dtype):
    if np.issubdtype(dtype, np.number):
        return "[f]"

    return "[t]"


def write_tsv(
    dataframe: pd.DataFrame, path_or_buf=None, encoding=None, type_header=True, **kwargs
):
    """
    Write an individual EcoTaxa TSV file.

    Args:
        dataframe: A pandas DataFrame.
        path_or_buf (str, path object or file-like object): ...
        encoding: ...
        enforce_types: Enforce the column dtypes provided in the header.
            Usually, it is desirable to allow pandas to infer the column dtypes.
        type_header (bool, default true): Include the type header ([t]/[f]).
            This is required for a successful import into EcoTaxa.

    Return:
        None or str

            If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
    """

    if encoding is None:
        encoding = "ascii"

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
