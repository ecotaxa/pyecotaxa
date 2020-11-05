"""Read and write EcoTaxa archives and individual EcoTaxa TSV files."""

import pandas as pd
import numpy as np

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
            # For now, we don't support such files and raise an Exception instead.
            raise ValueError("Unexpected type: {t!r}")

    if enforce_types:
        # Enforce [f] types
        dataframe[float_cols] = dataframe[float_cols].astype(float)
        dataframe[text_cols] = dataframe[text_cols].astype(str)

    return dataframe


def read_tsv(filepath_or_buffer, encoding=None, enforce_types=False) -> pd.DataFrame:
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

    # TODO: Use default encoding that EcoTaxa uses
    # if encoding is None:
    #     encoding = ...

    dataframe = pd.read_csv(
        filepath_or_buffer, sep="\t", encoding=encoding, header=[0, 1]
    )

    return _fix_types(dataframe, enforce_types)


def _dtype_to_ecotaxa(dtype):
    if np.issubdtype(dtype, np.number):
        return "[f]"

    return "[t]"


def write_tsv(dataframe: pd.DataFrame, path_or_buf=None, encoding=None):

    # TODO: Use default encoding that EcoTaxa uses
    # if encoding is None:
    #     encoding = ...

    # Make a copy before changing the index
    dataframe = dataframe.copy()

    # Inject types into header
    type_header = [_dtype_to_ecotaxa(dt) for dt in dataframe.dtypes]
    dataframe.columns = pd.MultiIndex.from_tuples(
        list(zip(dataframe.columns, type_header))
    )
    return dataframe.to_csv(path_or_buf, sep="\t", encoding=encoding, index=False)
