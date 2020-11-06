from io import StringIO

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pyecotaxa.archive import read_tsv, write_tsv


@pytest.mark.parametrize("enforce_types", [True, False])
@pytest.mark.parametrize("type_header", [True, False])
def test_read_tsv(enforce_types, type_header):
    if type_header:
        file_content = "a\tb\tc\td\n[t]\t[f]\t[t]\t[t]\n1\t2.0\ta\t\n3\t4.0\tb\t"
    else:
        file_content = "a\tb\tc\td\n1\t2.0\ta\t\n3\t4.0\tb\t"

    dataframe = read_tsv(StringIO(file_content), enforce_types=enforce_types)
    assert len(dataframe) == 2

    assert list(dataframe.columns) == ["a", "b", "c", "d"]

    if type_header and enforce_types:
        assert [dt.kind for dt in dataframe.dtypes] == ["O", "f", "O", "O"]
        assert_series_equal(dataframe["d"], pd.Series(["", ""]), check_names=False)
    else:
        assert [dt.kind for dt in dataframe.dtypes] == ["i", "f", "O", "f"]
        assert_series_equal(
            dataframe["d"], pd.Series([np.nan, np.nan]), check_names=False
        )


@pytest.mark.parametrize("type_header", [True, False])
def test_write_tsv(type_header):
    dataframe = pd.DataFrame(
        {"i": [1, 2, 3], "O": ["a", "b", "c"], "f": [1.0, 2.0, 3.0]}
    )

    content = write_tsv(dataframe, type_header=type_header)

    if type_header:
        assert content == "i\tO\tf\n[f]\t[t]\t[f]\n1\ta\t1.0\n2\tb\t2.0\n3\tc\t3.0\n"
    else:
        assert content == "i\tO\tf\n1\ta\t1.0\n2\tb\t2.0\n3\tc\t3.0\n"

    # Check round tripping
    dataframe2 = read_tsv(StringIO(content))
    assert_frame_equal(dataframe, dataframe2)


def test_empty_str_column():
    file_content = "a\tb\tc\n[t]\t[f]\t[t]\n\t2.0\ta"

    dataframe = read_tsv(StringIO(file_content), enforce_types=True)
    assert len(dataframe) == 1

    assert [dt.kind for dt in dataframe.dtypes] == ["O", "f", "O"]

    assert_series_equal(dataframe["a"], pd.Series([""]), check_names=False)
