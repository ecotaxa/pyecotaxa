from io import StringIO

import pandas as pd
import pytest
from pyecotaxa.archive import read_tsv, write_tsv
from pandas.testing import assert_frame_equal, assert_series_equal


@pytest.mark.parametrize("enforce_types", [True, False])
def test_read_tsv(enforce_types):
    file_content = "a\tb\tc\n[t]\t[f]\t[t]\n1\t2.0\ta"

    dataframe = read_tsv(StringIO(file_content), enforce_types=enforce_types)
    assert len(dataframe) == 1

    if enforce_types:
        assert [dt.kind for dt in dataframe.dtypes] == ["O", "f", "O"]
    else:
        assert [dt.kind for dt in dataframe.dtypes] == ["i", "f", "O"]


def test_read_tsv_unexpected_type():
    file_content = "a\tb\tc\n[s]\t[f]\t[t]\n1\t2.0\ta"

    with pytest.raises(ValueError, match=r"Unexpected type: '\[s\]'"):
        read_tsv(StringIO(file_content))


def test_write_tsv():
    dataframe = pd.DataFrame(
        {"i": [1, 2, 3], "O": ["a", "b", "c"], "f": [1.0, 2.0, 3.0]}
    )

    content = write_tsv(dataframe)

    assert content == "i\tO\tf\n[f]\t[t]\t[f]\n1\ta\t1.0\n2\tb\t2.0\n3\tc\t3.0\n"

    # Check round tripping
    dataframe2 = read_tsv(StringIO(content))
    assert_frame_equal(dataframe, dataframe2)


def test_empty_str_column():
    file_content = "a\tb\tc\n[t]\t[f]\t[t]\n\t2.0\ta"

    dataframe = read_tsv(StringIO(file_content), enforce_types=True)
    assert len(dataframe) == 1

    assert [dt.kind for dt in dataframe.dtypes] == ["O", "f", "O"]

    assert_series_equal(dataframe["a"], pd.Series([""]), check_names=False)