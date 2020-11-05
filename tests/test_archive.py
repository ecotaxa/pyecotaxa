from pyecotaxa.archive import read_tsv
from inspect import cleandoc
from io import StringIO
from textwrap import dedent
import pytest


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

    with pytest.raises(ValueError, match="Unexpected type"):
        read_tsv(StringIO(file_content))