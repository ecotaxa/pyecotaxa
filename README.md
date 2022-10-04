# pyecotaxa

[![Documentation Status](https://readthedocs.org/projects/pyecotaxa/badge/?version=stable)](https://pyecotaxa.readthedocs.io/en/stable/?badge=stable)
[![Tests](https://github.com/ecotaxa/pyecotaxa/workflows/Tests/badge.svg)](https://github.com/ecotaxa/pyecotaxa/actions?query=workflow%3ATests)
[![PyPI](https://img.shields.io/pypi/v/pyecotaxa)](https://pypi.org/project/pyecotaxa)

Python package to query EcoTaxa and process its output.

## `pyecotaxa.archive`: Read and write EcoTaxa archives

```python
from pyecotaxa.archive import read_tsv, write_tsv

# Read a .tsv file into a pandas DataFrame
# In contrast to pd.read_csv, this function transparently handles the type header
df = read_tsv(path_to_file)

# Write pandas DataFrame into a .tsv file
# In contrast to df.to_csv, this function can generate the type header
write_tsv(df, path_to_file)
```


## `pyecotaxa.remote`: Interact with a remote EcoTaxa server

```python
from pyecotaxa.remote import Remote

r = Remote()

# Login
r.login(username, password)

# Pull one or more project archives by project_id
r.pull(project_ids)
```
