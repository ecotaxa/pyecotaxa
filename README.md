# pyecotaxa

[![Documentation Status](https://readthedocs.org/projects/pyecotaxa/badge/?version=stable)](https://pyecotaxa.readthedocs.io/en/stable/?badge=stable)
[![Tests](https://github.com/ecotaxa/pyecotaxa/workflows/Tests/badge.svg)](https://github.com/ecotaxa/pyecotaxa/actions?query=workflow%3ATests)
[![PyPI](https://img.shields.io/pypi/v/pyecotaxa)](https://pypi.org/project/pyecotaxa)

Python package to query EcoTaxa and process its output.


## `pyecotaxa.remote`: Interact with a remote EcoTaxa server

```python
from pyecotaxa.remote import Remote

r = Remote()

# Login
r.login(username, password)

# Pull one or more project archives by project_id
r.pull(project_ids)
```
