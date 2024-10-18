import json
import os
import collections.abc
from typing import Dict
import warnings

ENV_PREFIX = "PYECOTAXA_"


def find_file_recursive(filename: str) -> str:
    """Find a file from the current directory upwards."""
    curdir = os.getcwd()

    while True:
        path = os.path.join(curdir, filename)
        if os.path.isfile(path):
            return path

        # Do not cross fs boundaries
        if os.path.ismount(curdir):
            break

        # Move up
        curdir = os.path.dirname(curdir)

    # If the file was not found anywhere, assume it should lie in the current directory
    return os.path.join(os.getcwd(), filename)


def load_env(verbose=False):
    config = {
        k[len(ENV_PREFIX) :].lower(): v
        for k, v in os.environ.items()
        if k.startswith(ENV_PREFIX) and v
    }

    if verbose:
        print(f"ENV: {config}")

    return config


class JsonConfig(collections.abc.MutableMapping):
    def __init__(self, filename, verbose=False) -> None:
        self.filename = filename
        self._data = self._try_load()

        if verbose:
            print(f"{self.filename}: {self._data}")

    def _try_load(self) -> Dict:
        try:
            with open(self.filename, "r") as f:
                contents = f.read()

            return json.loads(contents)
        except FileNotFoundError:
            return {}

    def update(self, *args, **kwargs):
        self._data.update(*args, **kwargs)
        return self

    def setdefaults(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        data.update(self._data)
        self._data = data
        return self

    def update_from(self, path: str):
        other = JsonConfig(path)
        return self.update(other)

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self._data, f)

        return self

    # Mutable mapping interface
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return self._data.__iter__()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({self.filename}): {self._data}>"


def check_config(config):
    if config["api_endpoint"][-1] != "/":
        config["api_endpoint"] = config["api_endpoint"] + "/"

    if config["exported_data_share"] is not None and not os.path.isdir(
        config["exported_data_share"]
    ):
        warnings.warn(
            "exported_data_share"
            + config["exported_data_share"]
            + " does not exist, resetting"
        )
        config["exported_data_share"] = None

    return config


default_config = {
    "api_endpoint": "https://ecotaxa.obs-vlfr.fr/api/",
    "exported_data_share": None,
    "api_token": None,
}
