import os

import pytest

from pyecotaxa.remote import Remote


@pytest.mark.slow
def test_pull(tmp_path):
    t = Remote()
    t.login("ecotaxa.api.user@gmail.com", "test!")

    target_directory = str(tmp_path / "download")
    os.makedirs(target_directory)

    t.pull(185, target_directory=target_directory)
