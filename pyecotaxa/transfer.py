import concurrent.futures
import glob
import os
import shutil
import time
import urllib.parse
import zipfile
from typing import Dict, List, Optional, Union
import getpass
import werkzeug

import dotenv
import requests
import tqdm

DEFAULT_EXPORTED_DATA_SHARE = "/remote/plankton_rw/ftp_plankton/Ecotaxa_Exported_data/"

# Read environment variables from .env
dotenv.load_dotenv()


class JobError(Exception):
    """Raised if a server-side job fails."""

    pass


def removeprefix(s: str, prefix: str) -> str:
    """Polyfill for str.removeprefix introduced in Python 3.9."""

    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s[:]


class Transfer:
    """
    Transfer data from and to an EcoTaxa server.

    Args:
        api_endpoint (str, optional): EcoTaxa API endpoint.
        api_token (str, optional): API token.
            If not given, it is read from the environment variable ECOTAXA_API_TOKEN.
            ECOTAXA_API_TOKEN can be given on the command line or defined in a .env file in the project root.
            If that does not succeed, the user is prompted for username and password.
        exported_data_share (str, optional): Locally accessible location for the results of an export job.
            If given, the archive will be copied, otherwise, it will be transferred over HTTP.
    """

    def __init__(
        self,
        *,
        api_endpoint="https://ecotaxa.obs-vlfr.fr/api/",
        api_token: Optional[str] = None,
        exported_data_share: Optional[str] = None,
        verbose=False,
    ):
        if api_endpoint[-1] != "/":
            api_endpoint = api_endpoint + "/"

        self.api_endpoint = api_endpoint

        if api_token is None:
            api_token = os.environ.get("ECOTAXA_API_TOKEN", None)

        self.api_token = api_token

        if exported_data_share is None:
            if os.path.isdir(DEFAULT_EXPORTED_DATA_SHARE):
                exported_data_share = DEFAULT_EXPORTED_DATA_SHARE
        self.exported_data_share = exported_data_share

        self.verbose = verbose

    def login(self, username: str, password: str):
        response = requests.post(
            urllib.parse.urljoin(self.api_endpoint, "login"),
            json={"password": password, "username": username},
        )

        self._check_response(response)

        API_TOKEN = response.text

    def login_interactive(self):
        username = input("Username: ")
        password = getpass.getpass()

        self.login(username, password)

    @property
    def auth_headers(self):
        if not self.api_token:
            raise ValueError("API token not set")

        return {"Authorization": f"Bearer {self.api_token}"}

    def _get_job(self, job_id) -> Dict:
        """Retrieve details about a job."""
        response = requests.get(
            urllib.parse.urljoin(self.api_endpoint, f"jobs/{job_id}"),
            headers=self.auth_headers,
        )

        self._check_response(response)

        return response.json()

    def _get_job_file_remote(self, job_id, *, target_directory: str) -> str:
        """Download an exported archive and return the local file name."""

        response = requests.get(
            urllib.parse.urljoin(self.api_endpoint, f"jobs/{job_id}/file"),
            params={},
            headers=self.auth_headers,
            stream=True,
        )

        self._check_response(response)

        _, options = werkzeug.http.parse_options_header(
            response.headers["content-disposition"]
        )
        filename = os.path.basename(options["filename"])

        dest = os.path.join(target_directory, filename)

        if self.verbose:
            print(f"Downloading {filename}...")

        try:
            with open(dest, "wb") as f:
                chunksize = 1024
                for chunk in tqdm.tqdm(response.iter_content(chunksize), desc=filename):
                    f.write(chunk)
        except:
            try:
                os.remove(dest)
            except FileNotFoundError:
                pass

        return dest

    def _get_job_file_local(self, job_id, *, target_directory: str) -> str:
        """Download an exported archive and return the local file name."""

        pattern = os.path.join(self.exported_data_share, f"task_{job_id}_*.zip")
        matches = glob.glob(pattern)

        if not matches:
            raise ValueError(
                f"No locally accessible export for job {job_id}.\nPattern: {pattern}"
            )

        remote_fn = matches[0]
        filename = os.path.basename(remote_fn)

        # Local filename should not have the task_<id>_ prefix to match get_job_file_remote
        dest = os.path.join(target_directory, removeprefix(filename, f"task_{job_id}_"))

        if self.verbose:
            print(f"Copying {filename} to {dest}...")
        shutil.copy(remote_fn, dest)

        return dest

    def _get_job_file(self, job_id, *, target_directory: str) -> str:
        if self.exported_data_share:
            return self._get_job_file_local(job_id, target_directory=target_directory)
        return self._get_job_file_remote(job_id, target_directory=target_directory)

    def _start_project_export(self, project_id):
        response = requests.post(
            urllib.parse.urljoin(self.api_endpoint, "object_set/export"),
            json={
                "filters": {},
                "request": {
                    "project_id": project_id,
                    "exp_type": "BAK",
                    "use_latin1": False,
                    "tsv_entities": "OPAS",
                    "split_by": "S",
                    "coma_as_separator": False,
                    "format_dates_times": False,
                    "with_images": True,
                    "with_internal_ids": False,
                    "only_first_image": False,
                    "sum_subtotal": "A",
                    "out_to_ftp": True,
                },
            },
            headers=self.auth_headers,
        )

        self._check_response(response)

        data = response.json()

        job_id = data["job_id"]
        print(f"Enqueued export job {job_id}.")

        # Get job data
        return self._get_job(job_id)

    def _get_jobs(self):
        response = requests.get(
            urllib.parse.urljoin(self.api_endpoint, "jobs"),
            params={"for_admin": False},
            headers=self.auth_headers,
        )

        self._check_response(response)

        return response.json()

    def _download_archive(self, project_id, *, target_directory: str) -> str:
        """Download exported archive."""

        # Find finished export task for project_id
        jobs = self._get_jobs()

        matches = [
            job
            for job in jobs
            if job.get("params", {}).get("req", {}).get("project_id") == project_id
        ]

        if not matches:
            if self.verbose:
                print(f"{project_id} is not yet exported.")

            job = self._start_project_export(project_id)
        else:
            job = matches[0]

        job_id = job["id"]

        # 'P' for Pending (Waiting for an execution thread)
        # 'R' for Running (Being executed inside a thread)
        # 'A' for Asking (Needing user information before resuming)
        # 'E' for Error (Stopped with error)
        # 'F' for Finished (Done)."
        while job["state"] not in "FE":
            print(f"Waiting for job {job_id} to finish...")
            time.sleep(10)

            # Update job data
            job = self._get_job(job_id)

        if job["state"] == "E":
            raise JobError(job["progress_msg"])

        # Download job file
        return self._get_job_file(job_id, target_directory=target_directory)

    def _check_archive(self, archive_fn) -> str:
        if self.verbose:
            print(f"Checking {archive_fn}...")

        try:
            with zipfile.ZipFile(archive_fn) as zf:
                zf.testzip()
        except Exception:
            if self.verbose:
                print(f"[ERROR] {archive_fn}")
            raise

        if self.verbose:
            print(f"[OK] {archive_fn}")

    def _cleanup_task_data(self, project_id):
        # Find finished export task for project_id
        jobs = self._get_jobs()

        matches = [
            job
            for job in jobs
            if job.get("params", {}).get("req", {}).get("project_id") == project_id
        ]

        for job in matches:
            job_id = job["id"]
            print(f"Cleaning up data for job {job_id}...")
            response = requests.delete(
                urllib.parse.urljoin(self.api_endpoint, f"jobs/{job_id}/"),
                headers=self.auth_headers,
            )

            self._check_response(response)

    def _get_archive_for_project(
        self,
        project_id: int,
        *,
        target_directory: str,
        check_integrity: bool,
        cleanup_task_data: bool,
    ) -> str:
        """Find and return the name of the local copy of the requested project."""

        pattern = os.path.join(target_directory, f"export_{project_id}_*.zip")
        matches = glob.glob(pattern)

        if matches:
            archive_fn = matches[0]
        else:
            if self.verbose:
                print(f"Project {project_id} is not locally available.")
            archive_fn = self._download_archive(
                project_id, target_directory=target_directory
            )

        if check_integrity:
            self._check_archive(archive_fn)

        if cleanup_task_data:
            self._cleanup_task_data(project_id)

        return archive_fn

    def _check_response(self, response: requests.Response):
        try:
            response.raise_for_status()
        except:
            if self.verbose:
                print("Request failed!")
                print("Request url:", response.request.method, response.request.url)
                print("Request headers:", response.request.headers)
                print("Response headers:", response.headers)
                print("Response text:", response.text)
            raise

    def pull(
        self,
        project_ids: Union[List[int], int],
        *,
        target_directory=".",
        n_parallel=1,
        check_integrity=True,
        cleanup_task_data=True,
    ) -> List[str]:
        """
        Export a project and transfer to a local directory.

        Args:
            project_ids (int or list): Project IDs to pull.
            target_directory (str, optional): Directory for exported archives.
            n_parallel (int, optional): Number of parallel tasks.
                More parallel tasks might speed up the pull but also put more load on the server.
            check_integrity (bool, optional): Ensure the integrity of the exported archive.
            cleanup_task_data (bool, optional): Clean up task data on the server after successful download.
        """

        if isinstance(project_ids, int):
            project_ids = [project_ids]

        executor = concurrent.futures.ThreadPoolExecutor(n_parallel)

        futures_iter = concurrent.futures.as_completed(
            [
                executor.submit(
                    self._get_archive_for_project,
                    project_id,
                    target_directory=target_directory,
                    check_integrity=check_integrity,
                    cleanup_task_data=cleanup_task_data,
                )
                for project_id in project_ids
            ]
        )

        if self.verbose:
            futures_iter = tqdm.tqdm(
                futures_iter,
                total=len(project_ids),
            )

        archive_fns = []
        for archive_fn_future in futures_iter:
            archive_fn = archive_fn_future.result()

            if self.verbose:
                print(f"Got {archive_fn}.")

            archive_fns.append(archive_fn)

        return archive_fns
