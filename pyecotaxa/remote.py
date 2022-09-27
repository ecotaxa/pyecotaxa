import concurrent.futures
import enum
import getpass
import glob
import os
import shutil
import time
import urllib.parse
import warnings
import zipfile
from typing import Dict, List, Optional, Union

import dotenv
import requests
import semantic_version
import tqdm
import werkzeug
from atomicwrites import atomic_write as _atomic_write

DEFAULT_EXPORTED_DATA_SHARE = "/remote/plankton_rw/ftp_plankton/Ecotaxa_Exported_data/"

# Read environment variables from .env
dotenv.load_dotenv()


def atomic_write(path, **kwargs):
    prefix = f".{os.path.basename(path)}-"
    suffix = ".part"
    return _atomic_write(path, mode="wb", prefix=prefix, suffix=suffix, **kwargs)


class JobError(Exception):
    """Raised if a server-side job fails."""

    pass


class ApiVersionWarning(Warning):
    pass


def removeprefix(s: str, prefix: str) -> str:
    """Polyfill for str.removeprefix introduced in Python 3.9."""

    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s[:]


def copyfile_progress(src, dst, chunksize=1024 ** 2):
    """Copy data from src to dst with progress"""

    with open(src, "rb") as fsrc:
        total = os.fstat(fsrc.fileno()).st_size

        print(src, "size", total)

        with atomic_write(dst) as fdst:
            n_written = 0
            while 1:
                buf = fsrc.read(chunksize)
                if not buf:
                    break
                fdst.write(buf)

                n_written += len(buf)

                yield n_written, total


class State(enum.Enum):
    FINISHED = 0
    RUNNING = 1
    FAILED = 2
    WAITING = 3


class ProgressListener:
    def __init__(self) -> None:
        self.progress_bars = {}

    def update(
        self,
        target,
        state: Optional[State] = None,
        message: Optional[str] = None,
        description: Optional[str] = None,
        progress: Optional[int] = None,
        total: Optional[int] = None,
        unit: Optional[str] = None,
    ):
        try:
            progress_bar = self.progress_bars[target]
        except KeyError:
            progress_bar = self.progress_bars[target] = tqdm.tqdm(
                position=0 if target is None else None, unit_scale=True
            )

        if progress_bar.disable:
            # Progress bar is already closed
            return

        if message is not None:
            if target is not None:
                message = f"{target}: {message}"
            progress_bar.write(message)

        if description is not None:
            if target is not None:
                description = f"{target}: {description}"
            progress_bar.set_description(description, refresh=False)

        if progress is not None:
            progress_bar.n = progress

        if total is not None:
            progress_bar.total = total

        if unit is not None:
            progress_bar.unit = unit
        else:
            progress_bar.unit = "it"

        if state == State.FINISHED:
            progress_bar.close()
        else:
            progress_bar.refresh()


class Obervable:
    def __init__(self) -> None:
        self.__observers = []

    def register_observer(self, fn):
        self.__observers.append(fn)
        return fn

    def _notify_observers(self, *args, **kwargs):
        for fn in self.__observers:
            fn(*args, **kwargs)


class Remote(Obervable):
    """
    Interact with a remote EcoTaxa server.

    Args:
        api_endpoint (str, optional): EcoTaxa API endpoint.
        api_token (str, optional): API token.
            If not given, it is read from the environment variable ECOTAXA_API_TOKEN.
            ECOTAXA_API_TOKEN can be given on the command line or defined in a .env file in the project root.
            If that does not succeed, the user is prompted for username and password.
        exported_data_share (str or bool, optional): Locally accessible location for the results of an export job.
            If given, the archive will be copied, otherwise, it will be transferred over HTTP.
            If not given, the default location (/remote/plankton_rw/ftp_plankton/Ecotaxa_Exported_data/) will be used if available.
            To disable the detection of the default location, pass False.
    """

    REQUIRED_OPENAPI_VERSION = "~=0.0.25"

    def __init__(
        self,
        *,
        api_endpoint="https://ecotaxa.obs-vlfr.fr/api/",
        api_token: Optional[str] = None,
        exported_data_share: Union[None, str, bool] = None,
    ):
        super().__init__()

        if api_endpoint[-1] != "/":
            api_endpoint = api_endpoint + "/"

        self.api_endpoint = api_endpoint

        if api_token is None:
            api_token = os.environ.get("ECOTAXA_API_TOKEN", None)

        self.api_token = api_token

        if exported_data_share is None or exported_data_share is True:
            if os.path.isdir(DEFAULT_EXPORTED_DATA_SHARE):
                exported_data_share = DEFAULT_EXPORTED_DATA_SHARE
            else:
                exported_data_share = None
        self.exported_data_share = exported_data_share

        self._check_version()

    def _check_version(self):
        response = requests.get(
            urllib.parse.urljoin(self.api_endpoint, "openapi.json"),
        )

        self._check_response(response)

        openapi_schema = response.json()

        version = openapi_schema.get("info", {}).get("version", "0.0.0")

        self._notify_observers(None, message=f"Server OpenAPI version is {version}")

        if semantic_version.Version(version) not in semantic_version.SimpleSpec(
            self.REQUIRED_OPENAPI_VERSION
        ):
            warnings.warn(
                f"Required OpenAPI version is {self.REQUIRED_OPENAPI_VERSION}, Server has {version}.",
                category=ApiVersionWarning,
                stacklevel=2,
            )

    def login(self, username: str, password: str):
        response = requests.post(
            urllib.parse.urljoin(self.api_endpoint, "login"),
            json={"password": password, "username": username},
        )

        self._check_response(response)

        self.api_token = response.json()

        self._notify_observers(None, message="Logged in successfully.")

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
            urllib.parse.urljoin(self.api_endpoint, f"jobs/{job_id}/"),
            headers=self.auth_headers,
        )

        self._check_response(response)

        return response.json()

    def _get_job_file_remote(self, project_id, job_id, *, target_directory: str) -> str:
        """Download an exported archive and return the local file name."""

        response = requests.get(
            urllib.parse.urljoin(self.api_endpoint, f"jobs/{job_id}/file"),
            params={},
            headers=self.auth_headers,
            stream=True,
        )

        self._check_response(response)

        content_length = int(response.headers.get("Content-Length", 0)) or None

        chunksize = 1024

        _, options = werkzeug.http.parse_options_header(
            response.headers["content-disposition"]
        )
        filename = os.path.basename(options["filename"])

        dest = os.path.join(target_directory, filename)

        try:
            self._notify_observers(
                project_id, description=f"Downloading...", progress=0, total=1
            )
            with atomic_write(dest) as f:
                total = (
                    content_length / chunksize if content_length is not None else None
                )
                progress = 0
                for chunk in response.iter_content(chunksize):
                    f.write(chunk)
                    progress += len(chunk)
                    self._notify_observers(
                        project_id,
                        description=f"Downloading...",
                        progress=progress,
                        total=content_length,
                        unit="iB",
                    )
                self._notify_observers(
                    project_id,
                    description=f"Downloading...",
                    progress=progress,
                    total=progress,
                    unit="iB",
                )
        except:
            # Cleaup destination file
            try:
                os.remove(dest)
            except FileNotFoundError:
                pass

            raise

        return dest

    def _get_job_file_local(self, project_id, job_id, *, target_directory: str) -> str:
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

        try:
            for n_written, total in copyfile_progress(remote_fn, dest):
                self._notify_observers(
                    project_id,
                    description=f"Copying...",
                    progress=n_written,
                    total=total,
                    unit="iB",
                )

            shutil.copymode(remote_fn, dest)
        except:
            # Cleaup destination file
            try:
                os.remove(dest)
            except FileNotFoundError:
                pass

        return dest

    def _get_job_file(self, project_id, job, *, target_directory: str) -> str:
        job_id = job["id"]

        out_to_ftp = job.get("params", {}).get("req", {}).get("out_to_ftp", False)

        if self.exported_data_share and out_to_ftp:
            return self._get_job_file_local(
                project_id, job_id, target_directory=target_directory
            )
        return self._get_job_file_remote(
            project_id, job_id, target_directory=target_directory
        )

    def _start_project_export(self, project_id, *, with_images):
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
                    "with_images": with_images,
                    "with_internal_ids": False,
                    "only_first_image": False,
                    "sum_subtotal": "A",
                    "out_to_ftp": self.exported_data_share is not None,
                },
            },
            headers=self.auth_headers,
        )

        self._check_response(response)

        data = response.json()

        job_id = data["job_id"]

        self._notify_observers(
            project_id,
            description=f"Enqueued export job.",
            progress=0,
            total=100,
            state=State.WAITING,
        )

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

    def _download_archive(
        self, project_id, *, target_directory: str, with_images: bool
    ) -> str:
        """Export and download project."""

        # Find finished export task for project_id
        jobs = self._get_jobs()

        matches = [
            job
            for job in jobs
            if job.get("params", {}).get("req", {}).get("project_id") == project_id
        ]

        if not matches:
            job = self._start_project_export(project_id, with_images=with_images)
        else:
            job = matches[0]

        job_id = job["id"]

        # 'P' for Pending (Waiting for an execution thread)
        # 'R' for Running (Being executed inside a thread)
        # 'A' for Asking (Needing user information before resuming)
        # 'E' for Error (Stopped with error)
        # 'F' for Finished (Done)."
        while job["state"] not in "FE":
            self._notify_observers(
                project_id,
                description=f"Exporting ({job['progress_msg']})...",
                progress=job["progress_pct"],
                total=100,
                state=State.RUNNING,
                unit="%",
            )

            time.sleep(5)

            # Update job data
            job = self._get_job(job_id)

        if job["state"] == "E":
            raise JobError(job["progress_msg"])

        # Download job file
        return self._get_job_file(project_id, job, target_directory=target_directory)

    def _check_archive(self, project_id, archive_fn) -> str:
        self._notify_observers(
            project_id, description=f"Checking archive...", progress=0, total=1
        )

        try:
            with zipfile.ZipFile(archive_fn) as zf:
                zf.testzip()
        except Exception:
            self._notify_observers(
                project_id, description=f"Checking archive...", state=State.FAILED
            )
            raise

        self._notify_observers(
            project_id, description=f"Checking archive...", progress=1, total=1
        )

    def _cleanup_task_data(self, project_id):
        # Find finished export task for project_id

        self._notify_observers(
            project_id,
            description=f"Cleaning up...",
            progress=0,
            total=1,
            state=State.RUNNING,
        )

        jobs = self._get_jobs()

        matches = [
            job
            for job in jobs
            if job.get("params", {}).get("req", {}).get("project_id") == project_id
        ]

        for job in matches:
            job_id = job["id"]

            response = requests.delete(
                urllib.parse.urljoin(self.api_endpoint, f"jobs/{job_id}"),
                headers=self.auth_headers,
            )

            self._check_response(response)

        self._notify_observers(
            project_id,
            description=f"Cleaning up...",
            progress=1,
            total=1,
            state=State.RUNNING,
        )

    def _get_archive_for_project(
        self,
        project_id: int,
        *,
        target_directory: str,
        check_integrity: bool,
        cleanup_task_data: bool,
        with_images: bool,
    ) -> str:
        """Find and return the name of the local copy of the requested project."""

        try:
            pattern = os.path.join(target_directory, f"export_{project_id}_*.zip")
            matches = glob.glob(pattern)

            if matches:
                archive_fn = matches[0]
            else:
                archive_fn = self._download_archive(
                    project_id,
                    target_directory=target_directory,
                    with_images=with_images,
                )

            if check_integrity:
                self._check_archive(project_id, archive_fn)

            if cleanup_task_data:
                self._cleanup_task_data(project_id)
        except Exception as exc:
            self._notify_observers(
                project_id,
                description=f"FAILED ({exc})",
                progress=1,
                total=1,
                state=State.FINISHED,
            )
            return None

        self._notify_observers(
            project_id, description="OK", progress=1, total=1, state=State.FINISHED
        )

        return archive_fn

    def _check_response(self, response: requests.Response):
        try:
            response.raise_for_status()
        except:
            self._notify_observers(
                None,
                message="\n".join(
                    [
                        "Request failed!",
                        f"Request url: {response.request.method} {response.request.url}",
                        f"Request headers: {response.request.headers}",
                        f"Response headers: {response.headers}",
                        f"Response text: {response.text}",
                    ]
                ),
            )
            raise

    def pull(
        self,
        project_ids: Union[List[int], int],
        *,
        target_directory=".",
        n_parallel=1,
        check_integrity=True,
        cleanup_task_data=True,
        with_images=True,
    ) -> List[str]:
        """
        Export a project and transfer to a local directory.

        Args:
            project_ids (int or list): Project IDs to pull.
            target_directory (str, optional): Directory for exported archives.
            n_parallel (int, optional): Number of projects to be processed in parallel (export jobs, file transfer, ...).
                More parallel tasks might speed up the pull but also put more load on the server.
            check_integrity (bool, optional): Ensure the integrity of the exported archive.
            cleanup_task_data (bool, optional): Clean up task data on the server after successful download.
            with_images (bool, optional): Include images in the exported archive.
        """

        if isinstance(project_ids, int):
            project_ids = [project_ids]

        os.makedirs(target_directory, exist_ok=True)

        executor = concurrent.futures.ThreadPoolExecutor(n_parallel)

        self._notify_observers(
            None, description="Pulling projects...", total=len(project_ids), unit="proj"
        )

        futures = [
            executor.submit(
                self._get_archive_for_project,
                project_id,
                target_directory=target_directory,
                check_integrity=check_integrity,
                cleanup_task_data=cleanup_task_data,
                with_images=with_images,
            )
            for project_id in project_ids
        ]

        try:
            archive_fns = []
            for i, archive_fn_future in enumerate(
                concurrent.futures.as_completed(futures)
            ):
                archive_fn = archive_fn_future.result()

                self._notify_observers(None, progress=i + 1, unit="proj")

                archive_fns.append(archive_fn)

            return archive_fns
        except:
            executor.shutdown(False)
            raise
