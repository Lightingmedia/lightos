"""Remote execution and workspace synchronization.

Two executors share one surface so the orchestrator does not care where the
payload runs:

* `RemoteJobExecutor`  - SSH/SFTP to a VM (AWS, GCP, bare metal).
* `KubernetesJobExecutor` - `exec` into a pod, no SSH daemon required.

Both expose `connect()`, `sync_workspace()`, `execute_and_stream_logs()` and
`disconnect()`, and both stream stdout back to the caller's terminal as the
job produces it rather than buffering until exit.
"""

from __future__ import annotations

import fnmatch
import io
import os
import posixpath
import shutil
import subprocess
import sys
import tarfile
import time
from typing import IO, Iterable, List, Optional, Sequence

DEFAULT_REMOTE_WORKDIR = "/tmp/lightos_workdir"

# Never shipped to a remote node: caches, local virtualenvs, VCS metadata and
# anything holding credentials.
DEFAULT_EXCLUDES: Sequence[str] = (
    ".git", ".git/*", "*/.git/*",
    "__pycache__", "*/__pycache__/*", "*.pyc", "*.pyo",
    ".venv", ".venv/*", "venv", "venv/*", "*/site-packages/*",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "*.egg-info",
    ".env", ".env.*", "*.pem", "*.key", "id_rsa", "id_rsa.*",
    ".DS_Store", "node_modules", "node_modules/*",
)


class ExecutionError(RuntimeError):
    """Raised when a remote command fails or the connection cannot be made."""


def _is_excluded(relative_path: str, excludes: Sequence[str]) -> bool:
    normalized = relative_path.replace(os.sep, "/")
    parts = normalized.split("/")
    for pattern in excludes:
        if fnmatch.fnmatch(normalized, pattern):
            return True
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
    return False


def _iter_workspace_files(local_dir: str,
                          excludes: Sequence[str]) -> Iterable[tuple]:
    """Yield (absolute_path, posix_relative_path) for every file to upload."""
    local_dir = os.path.abspath(os.path.expanduser(local_dir))
    for root, dirnames, filenames in os.walk(local_dir):
        rel_root = os.path.relpath(root, local_dir)
        rel_root = "" if rel_root == "." else rel_root
        # Prune excluded directories in place so os.walk does not descend.
        dirnames[:] = [
            d for d in dirnames
            if not _is_excluded(posixpath.join(rel_root.replace(os.sep, "/"), d)
                                if rel_root else d, excludes)
        ]
        for filename in filenames:
            rel_path = (posixpath.join(rel_root.replace(os.sep, "/"), filename)
                        if rel_root else filename)
            if _is_excluded(rel_path, excludes):
                continue
            yield os.path.join(root, filename), rel_path


class RemoteJobExecutor:
    """Manages secure file transfer and real-time remote orchestration over SSH."""

    def __init__(self,
                 host_ip: str,
                 username: str = "ubuntu",
                 ssh_key_path: str = "~/.ssh/id_rsa",
                 port: int = 22,
                 connect_timeout: int = 15,
                 connect_retries: int = 10,
                 excludes: Sequence[str] = DEFAULT_EXCLUDES,
                 stdout: Optional[IO[str]] = None):
        self.host_ip = host_ip
        self.username = username
        self.ssh_key_path = os.path.expanduser(ssh_key_path)
        self.port = port
        self.connect_timeout = connect_timeout
        self.connect_retries = connect_retries
        self.excludes = tuple(excludes)
        self.stdout = stdout or sys.stdout
        self._ssh_client = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    @property
    def ssh_client(self):
        """Lazily constructed so paramiko is only needed when running jobs."""
        if self._ssh_client is None:
            try:
                import paramiko
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ExecutionError(
                    "paramiko is required for SSH execution: "
                    "pip install paramiko") from exc
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            # A freshly provisioned node has a host key we have never seen.
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self._ssh_client = client
        return self._ssh_client

    def connect(self) -> None:
        """Establish the SSH connection, retrying while the node boots."""
        self._log(f"[SSH] Connecting to {self.username}@{self.host_ip}:"
                  f"{self.port}...")
        if not os.path.exists(self.ssh_key_path):
            raise ExecutionError(f"SSH key not found at {self.ssh_key_path}.")

        last_error: Optional[Exception] = None
        for attempt in range(1, self.connect_retries + 1):
            try:
                self.ssh_client.connect(
                    hostname=self.host_ip,
                    port=self.port,
                    username=self.username,
                    key_filename=self.ssh_key_path,
                    timeout=self.connect_timeout,
                )
                self._log("[SSH] Connection authenticated successfully.")
                return
            except Exception as exc:  # sshd is often not up on first boot
                last_error = exc
                if attempt < self.connect_retries:
                    self._log(f"[SSH] Not ready ({exc}); retrying "
                              f"{attempt}/{self.connect_retries}...")
                    time.sleep(5)
        raise ExecutionError(
            f"Could not reach {self.host_ip} after {self.connect_retries} "
            f"attempts: {last_error}")

    def disconnect(self) -> None:
        """Close connection hooks gracefully."""
        if self._ssh_client is not None:
            self._ssh_client.close()
            self._ssh_client = None

    def __enter__(self) -> "RemoteJobExecutor":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Workspace synchronization
    # ------------------------------------------------------------------
    def sync_workspace(self, local_dir: str,
                       remote_destination: str = DEFAULT_REMOTE_WORKDIR,
                       prefer_rsync: bool = True) -> int:
        """Push the local workspace to the node. Returns the file count.

        Uses native `rsync` when it is installed (far faster on incremental
        code updates, which is what SkyPilot does) and falls back to SFTP so
        the path works on hosts without rsync, Windows included.
        """
        local_dir = os.path.abspath(os.path.expanduser(local_dir))
        if not os.path.isdir(local_dir):
            raise ExecutionError(f"Workspace {local_dir} is not a directory.")

        self._log(f"[Sync] Pushing [{local_dir}] to [{remote_destination}]...")
        if prefer_rsync and shutil.which("rsync"):
            count = self._sync_with_rsync(local_dir, remote_destination)
        else:
            count = self._sync_with_sftp(local_dir, remote_destination)
        self._log(f"[Sync] Workspace synchronized ({count} files).")
        return count

    def _sync_with_rsync(self, local_dir: str, remote_destination: str) -> int:
        excludes: List[str] = []
        for pattern in self.excludes:
            excludes.extend(["--exclude", pattern])
        ssh_cmd = (f"ssh -i {self.ssh_key_path} -p {self.port} "
                   f"-o StrictHostKeyChecking=no "
                   f"-o UserKnownHostsFile=/dev/null")
        command = [
            "rsync", "-az", "--delete", "-e", ssh_cmd, *excludes,
            f"{local_dir.rstrip(os.sep)}/",
            f"{self.username}@{self.host_ip}:{remote_destination}/",
        ]
        self.run(f"mkdir -p {remote_destination}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            self._log(f"[Sync] rsync failed ({result.stderr.strip()}); "
                      "falling back to SFTP.")
            return self._sync_with_sftp(local_dir, remote_destination)
        return sum(1 for _ in _iter_workspace_files(local_dir, self.excludes))

    def _sync_with_sftp(self, local_dir: str, remote_destination: str) -> int:
        sftp = self.ssh_client.open_sftp()
        created = {""}

        def ensure_remote_dir(path: str) -> None:
            if path in created:
                return
            parent = posixpath.dirname(path)
            if parent and parent not in created:
                ensure_remote_dir(parent)
            try:
                sftp.mkdir(path)
            except IOError:
                pass  # already exists
            created.add(path)

        ensure_remote_dir(remote_destination)
        count = 0
        try:
            for local_path, rel_path in _iter_workspace_files(local_dir,
                                                              self.excludes):
                remote_path = posixpath.join(remote_destination, rel_path)
                ensure_remote_dir(posixpath.dirname(remote_path))
                self._log(f" -> Syncing: {rel_path}")
                sftp.put(local_path, remote_path)
                count += 1
        finally:
            sftp.close()
        return count

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute_and_stream_logs(self, bash_script: str,
                                run_directory: str = DEFAULT_REMOTE_WORKDIR,
                                env: Optional[dict] = None) -> int:
        """Run commands remotely, piping output back live. Returns exit code."""
        self._log("\n--- [Remote Execution Starting] ---")

        exports = ""
        if env:
            exports = "".join(
                f"export {key}={_shell_quote(str(value))}; "
                for key, value in env.items())
        # `set -o pipefail` so a failure mid-pipeline is not masked by a
        # successful tail command.
        wrapped_command = (f"set -eo pipefail; cd {run_directory} && "
                           f"{exports}{bash_script}")

        transport = self.ssh_client.get_transport()
        if transport is None:
            raise ExecutionError("Not connected; call connect() first.")
        channel = transport.open_session()
        # Request a pseudo-terminal so tools with colored/progress output render.
        channel.get_pty()
        channel.exec_command(wrapped_command)

        try:
            self._pump_channel(channel)
            exit_code = channel.recv_exit_status()
        finally:
            channel.close()

        self._log(f"--- [Remote Job Finalized With Exit Code: {exit_code}] ---\n")
        return exit_code

    def _pump_channel(self, channel) -> None:
        """Drain the channel to our stdout until the command exits.

        `select` only accepts real sockets on Windows, so we select where it
        works and fall back to a short polling loop where it does not.
        """
        use_select = os.name == "posix"
        if use_select:
            import select

        while True:
            ready = True
            if use_select:
                read_ready, _, _ = select.select([channel], [], [], 0.1)
                ready = channel in read_ready
            if ready and channel.recv_ready():
                chunk = channel.recv(4096)
                if chunk:
                    self._write(chunk.decode("utf-8", errors="replace"))
                    continue  # keep draining while data is pending
            if channel.recv_stderr_ready():
                chunk = channel.recv_stderr(4096)
                if chunk:
                    self._write(chunk.decode("utf-8", errors="replace"))
                    continue
            if channel.exit_status_ready() and not channel.recv_ready():
                break
            if not use_select:
                time.sleep(0.05)

    def run(self, command: str, check: bool = False) -> tuple:
        """Run a short command and return (exit_code, stdout, stderr)."""
        _, stdout, stderr = self.ssh_client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if check and exit_code != 0:
            raise ExecutionError(
                f"Command failed ({exit_code}): {command}\n{err}")
        return exit_code, out, err

    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        self._write(message + "\n")

    def _write(self, text: str) -> None:
        self.stdout.write(text)
        self.stdout.flush()


class KubernetesJobExecutor:
    """Runs the payload inside a pod, mirroring RemoteJobExecutor's surface."""

    def __init__(self,
                 pod_name: str,
                 namespace: str = "default",
                 container: str = "workload",
                 excludes: Sequence[str] = DEFAULT_EXCLUDES,
                 stdout: Optional[IO[str]] = None):
        self.pod_name = pod_name
        self.namespace = namespace
        self.container = container
        self.excludes = tuple(excludes)
        self.stdout = stdout or sys.stdout
        self._api = None

    @property
    def api(self):
        if self._api is None:
            try:
                from kubernetes import client, config
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ExecutionError(
                    "kubernetes is required for pod execution: "
                    "pip install kubernetes") from exc
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            self._api = client.CoreV1Api()
        return self._api

    def connect(self) -> None:
        pod = self.api.read_namespaced_pod(name=self.pod_name,
                                           namespace=self.namespace)
        if pod.status.phase != "Running":
            raise ExecutionError(
                f"Pod {self.pod_name} is {pod.status.phase}, not Running.")
        self._log(f"[K8s] Attached to {self.namespace}/{self.pod_name}.")

    def disconnect(self) -> None:
        pass  # exec sessions are per-command; nothing to hold open.

    def __enter__(self) -> "KubernetesJobExecutor":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()

    def sync_workspace(self, local_dir: str,
                       remote_destination: str = DEFAULT_REMOTE_WORKDIR,
                       prefer_rsync: bool = False) -> int:
        """Stream the workspace in as a tar on stdin (what `kubectl cp` does)."""
        from kubernetes.stream import stream

        local_dir = os.path.abspath(os.path.expanduser(local_dir))
        self._log(f"[Sync] Streaming [{local_dir}] into pod path "
                  f"[{remote_destination}]...")

        buffer = io.BytesIO()
        count = 0
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for local_path, rel_path in _iter_workspace_files(local_dir,
                                                              self.excludes):
                tar.add(local_path, arcname=rel_path)
                count += 1
        buffer.seek(0)

        command = ["/bin/sh", "-c",
                   f"mkdir -p {remote_destination} && "
                   f"tar xzf - -C {remote_destination}"]
        resp = stream(self.api.connect_get_namespaced_pod_exec,
                      self.pod_name, self.namespace, container=self.container,
                      command=command, stderr=True, stdin=True,
                      stdout=True, tty=False, _preload_content=False)
        resp.write_stdin(buffer.read())
        resp.close()
        resp.run_forever(timeout=300)
        self._log(f"[Sync] Workspace synchronized ({count} files).")
        return count

    def execute_and_stream_logs(self, bash_script: str,
                                run_directory: str = DEFAULT_REMOTE_WORKDIR,
                                env: Optional[dict] = None) -> int:
        from kubernetes.stream import stream

        self._log("\n--- [Pod Execution Starting] ---")
        exports = ""
        if env:
            exports = "".join(
                f"export {key}={_shell_quote(str(value))}; "
                for key, value in env.items())
        wrapped = (f"set -eo pipefail; cd {run_directory} && "
                   f"{exports}{bash_script}")

        resp = stream(self.api.connect_get_namespaced_pod_exec,
                      self.pod_name, self.namespace, container=self.container,
                      command=["/bin/sh", "-c", wrapped],
                      stderr=True, stdin=False, stdout=True, tty=False,
                      _preload_content=False)
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                self._write(resp.read_stdout())
            if resp.peek_stderr():
                self._write(resp.read_stderr())
        exit_code = 0 if resp.returncode in (None, 0) else resp.returncode
        resp.close()
        self._log(f"--- [Pod Job Finalized With Exit Code: {exit_code}] ---\n")
        return exit_code

    def _log(self, message: str) -> None:
        self._write(message + "\n")

    def _write(self, text: str) -> None:
        self.stdout.write(text)
        self.stdout.flush()


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


# ==========================================
# Component validation simulator
# ==========================================
if __name__ == "__main__":
    # Point this at a reachable node to exercise the full path live.
    host = os.environ.get("LIGHTOS_TEST_HOST", "127.0.0.1")
    payload = ("echo 'Setting up environment...' && python3 --version && "
               "python3 -c 'print(\"Task execution complete!\")'")

    executor = RemoteJobExecutor(host_ip=host)
    try:
        executor.connect()
        executor.sync_workspace(local_dir="./lightos")
        code = executor.execute_and_stream_logs(bash_script=payload)
        sys.exit(code)
    except ExecutionError as exc:
        print(f"Pipeline stopped: {exc}")
        sys.exit(1)
    finally:
        executor.disconnect()
