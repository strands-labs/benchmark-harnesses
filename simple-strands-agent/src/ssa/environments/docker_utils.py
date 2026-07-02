from dataclasses import dataclass, field
import io
import json
import tarfile
from datasets import load_dataset, load_from_disk
from typing import List, Optional, Dict, Any
import docker
import os
import uuid
from docker.models.containers import Container, ExecResult
from docker.client import DockerClient
from enum import Enum
import logging
from omegaconf import DictConfig
import threading
import time
import tomllib

from ._streaming_exec import HeadTailBuffer

# Flake8 command configuration
FLAKE8_CMD = "pipx run flake8"

class DockerType(Enum):
    SBV = "SBV"
    SBPRO = "SBPRO"
    TB2 = "TB2"

DOCKER_TYPE_MAP: Dict[str, DockerType] = {
    "sbv": DockerType.SBV,
    "sbpro": DockerType.SBPRO,
    "tb2": DockerType.TB2,
}

HF_SWE_PRO: str = "ScaleAI/SWE-bench_Pro"
@dataclass
class DockerConfig:
    docker_type: DockerType
    base_image: str                      # full ECR image URI
    workdir: str = "/workspace"          # inside the container
    setup_commands: List[str] = field(default_factory=list)
    root_setup_commands: List[str] = field(default_factory=list)  # commands that run as root before user setup
    hold_command: str = "bash" # command to spin a process keeping docker live
    command_prefix: Optional[str] = None  # e.g. "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
    internet_access: bool = False # provide internet access for agent. Does not include static initial setup
    # Resource limits (optional, read from task.toml)
    cpus: Optional[int] = None           # Number of CPUs
    memory: Optional[str] = None         # Memory limit (e.g., "8G", "2048M")
    storage: Optional[str] = None        # Storage limit (e.g., "10G")
    use_resource_limits: bool = False     # Whether to apply resource limits from task.toml (if true, cpus, memory, and storage must be set)

# SPECIAL PROJECTS
PYTEST: str = "pytest"

LOG = logging.getLogger(__name__)
uid = os.getuid()
gid = os.getgid()


def parse_docker_config(cfg: DictConfig) -> DockerConfig:
    """
    Parse Docker configuration from Hydra config.

    Args:
        cfg: Hydra DictConfig containing env.docker section

    Returns:
        DockerConfig instance
    """
    docker_cfg = cfg.env.docker

    # Determine docker type based on dataset name
    dataset_name = cfg.dataset.get("name", "sbv").lower()
    docker_type = DOCKER_TYPE_MAP.get(dataset_name, "")

    base_image = docker_cfg.base_image
    workdir = docker_cfg.workdir
    # Update base_image compatible with instance_id of dataset
    instance_id = cfg.dataset.identifier
    if dataset_name == "sbv":
        splits = instance_id.split("__", 1)
        if os.getenv("ECR_DOCKER_IMAGE"):
            base_image = os.getenv("ECR_DOCKER_IMAGE")
        else:
            base_image = f"swebench/sweb.eval.x86_64.{splits[0]}_1776_{splits[1]}"
        workdir = "/testbed"
    elif dataset_name == "sbpro":
        if os.getenv("ECR_DOCKER_IMAGE"):
            base_image = os.getenv("ECR_DOCKER_IMAGE")
        else:
            cached_path = os.getenv("HF_SBPRO_DATASET_OFFLINE_LOCATION", "")
            if os.path.exists(cached_path):
                data= load_from_disk(cached_path)["test"]
            else:
                data = load_dataset(HF_SWE_PRO, split="test")
            instance = data.filter(lambda x: x["instance_id"] == instance_id)
            if len(instance) == 0:
                raise ValueError(f"Could not find swe-pro instance: {instance_id}")
            dockerhub_tag = instance[0].get("dockerhub_tag")
            base_image = f"jefzda/sweap-images:{dockerhub_tag}"
        workdir = "/app"
    elif dataset_name == "tb2":
        path_tb2_ecr_map = os.environ.get("TB2_ECR_MAP")
        with open(path_tb2_ecr_map, "r") as f:
            tb2_ecr_map = json.load(f)
        if instance_id not in tb2_ecr_map:
            raise ValueError(f"TB2 instance_id {instance_id} not found in TB2_ECR_MAP")
        base_image = tb2_ecr_map[instance_id]
        workdir = "/app"

    LOG.info(f"Overriding base_image: {base_image} and workdir: {workdir}")

    return DockerConfig(
        docker_type=docker_type,
        base_image=base_image,
        workdir=workdir,
        setup_commands=list(docker_cfg.get("setup_commands", [])),
        root_setup_commands=list(docker_cfg.get("root_setup_commands", [])),
        hold_command=docker_cfg.get("hold_command", "bash"),
        command_prefix=docker_cfg.get("command_prefix"),
        internet_access=docker_cfg.get("internet_access", False),
    )

# TODO: (vatshank) tb2 specific, move to another file
def parse_task_toml_config(task_toml_path: str) -> Dict[str, Any]:
    """
    Parse task.toml file and extract configuration including resource limits, agent timeout, and verifier settings.
    All TB2 projects are expected to have complete task.toml files with all required fields.

    Args:
        task_toml_path: Path to the task.toml file

    Returns:
        Dictionary with keys:
            - 'cpus': Number of CPUs
            - 'memory': Memory limit string
            - 'storage': Storage limit string
            - 'agent_timeout_sec': Agent timeout in seconds
            - 'verifier_timeout_sec': Verifier timeout in seconds

    Raises:
        FileNotFoundError: If task.toml doesn't exist
        ValueError: If required sections or fields are missing
    """
    if not os.path.exists(task_toml_path):
        raise FileNotFoundError(f"task.toml not found at {task_toml_path}")

    try:
        with open(task_toml_path, 'rb') as f:
            toml_data = tomllib.load(f)

        # Parse environment section (required)
        if 'environment' not in toml_data:
            raise ValueError(f"No [environment] section found in {task_toml_path}")

        env_config = toml_data['environment']
        cpus = env_config.get('cpus')
        memory = env_config.get('memory')
        storage = env_config.get('storage')

        # Fall back to *_mb variants
        if memory is None and env_config.get('memory_mb') is not None:
            memory = f"{env_config['memory_mb']}M"
        if storage is None and env_config.get('storage_mb') is not None:
            storage = f"{env_config['storage_mb']}M"

        # Validate required environment fields
        if cpus is None:
            raise ValueError(f"'cpus' not specified in [environment] section of {task_toml_path}")
        if memory is None:
            raise ValueError(f"'memory' not specified in [environment] section of {task_toml_path}")
        if storage is None:
            raise ValueError(f"'storage' not specified in [environment] section of {task_toml_path}")

        # Parse agent section (optional for backward compatibility)
        agent_timeout_sec = None
        if 'agent' in toml_data:
            agent_config = toml_data['agent']
            agent_timeout_sec = float(agent_config.get('timeout_sec'))

        # Parse verifier section (required)
        if 'verifier' not in toml_data:
            raise ValueError(f"No [verifier] section found in {task_toml_path}")

        verifier_config = toml_data['verifier']
        verifier_timeout_sec = int(verifier_config.get('timeout_sec'))

        if verifier_timeout_sec is None:
            raise ValueError(f"'timeout_sec' not specified in [verifier] section of {task_toml_path}")

        config = {
            'cpus': cpus,
            'memory': memory,
            'storage': storage,
            'agent_timeout_sec': agent_timeout_sec,
            'verifier_timeout_sec': verifier_timeout_sec,
        }

        LOG.info(f"Parsed task.toml config from {task_toml_path}: cpus={config['cpus']}, "
                f"memory={config['memory']}, storage={config['storage']}, "
                f"agent_timeout={config['agent_timeout_sec']}s, "
                f"verifier_timeout={config['verifier_timeout_sec']}s")

        return config

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse task.toml at {task_toml_path}: {e}") from e


class DockerConnector:
    def __init__(
        self,
        cfg: DockerConfig,
        host_mount: str,
        **kwargs,
    ):
        """
        :param cfg:          DockerConfig with image + CMD details
        :param host_mount:   Host directory to bind-mount to cfg.workdir,
                             or None for no mount.
        """
        self.cfg = cfg
        self.docker_type = cfg.docker_type
        self.docker_workdir = cfg.workdir
        self.host_mount = os.path.abspath(host_mount)
        self.client: DockerClient = self._connect_to_docker()
        self.internet_access: bool = cfg.internet_access
        self.container: Optional[Container] = None
        self.command_prefix = cfg.command_prefix
        self.setup_commands: List[str] = cfg.setup_commands
        self.identifier = kwargs.get("identifier", None)

    @staticmethod
    def _connect_to_docker() -> DockerClient:
        """
            Connect to the Docker daemon, failing fast with an actionable error.
        """
        try:
            # 10 minute timeout for large image pulls
            client = docker.from_env(timeout=600)
            client.ping()
        except docker.errors.DockerException as e:
            raise RuntimeError(
                "env_type=docker requires a running Docker daemon, but it could "
                "not be reached. Ensure Docker is installed and running "
                "(e.g. start Docker Desktop), then try again.\n"
                f"Underlying error: {e}"
            ) from e
        return client

    def run_root_setup_commands(self) -> None:
        """Run setup commands as root (e.g. apt-get install)."""
        for cmd in self.cfg.root_setup_commands:
            full_cmd = f"{self.command_prefix} && {cmd}" if self.command_prefix else cmd
            LOG.info(f"Executing root setup command: {full_cmd}")
            result = self.container.exec_run(
                cmd=["bash", "-c", full_cmd],
                user="root",
                workdir=self.cfg.workdir,
            )
            if result.exit_code != 0:
                LOG.warning(f"Root setup command failed (exit {result.exit_code}): {result.output.decode()}")
            else:
                LOG.info(f"Root setup command succeeded: {result.output.decode()}")

    def run_setup_commands(self) -> None:
        def _execute_cmd(cmd) -> ExecResult:
            # Each command needs to source conda environment using the command prefix
            full_cmd = f"{self.command_prefix} && {cmd}"
            LOG.info(f"Executing setup command: {full_cmd}")
            result = self.container.exec_run(
                cmd=["bash", "-c", full_cmd],
                # user=f"{uid}:{gid}",
                workdir=self.cfg.workdir
            )
            return result    

        # Run installation commands sequentially
        c_idx = 0
        while True:
            if c_idx >= len(self.setup_commands):
                break
            command = self.setup_commands[c_idx]
            result = _execute_cmd(command)          
            if result.exit_code != 0:
                LOG.info(f"Setup command failed with exit code {result.exit_code}")
                LOG.info(f"Setup command output: {result.output.decode()}")
                # Handle error - maybe cleanup and raise exception
                # self.container.stop()
                # self.container.remove()
                command = self._possible_setup_fallback(command)
                if command is not None:
                    result = _execute_cmd(command)
                    c_idx += 1
                    continue
                raise RuntimeError(f"Installation failed: {command}")
            else:
                LOG.info(f"Setup command succeeded: {result.output.decode()}")
            c_idx += 1

        LOG.info("All Setup commands completed successfully!")

    def init_docker(self) -> None:
        LOG.info(f"Pulling image: {self.cfg.base_image} ...")
        self.client.images.pull(self.cfg.base_image)
        LOG.info("Image pull completed")

        # Build a single Mount object only if host_mount is specified
        mounts = []
        # Run container detached
        hold_command = (self.command_prefix + " && " if self.command_prefix else "") + self.cfg.hold_command

        # Prepare container kwargs
        container_kwargs = {
            "image": self.cfg.base_image,
            "entrypoint": ["/bin/bash", "-lc"],
            "command": [hold_command,],
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "shm_size": "16g",
            "network_mode": "bridge",
            "working_dir": self.cfg.workdir,
            "mounts": mounts,
            "name": f"ssa_dkr_{uuid.uuid4().hex[:8]}",
            "environment": {
                "HOME": "/tmp",  # ISSUE: Java docker has Home path issue
                "GIT_AUTHOR_NAME":     "SimpleStrandsAgent",
                "GIT_AUTHOR_EMAIL":    "ssa@example.com",
                "GIT_COMMITTER_NAME":  "SimpleStrandsAgent",
                "GIT_COMMITTER_EMAIL": "ssa@example.com",
            },
        }

        # Apply resource limits if specified
        if self.cfg.cpus is not None:
            # Docker SDK requires nano_cpus (1 CPU = 1e9 nano_cpus)
            container_kwargs["nano_cpus"] = int(self.cfg.cpus * 1e9)
            LOG.info(f"Applying CPU limit: {self.cfg.cpus} CPUs")

        if self.cfg.memory is not None:
            # Docker SDK accepts memory limit as string (e.g., "8G", "2048M")
            container_kwargs["mem_limit"] = self.cfg.memory
            LOG.info(f"Applying memory limit: {self.cfg.memory}")

        if self.cfg.storage is not None:
            LOG.info(f"Storage limit from task.toml: {self.cfg.storage} (note: storage limits "
                    "require specific Docker storage driver configuration)")

        self.container = self.client.containers.run(**container_kwargs)

        # Wait for container to be ready
        LOG.info("Waiting for container status to change to 'running'...")
        self.container.reload()
        while self.container.status != 'running':
            time.sleep(1)
            self.container.reload()
        LOG.info("Container status: running.")

        if self.docker_type in (DockerType.SBV, DockerType.SBPRO):
            if self.cfg.root_setup_commands:
                LOG.info("Running root setup commands inside docker container ...")
                self.run_root_setup_commands()
            LOG.info("Running setup commands inside docker container ...")
            self.run_setup_commands()

            if not self.internet_access:
                # Disconnect from bridge network to disable internet access during execution
                try:
                    self.client.networks.get("bridge").disconnect(self.container)
                    LOG.info("Disconnected container from bridge network — internet access disabled.")
                except Exception as e:
                    LOG.warning(f"Failed to disconnect container from bridge network: {e}")
        else:
            # Install git if it is not present in the tb2 container
            # TODO: (better way to deal with this?) add emptying text file for init_csm's git commit to succeed even when the original repo has no files (e.g. empty repo or all files are in .dockerignore)
            self.exec("touch dummy_file_for_git_commit.txt", cwd=self.docker_workdir, verbose=True)            
            LOG.info("Checking git installation inside docker container for TB2...")
            exit_code, _ = self.exec("git --version", cwd="", verbose=True)
            if exit_code != 0:
                LOG.info("Git not found inside docker container. Installing git...")
                self.exec("apt-get update && apt-get install -y git", cwd=self.docker_workdir, verbose=True)
                # mark dir safe
                self.exec(f"git config --global --add safe.directory {self.docker_workdir}", cwd=self.docker_workdir, verbose=True)
            else:
                LOG.info("Git is already installed inside docker container.")
        LOG.info(f"Started docker container {self.container.id[:12]}")
        # raise Exception("Bye Bye Bye.")

    def exec(self, cmd: str, cwd: str, verbose: bool = False, timeout_sec: Optional[int] = None) -> str:
        if not self.container:
            raise RuntimeError("Container not started; call init_docker() first")
        start_time = time.time()

        if verbose:
            LOG.info(f"Cmd executed inside docker container: {cmd} in workdir: {cwd}")
        
        if self.command_prefix:
            cmd = f"{self.command_prefix} && {cmd}"
            if verbose:
                LOG.info(f"Cmd with prefix: {cmd}")

        result = {
            'completed': False,
            'exit_code': None,
            'output': None,
            'exception': None
        }

        def execute(cmd):
            buf = HeadTailBuffer()
            try:
                api = self.container.client.api
                # exec_create + exec_start(stream=True) so we can drain stdout
                # as it arrives.
                exec_id = api.exec_create(
                    self.container.id,
                    cmd,
                    workdir=cwd,
                    tty=False,
                    environment={"PYTHONUNBUFFERED": "1"},
                )["Id"]
                for chunk in api.exec_start(exec_id, stream=True):
                    if chunk:
                        buf.append(chunk)
                info = api.exec_inspect(exec_id)
                result.update({
                    "completed": True,
                    "exit_code": info.get("ExitCode", 1),
                    "output": buf.materialize(),
                })
            except Exception as e:
                result["exception"] = e
                result["exit_code"] = 1
                result["output"] = buf.materialize() + f"\n[exec error: {e}]"

        if timeout_sec is None:
            cmd = ["/bin/bash", "-lc", cmd]
            execute(cmd)
            exit_code = result["exit_code"]
        else:
            # -k sends SIGKILL if SIGTERM doesn't stop the process within 30s,
            # Uses short flag (-k) for portability across GNU coreutils and BusyBox (Alpine).
            # preventing orphaned child processes (e.g. node.js) from keeping the exec session alive.
            cmd = ["timeout", "-k", "30s", f"{timeout_sec}s", "/bin/bash", "-lc", cmd]
            # Python-level thread timeout guards against exec_run blocking beyond the expected duration
            # (e.g. when orphaned grandchild processes keep the session alive despite the shell being killed).
            thread = threading.Thread(target=execute, args=(cmd,), daemon=True)
            thread.start()
            thread.join(timeout=timeout_sec + 60)  # allow 60s grace beyond the in-container timeout
            if thread.is_alive():
                LOG.warning(
                    f"exec_run still blocked {timeout_sec + 60}s after start; "
                    "returning synthetic timeout exit code 124"
                )
                result["completed"] = False
                result["exit_code"] = 124
                # Inner thread never finished materializing; we have nothing
                # to report. In-container `timeout -k 30s` catches ~all stuck
                # processes, so this path is rare in practice.
                if result.get("output") is None:
                    result["output"] = ""
            exit_code = result["exit_code"]

        # TODO: (vatshc) Scan for incorrect workdir related exit codes (127, 128, what else?) and raise them here.
        
        # Convert paths in docker response to local paths
        # output = result["output"].replace(self.docker_workdir, self.host_mount)
        output = result["output"]
        if verbose:
            LOG.info(f"docker operation completed. Time taken={time.time()-start_time} sec")
        
        return exit_code, output

    def write_file(self, content: bytes, dest_path: str) -> None:
        """Write file content into the container using put_archive (no size limits)."""
        if not self.container:
            raise RuntimeError("Container not started; call init_docker() first")
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode='w') as tar:
            info = tarfile.TarInfo(name=os.path.basename(dest_path))
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        tarstream.seek(0)
        self.container.put_archive(os.path.dirname(dest_path), tarstream)

    def cleanup(self) -> None:
        if self.container:
            try:
                self.container.kill()  # Force stop
                self.container.remove(force=True)
            except Exception:
                LOG.warning("Container could not stop/removed on cleanup")
            finally:
                self.container = None
