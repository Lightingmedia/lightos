"""Task blueprints: the YAML contract a user writes to request a job."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# Matches a value that is exactly one variable reference: ${VAR} or $VAR.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


class TaskError(ValueError):
    """Raised when a blueprint is missing or malformed."""


def _expand_envs(envs: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve `${VAR}` in env values from the launching shell.

    Lets a blueprint reference a secret by name - `HF_TOKEN: ${HF_TOKEN}` -
    so the value is never committed. An unset variable is an error rather
    than an empty export, which otherwise fails confusingly on the node.
    """
    if not isinstance(envs, dict):
        raise TaskError("'envs' must be a mapping.")

    resolved: Dict[str, Any] = {}
    for key, value in envs.items():
        if isinstance(value, str):
            match = _ENV_REF.fullmatch(value.strip())
            if match:
                name = match.group(1) or match.group(2)
                if name not in os.environ:
                    raise TaskError(
                        f"env '{key}' references ${{{name}}}, which is not "
                        f"set in this shell.")
                resolved[key] = os.environ[name]
                continue
        resolved[key] = value
    return resolved


@dataclass
class Task:
    """A parsed task blueprint.

    Mirrors the SkyPilot YAML shape:

        name: my-job
        workdir: .
        resources:
          accelerators: A10G
          gpu_count: 1
          cloud: aws          # optional hard constraint
          region: us-east-1   # optional hard constraint
        envs:
          HF_TOKEN: ...
        setup: |
          pip install torch
        run: |
          python train.py
    """

    name: str
    accelerator: Optional[str] = None
    gpu_count: int = 1
    cloud: Optional[str] = None
    region: Optional[str] = None
    setup: str = ""
    run: str = ""
    workdir: str = "."
    envs: Dict[str, Any] = field(default_factory=dict)
    remote_workdir: str = "/tmp/lightos_workdir"
    autostop: bool = True

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Task":
        if not isinstance(config, dict):
            raise TaskError("Task blueprint must be a YAML mapping.")
        if "name" not in config:
            raise TaskError("Task blueprint is missing required field 'name'.")

        resources = config.get("resources") or {}
        if not isinstance(resources, dict):
            raise TaskError("'resources' must be a mapping.")

        accelerator = resources.get("accelerators")
        gpu_count = resources.get("gpu_count")
        # `accelerators: A100:8` is the SkyPilot shorthand for 8x A100.
        if isinstance(accelerator, str) and ":" in accelerator:
            accelerator, _, count = accelerator.partition(":")
            try:
                gpu_count = int(count) if gpu_count is None else gpu_count
            except ValueError as exc:
                raise TaskError(
                    f"Could not read a GPU count from '{count}'.") from exc

        if gpu_count is None:
            gpu_count = 1 if accelerator else 0

        run = config.get("run", "")
        if not run.strip():
            raise TaskError(
                f"Task '{config['name']}' has no 'run' payload to execute.")

        clouds = resources.get("cloud")
        envs = _expand_envs(config.get("envs") or {})
        return cls(
            name=str(config["name"]),
            accelerator=accelerator,
            gpu_count=int(gpu_count),
            cloud=clouds.lower() if isinstance(clouds, str) else None,
            region=resources.get("region"),
            setup=config.get("setup", "") or "",
            run=run,
            workdir=config.get("workdir", "."),
            envs=envs,
            remote_workdir=config.get("remote_workdir",
                                      "/tmp/lightos_workdir"),
            autostop=bool(config.get("autostop", True)),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Task":
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise TaskError(f"Task blueprint not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        task = cls.from_dict(config)
        # workdir is relative to the blueprint, not the shell's cwd.
        if not os.path.isabs(task.workdir):
            task.workdir = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(path)),
                             task.workdir))
        return task

    # ------------------------------------------------------------------
    @property
    def clouds(self) -> Optional[List[str]]:
        """Cloud constraint in the form the catalog expects."""
        return [self.cloud] if self.cloud else None

    def job_id(self) -> str:
        """Unique, DNS-safe id used to tag cloud resources."""
        slug = "".join(c if c.isalnum() else "-" for c in self.name.lower())
        return f"{slug.strip('-')[:40]}-{uuid.uuid4().hex[:8]}"

    def payload(self) -> str:
        """setup and run concatenated into the single script we execute."""
        parts = [p.strip() for p in (self.setup, self.run) if p and p.strip()]
        return "\n".join(parts)

    def describe(self) -> str:
        hardware = (f"{self.gpu_count}x {self.accelerator}"
                    if self.accelerator else "CPU-only")
        return f"{self.name} [{hardware}]"
