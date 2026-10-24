"""The end-to-end pipeline: blueprint -> cheapest node -> run -> teardown."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .backends import CloudProvider, ProvisionedNode, get_provider
from .catalog import CostCatalog, estimate_cost, total_vram_gb
from .execution import (DEFAULT_REMOTE_WORKDIR, KubernetesJobExecutor,
                        RemoteJobExecutor)
from .task import Task

ProviderFactory = Callable[[str, Dict[str, Any]], CloudProvider]
ExecutorFactory = Callable[[ProvisionedNode, Task], Any]


@dataclass
class JobResult:
    """What a launch produced, whether or not the payload succeeded."""

    job_id: str
    task_name: str
    offer: Dict[str, Any]
    exit_code: Optional[int] = None
    instance_id: Optional[str] = None
    duration_seconds: float = 0.0
    files_synced: int = 0
    terminated: bool = False
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and self.error is None

    @property
    def estimated_cost(self) -> float:
        return estimate_cost(self.offer, self.duration_seconds / 3600.0)

    def summary(self) -> str:
        status = "SUCCEEDED" if self.succeeded else "FAILED"
        lines = [
            f"Job        : {self.job_id}",
            f"Task       : {self.task_name}",
            f"Status     : {status}"
            + (f" (exit {self.exit_code})" if self.exit_code is not None else ""),
            f"Cluster    : {self.offer['cloud']}/{self.offer['instance_type']}"
            f" @ {self.offer['region']} (${self.offer['hourly_cost']}/hr)",
            f"Duration   : {self.duration_seconds:.1f}s",
            f"Est. cost  : ${self.estimated_cost:.4f}",
            f"Teardown   : {'done' if self.terminated else 'NOT DONE'}",
        ]
        if self.error:
            lines.append(f"Error      : {self.error}")
        return "\n".join(lines)


def default_executor_factory(node: ProvisionedNode, task: Task):
    """Pick SSH or pod exec based on what the backend handed back."""
    if node.host_ip:
        # The blueprint wins over the backend's default login, so a custom
        # AMI or a key that is not at ~/.ssh/id_rsa is reachable.
        return RemoteJobExecutor(host_ip=node.host_ip,
                                 username=task.ssh_user or node.username,
                                 ssh_key_path=task.ssh_key_path)
    return KubernetesJobExecutor(
        pod_name=node.instance_id,
        namespace=node.metadata.get("namespace", "default"),
        container=node.metadata.get("container", "workload"))


class Orchestrator:
    """Coordinates the catalog, the cloud backends and the executors.

    `provider_factory` and `executor_factory` are injectable so the pipeline
    can be exercised end to end against mocks without touching a real cloud.
    """

    def __init__(self,
                 catalog: Optional[CostCatalog] = None,
                 provider_factory: Optional[ProviderFactory] = None,
                 executor_factory: Optional[ExecutorFactory] = None,
                 provider_kwargs: Optional[Dict[str, Any]] = None):
        self.catalog = catalog or CostCatalog()
        self.provider_factory = provider_factory or (
            lambda cloud, kwargs: get_provider(cloud, **kwargs))
        self.executor_factory = executor_factory or default_executor_factory
        self.provider_kwargs = provider_kwargs or {}

    # ------------------------------------------------------------------
    def select_offer(self, task: Task) -> Dict[str, Any]:
        """Stage 2: resolve the hardware request to the cheapest offer."""
        offer = self.catalog.find_cheapest_match(
            accelerator=task.accelerator,
            gpu_count=task.gpu_count,
            clouds=task.clouds,
            region=task.region,
            min_vram_gb=task.min_vram_gb)
        vram = total_vram_gb(offer)
        detail = f" {vram} GB VRAM" if vram else ""
        print(f"[Optimizer] Selected {offer['cloud'].upper()} "
              f"{offer['instance_type']} @ {offer['region']}"
              f"{detail} (${offer['hourly_cost']}/hr)")
        return offer

    def launch(self, task: Task, dry_run: bool = False) -> JobResult:
        """Run the whole REA loop for one blueprint."""
        print(f"\n[LightOS] Launching {task.describe()}")
        offer = self.select_offer(task)
        job_id = task.job_id()
        result = JobResult(job_id=job_id, task_name=task.name, offer=offer)

        if dry_run:
            print("[LightOS] --dry-run: stopping before provisioning.")
            result.extra["dry_run"] = True
            result.terminated = True
            return result

        # Every VM backend takes `region`; GCP derives its zone from it.
        provider_kwargs = dict(self.provider_kwargs)
        if offer.get("region") and offer["cloud"] != "k8s":
            provider_kwargs.setdefault("region", offer["region"])
        provider = self.provider_factory(offer["cloud"], provider_kwargs)

        node: Optional[ProvisionedNode] = None
        started = time.time()
        try:
            # Stage 3: provisioning.
            node = self._provision(provider, offer, job_id)
            result.instance_id = node.instance_id

            # Stage 4: sync and execute.
            executor = self.executor_factory(node, task)
            executor.connect()
            try:
                result.files_synced = executor.sync_workspace(
                    local_dir=task.workdir,
                    remote_destination=task.remote_workdir)
                result.exit_code = executor.execute_and_stream_logs(
                    bash_script=task.payload(),
                    run_directory=task.remote_workdir,
                    env=task.envs)
            finally:
                executor.disconnect()
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            print(f"[LightOS] Pipeline error: {result.error}")
        finally:
            result.duration_seconds = time.time() - started
            # Stage 5: teardown always runs; an orphaned GPU node is the one
            # failure mode that keeps costing money after we exit.
            if node is not None and task.autostop:
                result.terminated = self._teardown(provider, node)
            elif node is not None:
                print(f"[LightOS] autostop disabled; {node} left running.")

        print("\n" + result.summary())
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _provision(provider: CloudProvider, offer: Dict[str, Any],
                   job_id: str) -> ProvisionedNode:
        print(f"[Provision] Requesting {offer['instance_type']} "
              f"on {offer['cloud']}...")
        node = provider.provision(offer, job_id)
        print(f"[Provision] Node ready: {node}")
        return node

    @staticmethod
    def _teardown(provider: CloudProvider, node: ProvisionedNode) -> bool:
        try:
            provider.terminate_instance(node.instance_id)
            print(f"[Teardown] Released {node.instance_id}.")
            return True
        except Exception as exc:
            # Loud, because the user is now paying for a node we failed to kill.
            print(f"[Teardown] FAILED to release {node.instance_id}: {exc}\n"
                  f"[Teardown] Terminate it manually to stop charges.")
            return False


__all__ = ["Orchestrator", "JobResult", "default_executor_factory",
           "DEFAULT_REMOTE_WORKDIR"]
