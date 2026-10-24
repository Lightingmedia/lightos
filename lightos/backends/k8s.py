"""Kubernetes backend: runs the payload in a pod instead of a VM.

Pods are reached with `kubectl exec` semantics rather than SSH, so this
backend reports no host_ip; the orchestrator routes it through
`lightos.execution.KubernetesJobExecutor`.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import CloudProvider, ProvisionedNode, ProvisionError

NAMESPACE_ENV = "LIGHTOS_K8S_NAMESPACE"
IMAGE_ENV = "LIGHTOS_K8S_IMAGE"
DEFAULT_IMAGE = "python:3.11-slim"


class KubernetesProvider(CloudProvider):
    """Creates a long-lived pod that the executor runs commands inside."""

    name = "k8s"

    def __init__(self,
                 namespace: Optional[str] = None,
                 image: Optional[str] = None,
                 gpu_count: int = 0,
                 service_account: Optional[str] = None):
        self.namespace = namespace or os.environ.get(NAMESPACE_ENV, "default")
        self.image = image or os.environ.get(IMAGE_ENV, DEFAULT_IMAGE)
        self.gpu_count = gpu_count
        self.service_account = service_account
        self._api = None

    @property
    def api(self):
        if self._api is None:
            try:
                from kubernetes import client, config
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ProvisionError(
                    "kubernetes is required for the K8s backend: "
                    "pip install kubernetes") from exc
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            self._api = client.CoreV1Api()
        return self._api

    def launch_instance(self, instance_type: str, job_id: str,
                        **kwargs: Any) -> str:
        from kubernetes import client

        pod_name = f"lightos-{job_id}".lower()[:62]
        resources = None
        if self.gpu_count:
            resources = client.V1ResourceRequirements(
                limits={"nvidia.com/gpu": str(self.gpu_count)})

        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=pod_name,
                labels={"lightos-job": job_id.lower(),
                        "lightos-managed": "true"},
            ),
            spec=client.V1PodSpec(
                restart_policy="Never",
                service_account_name=self.service_account,
                containers=[
                    client.V1Container(
                        name="workload",
                        image=self.image,
                        # Keep the pod alive so we can exec the payload into it.
                        command=["/bin/sh", "-c", "sleep infinity"],
                        resources=resources,
                    )
                ],
            ),
        )

        print(f"[K8s] Creating pod {pod_name} in namespace {self.namespace}...")
        self.api.create_namespaced_pod(namespace=self.namespace, body=pod)
        return pod_name

    def wait_until_ready(self, instance_id: str,
                         timeout: int = 600) -> ProvisionedNode:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            pod = self.api.read_namespaced_pod(name=instance_id,
                                               namespace=self.namespace)
            if pod.status.phase == "Running":
                print(f"[K8s] Pod {instance_id} is running.")
                return ProvisionedNode(
                    instance_id=instance_id,
                    host_ip=None,  # reached via kubectl exec, not SSH
                    cloud=self.name,
                    metadata={"namespace": self.namespace,
                              "container": "workload"},
                )
            if pod.status.phase in ("Failed", "Succeeded"):
                raise ProvisionError(
                    f"Pod {instance_id} entered {pod.status.phase} before use.")
            time.sleep(3)
        raise ProvisionError(
            f"Pod {instance_id} did not become ready within {timeout}s.")

    def terminate_instance(self, instance_id: str) -> None:
        print(f"[K8s] Deleting pod {instance_id}...")
        self.api.delete_namespaced_pod(name=instance_id,
                                       namespace=self.namespace)
