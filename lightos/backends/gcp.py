"""Google Compute Engine backend."""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import CloudProvider, ProvisionedNode, ProvisionError

PROJECT_ENV = "LIGHTOS_GCP_PROJECT"
IMAGE_ENV = "LIGHTOS_GCP_IMAGE"
DEFAULT_IMAGE = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"


class GCPProvider(CloudProvider):
    """Provisions GCE instances labelled with the LightOS job id."""

    name = "gcp"

    def __init__(self,
                 project: Optional[str] = None,
                 zone: Optional[str] = None,
                 region: Optional[str] = None,
                 image: Optional[str] = None,
                 boot_disk_gb: int = 100):
        self.project = project or os.environ.get(PROJECT_ENV)
        # The catalog stores regions ("us-central1") but the Compute API takes
        # zones ("us-central1-a"), so derive one when only a region is known.
        self.zone = zone or self._default_zone(region)
        self.image = image or os.environ.get(IMAGE_ENV, DEFAULT_IMAGE)
        self.boot_disk_gb = boot_disk_gb
        self._client = None

    @staticmethod
    def _default_zone(region: Optional[str]) -> str:
        region = region or "us-central1"
        # A zone has three hyphen-separated parts ("us-central1-a"); a region
        # has two. Only append a zone letter when we were given a region.
        return region if region.count("-") >= 2 else f"{region}-a"

    @property
    def region(self) -> str:
        """The region the configured zone sits in."""
        return self.zone.rsplit("-", 1)[0]

    @property
    def client(self):
        if self._client is None:
            try:
                from google.cloud import compute_v1
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ProvisionError(
                    "google-cloud-compute is required for the GCP backend: "
                    "pip install google-cloud-compute") from exc
            self._client = compute_v1.InstancesClient()
        return self._client

    def launch_instance(self, instance_type: str, job_id: str,
                        **kwargs: Any) -> str:
        if not self.project:
            raise ProvisionError(
                f"No GCP project configured. Set {PROJECT_ENV} or pass project=.")
        from google.cloud import compute_v1

        name = f"lightos-{job_id}".lower()[:62]
        instance = compute_v1.Instance(
            name=name,
            machine_type=f"zones/{self.zone}/machineTypes/{instance_type}",
            disks=[
                compute_v1.AttachedDisk(
                    boot=True,
                    auto_delete=True,
                    initialize_params=compute_v1.AttachedDiskInitializeParams(
                        source_image=self.image,
                        disk_size_gb=self.boot_disk_gb,
                    ),
                )
            ],
            network_interfaces=[
                compute_v1.NetworkInterface(
                    name="global/networks/default",
                    access_configs=[
                        compute_v1.AccessConfig(name="External NAT",
                                                type_="ONE_TO_ONE_NAT")
                    ],
                )
            ],
            labels={"lightos-job": job_id.lower(), "lightos-managed": "true"},
        )

        print(f"[GCP] Launching {instance_type} in {self.zone}...")
        operation = self.client.insert(project=self.project, zone=self.zone,
                                       instance_resource=instance)
        operation.result(timeout=300)
        print(f"[GCP] Instance created: {name}")
        return name

    def wait_until_ready(self, instance_id: str,
                         timeout: int = 600) -> ProvisionedNode:
        instance = self.client.get(project=self.project, zone=self.zone,
                                   instance=instance_id)
        host_ip = None
        for nic in instance.network_interfaces:
            for access in nic.access_configs:
                if access.nat_i_p:
                    host_ip = access.nat_i_p
                    break
        if not host_ip:
            raise ProvisionError(
                f"{instance_id} has no external address; configure a NAT or "
                "run the orchestrator inside the VPC.")
        print(f"[GCP] Node ready at {host_ip}")
        return ProvisionedNode(instance_id=instance_id, host_ip=host_ip,
                               username="ubuntu", cloud=self.name,
                               region=self.zone)

    def terminate_instance(self, instance_id: str) -> None:
        print(f"[GCP] Deleting {instance_id}...")
        operation = self.client.delete(project=self.project, zone=self.zone,
                                       instance=instance_id)
        operation.result(timeout=300)
