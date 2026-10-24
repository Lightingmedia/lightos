"""AWS EC2 backend (boto3)."""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import CloudProvider, ProvisionedNode, ProvisionError

# Ubuntu 22.04 LTS; override per region with LIGHTOS_AWS_AMI.
DEFAULT_AMI_ENV = "LIGHTOS_AWS_AMI"
DEFAULT_KEY_NAME_ENV = "LIGHTOS_AWS_KEY_NAME"


class AWSProvider(CloudProvider):
    """Provisions EC2 instances tagged with the LightOS job id."""

    name = "aws"

    def __init__(self,
                 region: str = "us-east-1",
                 ami: Optional[str] = None,
                 key_name: Optional[str] = None,
                 security_group_ids: Optional[list] = None,
                 subnet_id: Optional[str] = None):
        self.region = region
        self.ami = ami or os.environ.get(DEFAULT_AMI_ENV)
        self.key_name = key_name or os.environ.get(DEFAULT_KEY_NAME_ENV)
        self.security_group_ids = security_group_ids
        self.subnet_id = subnet_id
        self._client = None

    @property
    def client(self):
        """Lazily built so importing this module needs no credentials."""
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ProvisionError(
                    "boto3 is required for the AWS backend: pip install boto3"
                ) from exc
            self._client = boto3.client("ec2", region_name=self.region)
        return self._client

    def launch_instance(self, instance_type: str, job_id: str,
                        **kwargs: Any) -> str:
        if not self.ami:
            raise ProvisionError(
                f"No AMI configured. Set {DEFAULT_AMI_ENV} or pass ami=.")
        params = {
            "ImageId": self.ami,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "TagSpecifications": [{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"lightos-{job_id}"},
                    {"Key": "lightos:job", "Value": job_id},
                    {"Key": "lightos:managed", "Value": "true"},
                ],
            }],
        }
        if self.key_name:
            params["KeyName"] = self.key_name
        if self.security_group_ids:
            params["SecurityGroupIds"] = self.security_group_ids
        if self.subnet_id:
            params["SubnetId"] = self.subnet_id

        print(f"[AWS] Launching {instance_type} in {self.region}...")
        response = self.client.run_instances(**params)
        instance_id = response["Instances"][0]["InstanceId"]
        print(f"[AWS] Instance requested: {instance_id}")
        return instance_id

    def wait_until_ready(self, instance_id: str,
                         timeout: int = 600) -> ProvisionedNode:
        print(f"[AWS] Waiting for {instance_id} to pass status checks...")
        waiter = self.client.get_waiter("instance_running")
        waiter.wait(InstanceIds=[instance_id],
                    WaiterConfig={"Delay": 10,
                                  "MaxAttempts": max(1, timeout // 10)})
        described = self.client.describe_instances(InstanceIds=[instance_id])
        instance = described["Reservations"][0]["Instances"][0]
        host_ip = instance.get("PublicIpAddress") or instance.get(
            "PrivateIpAddress")
        if not host_ip:
            raise ProvisionError(
                f"{instance_id} came up without a reachable address.")
        print(f"[AWS] Node ready at {host_ip}")
        return ProvisionedNode(instance_id=instance_id, host_ip=host_ip,
                               username="ubuntu", cloud=self.name,
                               region=self.region)

    def terminate_instance(self, instance_id: str) -> None:
        print(f"[AWS] Terminating {instance_id}...")
        self.client.terminate_instances(InstanceIds=[instance_id])
