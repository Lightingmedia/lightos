"""Cloud provider interface.

Every backend turns a catalog offer into a reachable node: it provisions
capacity, reports the address to SSH into, and tears the capacity back down.
The orchestrator only ever talks to this interface, so adding a cloud means
adding one subclass and one entry in `lightos.backends.get_provider`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProvisionedNode:
    """A node the orchestrator can sync code to and run commands on."""

    instance_id: str
    host_ip: Optional[str] = None
    username: str = "ubuntu"
    cloud: str = ""
    instance_type: str = ""
    region: str = ""
    hourly_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        where = f"{self.cloud}/{self.instance_type}"
        if self.region:
            where += f"@{self.region}"
        return f"{self.instance_id} ({where})"


class ProvisionError(RuntimeError):
    """Raised when capacity could not be acquired."""


class CloudProvider(abc.ABC):
    """Abstract base class for a compute backend."""

    #: Short name matching the `cloud` column in the cost catalog.
    name: str = ""

    @abc.abstractmethod
    def launch_instance(self, instance_type: str, job_id: str,
                        **kwargs: Any) -> str:
        """Provision one node and return its provider-side identifier."""

    @abc.abstractmethod
    def terminate_instance(self, instance_id: str) -> None:
        """Release the node. Must be safe to call twice."""

    def wait_until_ready(self, instance_id: str,
                         timeout: int = 600) -> ProvisionedNode:
        """Block until the node accepts connections.

        Backends that return an immediately-usable node may leave this as is.
        """
        return ProvisionedNode(instance_id=instance_id, cloud=self.name)

    # Convenience: the orchestrator calls this one.
    def provision(self, offer: Dict[str, Any], job_id: str,
                  **kwargs: Any) -> ProvisionedNode:
        """Launch capacity for a catalog offer and wait for it to come up."""
        instance_id = self.launch_instance(offer["instance_type"], job_id,
                                           region=offer.get("region"),
                                           **kwargs)
        node = self.wait_until_ready(instance_id)
        node.cloud = offer["cloud"]
        node.instance_type = offer["instance_type"]
        node.region = offer.get("region", "")
        node.hourly_cost = offer.get("hourly_cost", 0.0)
        return node
