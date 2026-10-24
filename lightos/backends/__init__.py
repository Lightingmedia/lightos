"""Cloud backends and the registry the orchestrator resolves them through."""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import CloudProvider, ProvisionedNode, ProvisionError

__all__ = [
    "CloudProvider",
    "ProvisionedNode",
    "ProvisionError",
    "get_provider",
    "register_provider",
    "available_providers",
]

# Imported lazily: a user with only AWS credentials should not need the GCP or
# Kubernetes client libraries installed to run an AWS job.
_PROVIDER_PATHS: Dict[str, tuple] = {
    "aws": ("lightos.backends.aws", "AWSProvider"),
    "gcp": ("lightos.backends.gcp", "GCPProvider"),
    "k8s": ("lightos.backends.k8s", "KubernetesProvider"),
}

_OVERRIDES: Dict[str, Type[CloudProvider]] = {}


def register_provider(name: str, provider_cls: Type[CloudProvider]) -> None:
    """Register a backend (or replace a built-in one, e.g. in tests)."""
    _OVERRIDES[name.lower()] = provider_cls


def available_providers() -> list:
    return sorted(set(_PROVIDER_PATHS) | set(_OVERRIDES))


def get_provider(cloud: str, **kwargs: Any) -> CloudProvider:
    """Instantiate the backend for a catalog `cloud` value."""
    key = cloud.lower()
    if key in _OVERRIDES:
        return _OVERRIDES[key](**kwargs)
    if key not in _PROVIDER_PATHS:
        raise ProvisionError(
            f"Unknown cloud '{cloud}'. Known backends: "
            f"{', '.join(available_providers())}")
    module_path, class_name = _PROVIDER_PATHS[key]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**kwargs)
