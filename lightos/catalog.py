"""Multi-cloud hardware cost catalog and matchmaking engine.

The catalog answers one question: given a hardware request (accelerator type
and count), which cloud/instance_type/region combination satisfies it at the
lowest hourly cost?

In production the catalog is refreshed from live pricing APIs (the AWS Price
List API, the GCP Cloud Billing Catalog API, or a cluster's own node inventory
for Kubernetes). Those scrapers write a JSON snapshot which this module loads;
the bundled snapshot below keeps the engine usable offline and in tests.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

# Pricing snapshot. Each entry is one purchasable unit of capacity.
# (SkyPilot stores the equivalent data as per-cloud CSV files refreshed by
# scrapers; we keep a single normalized table.)
SIMULATED_CATALOG_DATA: List[Dict[str, Any]] = [
    # --- GPU capacity ---
    {"cloud": "aws", "instance_type": "g5.xlarge", "accelerator": "A10G",
     "gpus": 1, "hourly_cost": 1.006, "region": "us-east-1"},
    {"cloud": "aws", "instance_type": "g5.2xlarge", "accelerator": "A10G",
     "gpus": 1, "hourly_cost": 1.212, "region": "us-east-1"},
    {"cloud": "aws", "instance_type": "g5.12xlarge", "accelerator": "A10G",
     "gpus": 4, "hourly_cost": 5.672, "region": "us-east-1"},
    {"cloud": "aws", "instance_type": "p4d.24xlarge", "accelerator": "A100",
     "gpus": 8, "hourly_cost": 32.77, "region": "us-east-1"},
    {"cloud": "gcp", "instance_type": "a2-highgpu-1g", "accelerator": "A100",
     "gpus": 1, "hourly_cost": 3.67, "region": "us-central1"},
    {"cloud": "gcp", "instance_type": "a2-highgpu-8g", "accelerator": "A100",
     "gpus": 8, "hourly_cost": 29.39, "region": "us-central1"},
    {"cloud": "gcp", "instance_type": "g2-standard-4", "accelerator": "L4",
     "gpus": 1, "hourly_cost": 0.698, "region": "us-central1"},
    {"cloud": "k8s", "instance_type": "local-cluster", "accelerator": "A100",
     "gpus": 1, "hourly_cost": 0.00, "region": "default"},
    # --- CPU-only capacity (accelerator=None requests) ---
    {"cloud": "aws", "instance_type": "m6i.large", "accelerator": None,
     "gpus": 0, "hourly_cost": 0.096, "region": "us-east-1"},
    {"cloud": "gcp", "instance_type": "n2-standard-2", "accelerator": None,
     "gpus": 0, "hourly_cost": 0.097, "region": "us-central1"},
    {"cloud": "k8s", "instance_type": "local-cluster-cpu", "accelerator": None,
     "gpus": 0, "hourly_cost": 0.00, "region": "default"},
]

# Set LIGHTOS_CATALOG=/path/to/snapshot.json to use a scraped catalog instead.
CATALOG_PATH_ENV = "LIGHTOS_CATALOG"

_REQUIRED_FIELDS = ("cloud", "instance_type", "accelerator", "gpus",
                    "hourly_cost", "region")


class CatalogError(ValueError):
    """Raised when the catalog cannot satisfy a request or is malformed."""


class CostCatalog:
    """Searchable multi-cloud price database."""

    def __init__(self,
                 entries: Optional[Sequence[Dict[str, Any]]] = None,
                 catalog_path: Optional[str] = None):
        if entries is not None:
            self.db = [dict(e) for e in entries]
        else:
            path = catalog_path or os.environ.get(CATALOG_PATH_ENV)
            if path:
                self.db = self._load_snapshot(path)
            else:
                self.db = [dict(e) for e in SIMULATED_CATALOG_DATA]
        self._validate(self.db)

    # ------------------------------------------------------------------
    # Loading / refreshing
    # ------------------------------------------------------------------
    @staticmethod
    def _load_snapshot(path: str) -> List[Dict[str, Any]]:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        # Accept either a bare list or {"instances": [...]}.
        entries = payload["instances"] if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise CatalogError(f"Catalog at {path} is not a list of entries.")
        return [dict(e) for e in entries]

    @staticmethod
    def _validate(entries: Sequence[Dict[str, Any]]) -> None:
        for i, entry in enumerate(entries):
            missing = [f for f in _REQUIRED_FIELDS if f not in entry]
            if missing:
                raise CatalogError(
                    f"Catalog entry {i} is missing fields: {', '.join(missing)}")

    def to_json(self, path: str) -> str:
        """Persist the current table so a scraper run can be cached."""
        path = os.path.expanduser(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"instances": self.db}, fh, indent=2)
        return path

    # ------------------------------------------------------------------
    # Matchmaking
    # ------------------------------------------------------------------
    def find_matches(self,
                     accelerator: Optional[str],
                     gpu_count: int = 1,
                     clouds: Optional[Sequence[str]] = None,
                     region: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return every offer that satisfies the request, cheapest first.

        An offer satisfies the request when its accelerator matches
        (case-insensitively) and it carries at least `gpu_count` of them.
        Passing accelerator=None selects CPU-only capacity.
        """
        allowed = {c.lower() for c in clouds} if clouds else None
        wanted = accelerator.lower() if accelerator else None

        valid_options: List[Dict[str, Any]] = []
        for entry in self.db:
            entry_accel = entry["accelerator"]
            entry_accel = entry_accel.lower() if entry_accel else None
            if entry_accel != wanted:
                continue
            if entry["gpus"] < gpu_count:
                continue
            if allowed is not None and entry["cloud"].lower() not in allowed:
                continue
            if region is not None and entry["region"] != region:
                continue
            valid_options.append(entry)

        # Cheapest first; ties broken deterministically so repeated runs of the
        # same blueprint land on the same cluster.
        valid_options.sort(key=lambda x: (x["hourly_cost"], x["gpus"],
                                          x["cloud"], x["instance_type"]))
        return valid_options

    def find_cheapest_match(self,
                            accelerator: Optional[str],
                            gpu_count: int = 1,
                            clouds: Optional[Sequence[str]] = None,
                            region: Optional[str] = None) -> Dict[str, Any]:
        """Return the single lowest-cost offer for the request.

        Raises CatalogError (a ValueError) when nothing matches.
        """
        options = self.find_matches(accelerator, gpu_count, clouds, region)
        if not options:
            spec = f"{gpu_count}x {accelerator}" if accelerator else "CPU-only"
            constraint = ""
            if clouds:
                constraint += f" on {'/'.join(clouds)}"
            if region:
                constraint += f" in {region}"
            raise CatalogError(
                f"No available infrastructure found matching {spec}{constraint}.")
        return options[0]

    # ------------------------------------------------------------------
    # Introspection helpers (used by `lightos catalog`)
    # ------------------------------------------------------------------
    def accelerators(self) -> List[str]:
        return sorted({e["accelerator"] for e in self.db if e["accelerator"]})

    def clouds(self) -> List[str]:
        return sorted({e["cloud"] for e in self.db})

    def __len__(self) -> int:
        return len(self.db)

    def __repr__(self) -> str:
        return f"CostCatalog(entries={len(self.db)}, clouds={self.clouds()})"


def estimate_cost(offer: Dict[str, Any], hours: float) -> float:
    """Hourly price projected over a wall-clock duration."""
    return round(offer["hourly_cost"] * hours, 4)
