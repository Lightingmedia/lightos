"""Multi-cloud hardware cost catalog and matchmaking engine.

The catalog answers one question: given a hardware request, which
cloud/instance_type/region combination satisfies it at the lowest hourly cost?

A request can be expressed two ways, and for LLM work the second matters more:

* by accelerator -- "1x A100", the classic form.
* by memory -- "at least 160 GB of VRAM", because what decides whether a
  model runs is aggregate GPU memory, not the part number. A 70B model at
  bf16 needs ~140 GB of weights plus KV cache, which 2x H100 80GB satisfies
  and 4x L4 24GB does not, however the price compares.

Coverage spans hyperscalers (aws, gcp), Kubernetes, and neoclouds -- GPU
specialists that are routinely several times cheaper for the same silicon,
which is the whole reason a cost optimizer is worth having.

PRICES IN THE BUNDLED SNAPSHOT ARE APPROXIMATE AND GO STALE. They are
third-party published figures (see `price_as_of`), not quotes, and tiers
differ per provider -- RunPod Community vs Secure, spot vs on-demand,
committed-use discounts. Refresh from live APIs before trusting a decision
to them: a cost optimizer fed stale prices does not fail loudly, it quietly
picks the wrong cloud. `CostCatalog.stale_entries()` reports the age, and
`lightos catalog` warns when the snapshot is old.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

# Pricing snapshot. Each entry is one purchasable unit of capacity.
# (SkyPilot stores the equivalent data as per-cloud CSV files refreshed by
# scrapers; we keep a single normalized table.)
SIMULATED_CATALOG_DATA: List[Dict[str, Any]] = [
    # ---------------- Hyperscalers: AWS ----------------
    # Inherited from the original project spec; verify before relying on them.
    {"cloud": "aws", "instance_type": "g5.xlarge", "accelerator": "A10G",
     "gpus": 1, "gpu_memory_gb": 24, "hourly_cost": 1.006,
     "region": "us-east-1", "price_as_of": "2024-01-01"},
    {"cloud": "aws", "instance_type": "g5.12xlarge", "accelerator": "A10G",
     "gpus": 4, "gpu_memory_gb": 24, "hourly_cost": 5.672,
     "region": "us-east-1", "price_as_of": "2024-01-01"},
    {"cloud": "aws", "instance_type": "p4d.24xlarge", "accelerator": "A100",
     "gpus": 8, "gpu_memory_gb": 40, "hourly_cost": 32.77,
     "region": "us-east-1", "price_as_of": "2024-01-01"},
    {"cloud": "aws", "instance_type": "p5.48xlarge", "accelerator": "H100",
     "gpus": 8, "gpu_memory_gb": 80, "hourly_cost": 98.32,
     "region": "us-east-1", "price_as_of": "2026-10-23"},

    # ---------------- Hyperscalers: GCP ----------------
    {"cloud": "gcp", "instance_type": "a2-highgpu-1g", "accelerator": "A100",
     "gpus": 1, "gpu_memory_gb": 40, "hourly_cost": 3.67,
     "region": "us-central1", "price_as_of": "2024-01-01"},
    {"cloud": "gcp", "instance_type": "a2-highgpu-8g", "accelerator": "A100",
     "gpus": 8, "gpu_memory_gb": 40, "hourly_cost": 29.39,
     "region": "us-central1", "price_as_of": "2024-01-01"},
    {"cloud": "gcp", "instance_type": "g2-standard-4", "accelerator": "L4",
     "gpus": 1, "gpu_memory_gb": 24, "hourly_cost": 0.698,
     "region": "us-central1", "price_as_of": "2024-01-01"},

    # ---------------- Neoclouds ----------------
    # RunPod Community tier; Secure Cloud is roughly 25-45% higher.
    {"cloud": "runpod", "instance_type": "L40S-48G", "accelerator": "L40S",
     "gpus": 1, "gpu_memory_gb": 48, "hourly_cost": 0.79,
     "region": "community", "price_as_of": "2026-10-23"},
    {"cloud": "runpod", "instance_type": "A100-80G", "accelerator": "A100",
     "gpus": 1, "gpu_memory_gb": 80, "hourly_cost": 1.39,
     "region": "community", "price_as_of": "2026-10-23"},
    {"cloud": "runpod", "instance_type": "H100-PCIe-80G", "accelerator": "H100",
     "gpus": 1, "gpu_memory_gb": 80, "hourly_cost": 1.99,
     "region": "community", "price_as_of": "2026-10-23"},
    {"cloud": "runpod", "instance_type": "H100-SXM-80G", "accelerator": "H100",
     "gpus": 1, "gpu_memory_gb": 80, "hourly_cost": 2.69,
     "region": "community", "price_as_of": "2026-10-23"},
    {"cloud": "runpod", "instance_type": "H100-SXM-80G-8x",
     "accelerator": "H100", "gpus": 8, "gpu_memory_gb": 80,
     "hourly_cost": 21.52, "region": "community",
     "price_as_of": "2026-10-23"},

    # Lambda on-demand.
    {"cloud": "lambda", "instance_type": "gpu_1x_a100_40gb",
     "accelerator": "A100", "gpus": 1, "gpu_memory_gb": 40,
     "hourly_cost": 1.99, "region": "us-west-1",
     "price_as_of": "2026-10-23"},
    {"cloud": "lambda", "instance_type": "gpu_1x_h100_pcie",
     "accelerator": "H100", "gpus": 1, "gpu_memory_gb": 80,
     "hourly_cost": 3.29, "region": "us-west-1",
     "price_as_of": "2026-10-23"},
    {"cloud": "lambda", "instance_type": "gpu_8x_a100_80gb_sxm",
     "accelerator": "A100", "gpus": 8, "gpu_memory_gb": 80,
     "hourly_cost": 22.32, "region": "us-west-1",
     "price_as_of": "2026-10-23"},
    {"cloud": "lambda", "instance_type": "gpu_8x_h100_sxm",
     "accelerator": "H100", "gpus": 8, "gpu_memory_gb": 80,
     "hourly_cost": 31.92, "region": "us-west-1",
     "price_as_of": "2026-10-23"},

    # ---------------- Kubernetes (your own capacity) ----------------
    {"cloud": "k8s", "instance_type": "local-cluster", "accelerator": "A100",
     "gpus": 1, "gpu_memory_gb": 80, "hourly_cost": 0.00,
     "region": "default", "price_as_of": "2026-10-23"},

    # ---------------- CPU-only (accelerator=None requests) ----------------
    {"cloud": "aws", "instance_type": "m6i.large", "accelerator": None,
     "gpus": 0, "gpu_memory_gb": 0, "hourly_cost": 0.096,
     "region": "us-east-1", "price_as_of": "2024-01-01"},
    {"cloud": "gcp", "instance_type": "n2-standard-2", "accelerator": None,
     "gpus": 0, "gpu_memory_gb": 0, "hourly_cost": 0.097,
     "region": "us-central1", "price_as_of": "2024-01-01"},
    {"cloud": "k8s", "instance_type": "local-cluster-cpu",
     "accelerator": None, "gpus": 0, "gpu_memory_gb": 0,
     "hourly_cost": 0.00, "region": "default",
     "price_as_of": "2026-10-23"},
]

# Set LIGHTOS_CATALOG=/path/to/snapshot.json to use a scraped catalog instead.
CATALOG_PATH_ENV = "LIGHTOS_CATALOG"

_REQUIRED_FIELDS = ("cloud", "instance_type", "accelerator", "gpus",
                    "hourly_cost", "region")

# Optional fields and their defaults, so a hand-written or older snapshot
# still loads instead of erroring on a field added later.
_OPTIONAL_DEFAULTS = {"gpu_memory_gb": 0, "price_as_of": None}


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
            for field_name, default in _OPTIONAL_DEFAULTS.items():
                entry.setdefault(field_name, default)

    # ------------------------------------------------------------------
    # Price freshness
    # ------------------------------------------------------------------
    def stale_entries(self, max_age_days: int = 90,
                      today: Optional[date] = None) -> List[Dict[str, Any]]:
        """Entries whose price is older than max_age_days, or undated.

        Stale prices are the quiet failure mode of a cost optimizer: it keeps
        answering confidently while ranking the clouds wrongly.
        """
        today = today or date.today()
        stale = []
        for entry in self.db:
            as_of = entry.get("price_as_of")
            if not as_of:
                stale.append(entry)
                continue
            try:
                priced = date.fromisoformat(as_of)
            except (TypeError, ValueError):
                stale.append(entry)
                continue
            if (today - priced).days > max_age_days:
                stale.append(entry)
        return stale

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
                     accelerator: Optional[str] = None,
                     gpu_count: int = 1,
                     clouds: Optional[Sequence[str]] = None,
                     region: Optional[str] = None,
                     min_vram_gb: Optional[int] = None
                     ) -> List[Dict[str, Any]]:
        """Return every offer that satisfies the request, cheapest first.

        `accelerator` matches case-insensitively and `gpu_count` is a minimum.
        `min_vram_gb` constrains aggregate GPU memory (gpus * gpu_memory_gb),
        which is what decides whether a model fits.

        accelerator=None with no min_vram_gb means CPU-only; accelerator=None
        *with* min_vram_gb means "any GPU with at least this much memory",
        which is how you shop for an LLM across part numbers.
        """
        allowed = {c.lower() for c in clouds} if clouds else None
        wanted = accelerator.lower() if accelerator else None
        any_gpu = accelerator is None and min_vram_gb is not None

        valid_options: List[Dict[str, Any]] = []
        for entry in self.db:
            entry_accel = entry["accelerator"]
            entry_accel = entry_accel.lower() if entry_accel else None
            if any_gpu:
                if entry_accel is None:
                    continue  # CPU-only cannot satisfy a VRAM floor
            elif entry_accel != wanted:
                continue
            if entry["gpus"] < gpu_count:
                continue
            if min_vram_gb is not None:
                total_vram = entry["gpus"] * entry.get("gpu_memory_gb", 0)
                if total_vram < min_vram_gb:
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
                            accelerator: Optional[str] = None,
                            gpu_count: int = 1,
                            clouds: Optional[Sequence[str]] = None,
                            region: Optional[str] = None,
                            min_vram_gb: Optional[int] = None
                            ) -> Dict[str, Any]:
        """Return the single lowest-cost offer for the request.

        Raises CatalogError (a ValueError) when nothing matches.
        """
        options = self.find_matches(accelerator, gpu_count, clouds, region,
                                    min_vram_gb)
        if not options:
            raise CatalogError(
                "No available infrastructure found matching "
                f"{describe_request(accelerator, gpu_count, min_vram_gb)}"
                f"{_describe_constraints(clouds, region)}.")
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


def total_vram_gb(offer: Dict[str, Any]) -> int:
    """Aggregate GPU memory an offer provides."""
    return offer["gpus"] * offer.get("gpu_memory_gb", 0)


def describe_request(accelerator: Optional[str], gpu_count: int = 1,
                     min_vram_gb: Optional[int] = None) -> str:
    """Human-readable form of a hardware request."""
    parts = []
    if accelerator:
        parts.append(f"{gpu_count}x {accelerator}")
    elif min_vram_gb is None:
        parts.append("CPU-only")
    else:
        parts.append("any GPU")
    if min_vram_gb is not None:
        parts.append(f">={min_vram_gb} GB VRAM")
    return " with ".join(parts)


def _describe_constraints(clouds: Optional[Sequence[str]],
                          region: Optional[str]) -> str:
    constraint = ""
    if clouds:
        constraint += f" on {'/'.join(clouds)}"
    if region:
        constraint += f" in {region}"
    return constraint


# Weights-only memory for a dense model at a given precision, plus the
# headroom KV cache and activations need in practice. A planning aid, not a
# guarantee: real usage depends on batch size, context length and the serving
# engine.
_BYTES_PER_PARAM = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0,
                    "int8": 1.0, "int4": 0.5}


def vram_for_model(billion_params: float, precision: str = "bf16",
                   overhead: float = 1.2) -> int:
    """Rough VRAM floor, in GB, to serve a dense model of this size.

    >>> vram_for_model(70)        # 70B at bf16
    168
    """
    try:
        per_param = _BYTES_PER_PARAM[precision.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown precision '{precision}'. "
            f"Known: {', '.join(sorted(_BYTES_PER_PARAM))}.") from None
    return int(billion_params * per_param * overhead + 0.999)
