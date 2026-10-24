"""Command-line interface.

    lightos launch task_config.yaml [--dry-run] [--cloud aws]
    lightos catalog [--accelerator A100] [--gpus 8]
    lightos validate task_config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from . import __version__
from .catalog import (CatalogError, CostCatalog, describe_request,
                      estimate_cost, total_vram_gb, vram_for_model)
from .execution import (DEFAULT_SSH_KEY, SSH_KEY_ENV,
                        describe_available_keys)
from .orchestrator import Orchestrator
from .task import Task, TaskError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lightos",
        description="Cost-optimized multi-cloud job orchestrator.")
    parser.add_argument("--version", action="version",
                        version=f"lightos {__version__}")
    parser.add_argument("--catalog", metavar="PATH",
                        help="JSON pricing snapshot to use instead of the "
                             "bundled one.")
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch",
                            help="Run a task blueprint end to end.")
    launch.add_argument("blueprint", help="Path to the task YAML.")
    launch.add_argument("--dry-run", action="store_true",
                        help="Resolve the cheapest cluster and stop; "
                             "provisions nothing.")
    launch.add_argument("--cloud",
                        help="Restrict to one cloud, overriding the YAML.")
    launch.add_argument("--region",
                        help="Restrict to one region, overriding the YAML.")
    launch.add_argument("--no-autostop", action="store_true",
                        help="Leave the cluster running after the job exits.")

    catalog = sub.add_parser("catalog",
                             help="Show matching capacity and prices.")
    catalog.add_argument("-a", "--accelerator",
                         help="Accelerator to match, e.g. A100. Omit for "
                              "CPU-only capacity.")
    catalog.add_argument("-n", "--gpus", type=int, default=1,
                         help="Minimum accelerator count (default 1).")
    catalog.add_argument("--cloud", action="append", dest="clouds",
                         help="Limit to a cloud; repeatable.")
    catalog.add_argument("--region", help="Limit to one region.")
    catalog.add_argument("--hours", type=float, default=1.0,
                         help="Project the cost over this many hours.")
    catalog.add_argument("--min-vram", type=int, dest="min_vram",
                         help="Minimum aggregate GPU memory in GB. Omit the "
                              "accelerator to shop across part numbers.")
    catalog.add_argument("--model-size", type=float, dest="model_size",
                         help="Dense model size in billions of parameters; "
                              "sets --min-vram from it.")
    catalog.add_argument("--precision", default="bf16",
                         help="Precision for --model-size (bf16, fp16, fp8, "
                              "int8, int4, fp32). Default bf16.")

    validate = sub.add_parser("validate",
                              help="Parse a blueprint and report what it asks "
                                   "for, without provisioning.")
    validate.add_argument("blueprint", help="Path to the task YAML.")

    serve = sub.add_parser("serve",
                           help="Run the HTTP API that a frontend calls.")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int,
                       default=int(os.environ.get("PORT", "8080")))

    return parser


def _load_task(args: argparse.Namespace) -> Task:
    task = Task.from_yaml(args.blueprint)
    if getattr(args, "cloud", None):
        task.cloud = args.cloud.lower()
    if getattr(args, "region", None):
        task.region = args.region
    if getattr(args, "no_autostop", False):
        task.autostop = False
    return task


def _cmd_launch(args: argparse.Namespace) -> int:
    task = _load_task(args)
    orchestrator = Orchestrator(catalog=CostCatalog(catalog_path=args.catalog))
    result = orchestrator.launch(task, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    return 0 if result.succeeded else (result.exit_code or 1)


def _resolve_min_vram(args: argparse.Namespace) -> Optional[int]:
    """Turn --model-size into a VRAM floor, keeping whichever is larger."""
    if getattr(args, "model_size", None):
        need = vram_for_model(args.model_size, args.precision)
        print(f"{args.model_size}B at {args.precision} needs about {need} GB "
              f"of VRAM (weights plus headroom).")
        if getattr(args, "min_vram", None):
            return max(need, args.min_vram)
        return need
    return getattr(args, "min_vram", None)


def _warn_if_stale(catalog: CostCatalog) -> None:
    stale = catalog.stale_entries()
    if stale:
        print(f"warning: {len(stale)} of {len(catalog)} prices are over 90 "
              f"days old or undated; refresh before trusting a ranking.",
              file=sys.stderr)


def _cmd_catalog(args: argparse.Namespace) -> int:
    catalog = CostCatalog(catalog_path=args.catalog)
    min_vram = _resolve_min_vram(args)
    options = catalog.find_matches(args.accelerator, args.gpus,
                                   clouds=args.clouds, region=args.region,
                                   min_vram_gb=min_vram)
    if not options:
        target = describe_request(args.accelerator, args.gpus, min_vram)
        print(f"No capacity matches {target}.")
        print(f"Known accelerators: {', '.join(catalog.accelerators())}")
        print(f"Known clouds: {', '.join(catalog.clouds())}")
        return 1

    header = (f"{'CLOUD':<8} {'INSTANCE_TYPE':<22} {'ACCEL':<6} {'GPUS':>4} "
              f"{'VRAM':>6} {'REGION':<12} {'$/HR':>9} "
              f"{'$/' + str(args.hours) + 'HR':>11}")
    print(header)
    print("-" * len(header))
    for option in options:
        vram = total_vram_gb(option)
        print(f"{option['cloud']:<8} {option['instance_type']:<22} "
              f"{str(option['accelerator'] or '-'):<6} {option['gpus']:>4} "
              f"{(str(vram) + 'G') if vram else '-':>6} "
              f"{option['region']:<12} {option['hourly_cost']:>9.4f} "
              f"{estimate_cost(option, args.hours):>11.4f}")

    cheapest, dearest = options[0], options[-1]
    print(f"\nCheapest: {cheapest['cloud'].upper()} "
          f"{cheapest['instance_type']} @ ${cheapest['hourly_cost']}/hr")
    # The spread is the point: same silicon, very different prices.
    if cheapest["hourly_cost"] > 0 and dearest is not cheapest:
        factor = dearest["hourly_cost"] / cheapest["hourly_cost"]
        print(f"Dearest match is {factor:.1f}x that: "
              f"{dearest['cloud'].upper()} {dearest['instance_type']} "
              f"@ ${dearest['hourly_cost']}/hr")
    _warn_if_stale(catalog)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    task = Task.from_yaml(args.blueprint)
    catalog = CostCatalog(catalog_path=args.catalog)
    print(f"Task       : {task.name}")
    print("Hardware   : "
          + describe_request(task.accelerator, task.gpu_count,
                             task.min_vram_gb))
    print(f"Workdir    : {task.workdir}")
    print(f"Remote dir : {task.remote_workdir}")
    print(f"Autostop   : {task.autostop}")
    print(f"Env vars   : {', '.join(task.envs) or '(none)'}")

    # Check the key now: otherwise the first sign of trouble is a failed
    # connect, after a node has been provisioned and is already billing.
    key_path = os.path.expanduser(
        task.ssh_key_path or os.environ.get(SSH_KEY_ENV) or DEFAULT_SSH_KEY)
    key_missing = not os.path.exists(key_path)
    if key_missing:
        print(f"SSH key    : MISSING at {key_path}")
        print(f"             {describe_available_keys()}")
        print(f"             Set {SSH_KEY_ENV} or add 'ssh_key_path:' to the "
              f"blueprint.")
    else:
        print(f"SSH key    : {key_path}")
    offer = catalog.find_cheapest_match(task.accelerator, task.gpu_count,
                                        task.clouds, task.region,
                                        task.min_vram_gb)
    print(f"Would use  : {offer['cloud'].upper()} {offer['instance_type']} "
          f"@ {offer['region']} (${offer['hourly_cost']}/hr)")

    if key_missing:
        # The blueprint parses, but launching it would provision a node and
        # then fail to log in -- billing you for nothing. Report that as a
        # failure so it is caught before any spend.
        print("\nBlueprint parses, but the launch would fail: no usable SSH "
              "key.")
        return 1
    print("\nBlueprint is valid.")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(host=args.host, port=args.port)
    return 0


_COMMANDS = {
    "launch": _cmd_launch,
    "catalog": _cmd_catalog,
    "validate": _cmd_validate,
    "serve": _cmd_serve,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (TaskError, CatalogError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
