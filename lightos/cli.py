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
from .catalog import CatalogError, CostCatalog, estimate_cost
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

    validate = sub.add_parser("validate",
                              help="Parse a blueprint and report what it asks "
                                   "for, without provisioning.")
    validate.add_argument("blueprint", help="Path to the task YAML.")

    serve = sub.add_parser("serve",
                           help="Run the HTTP API (what agentic.lightos.sh "
                                "serves).")
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


def _cmd_catalog(args: argparse.Namespace) -> int:
    catalog = CostCatalog(catalog_path=args.catalog)
    options = catalog.find_matches(args.accelerator, args.gpus,
                                   clouds=args.clouds, region=args.region)
    if not options:
        target = (f"{args.gpus}x {args.accelerator}" if args.accelerator
                  else "CPU-only")
        print(f"No capacity matches {target}.")
        print(f"Known accelerators: {', '.join(catalog.accelerators())}")
        return 1

    header = f"{'CLOUD':<6} {'INSTANCE_TYPE':<18} {'ACCEL':<7} {'GPUS':>4} " \
             f"{'REGION':<12} {'$/HR':>9} {'$/' + str(args.hours) + 'HR':>11}"
    print(header)
    print("-" * len(header))
    for option in options:
        print(f"{option['cloud']:<6} {option['instance_type']:<18} "
              f"{str(option['accelerator'] or '-'):<7} {option['gpus']:>4} "
              f"{option['region']:<12} {option['hourly_cost']:>9.4f} "
              f"{estimate_cost(option, args.hours):>11.4f}")
    cheapest = options[0]
    print(f"\nCheapest: {cheapest['cloud'].upper()} "
          f"{cheapest['instance_type']} @ ${cheapest['hourly_cost']}/hr")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    task = Task.from_yaml(args.blueprint)
    catalog = CostCatalog(catalog_path=args.catalog)
    print(f"Task       : {task.name}")
    print(f"Hardware   : {task.gpu_count}x {task.accelerator}"
          if task.accelerator else "Hardware   : CPU-only")
    print(f"Workdir    : {task.workdir}")
    print(f"Remote dir : {task.remote_workdir}")
    print(f"Autostop   : {task.autostop}")
    print(f"Env vars   : {', '.join(task.envs) or '(none)'}")
    offer = catalog.find_cheapest_match(task.accelerator, task.gpu_count,
                                        task.clouds, task.region)
    print(f"Would use  : {offer['cloud'].upper()} {offer['instance_type']} "
          f"@ {offer['region']} (${offer['hourly_cost']}/hr)")
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
