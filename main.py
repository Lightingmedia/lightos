#!/usr/bin/env python3
"""LightOS entry point.

    python main.py                       # launch ./task_config.yaml
    python main.py other_task.yaml       # launch a specific blueprint
    python main.py --dry-run             # resolve the cheapest cluster only
    python main.py catalog -a A100 -n 8  # any lightos CLI subcommand

Bare invocation defaults to `launch task_config.yaml` so the common case is
one command; anything else is handed straight to `lightos.cli`.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

from lightos.cli import main as cli_main

DEFAULT_BLUEPRINT = "task_config.yaml"
_SUBCOMMANDS = ("launch", "catalog", "validate", "serve")


def _normalize(argv: Sequence[str]) -> List[str]:
    """Let `python main.py [blueprint] [flags]` mean `launch`."""
    args = list(argv)
    if args and (args[0] in _SUBCOMMANDS or args[0] in ("--version", "-h",
                                                        "--help")):
        return args

    blueprint: Optional[str] = None
    rest: List[str] = []
    for arg in args:
        if blueprint is None and not arg.startswith("-"):
            blueprint = arg
        else:
            rest.append(arg)

    if blueprint is None:
        blueprint = os.environ.get("LIGHTOS_TASK", DEFAULT_BLUEPRINT)
    return ["launch", blueprint, *rest]


if __name__ == "__main__":
    sys.exit(cli_main(_normalize(sys.argv[1:])))
