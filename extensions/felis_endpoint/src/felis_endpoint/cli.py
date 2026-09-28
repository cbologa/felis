"""Separate CLI: no changes to the ABFE command or stage graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from felis_workflows.common import WorkflowError, read
from felis_workflows.config import site_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Derived FELIS physical endpoint and qualification workflows")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan", help="Freeze explicit endpoint starting structures")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--site", type=Path, required=True)
    for name in ("submit", "resume"):
        p = commands.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--site", type=Path, required=True)
        p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("status")
    p.add_argument("--run", type=Path, required=True)
    p = commands.add_parser("run-unit", help=argparse.SUPPRESS)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--site", type=Path, required=True)
    p.add_argument("--task-id", required=True)
    p = commands.add_parser("runtime-probe", help=argparse.SUPPRESS)
    p.add_argument("--site", type=Path, required=True)
    p = commands.add_parser("qualify", help="Report ABFE artifact coverage and native convergence evidence")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--site", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("analyze-networks")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--site", type=Path, required=True)
    p = commands.add_parser("doctor")
    p.add_argument("--site", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "runtime-probe":
            from .identity import identity
            print("FELIS_ENDPOINT_IDENTITY_JSON=" + json.dumps(identity(site_config(args.site)), sort_keys=True))
            return 0
        elif args.command == "doctor":
            from .identity import identity, runner_identity
            site = site_config(args.site)
            result = {"cli": identity(site), "simulation_runner": runner_identity(args.site, site)}
        elif args.command == "plan":
            from .plan import plan
            result = plan(args.config, args.output, args.site)
        elif args.command in ("submit", "resume"):
            from .submission import submit
            result = submit(args.run, args.site, resume=args.command == "resume", dry_run=args.dry_run)
        elif args.command == "status":
            from .execution import status
            from .identity import activate
            activate(read(args.run / "site.json"))
            result = status(args.run)
        elif args.command == "run-unit":
            from .execution import run_task
            result = run_task(args.run, args.task_id, read(args.site), site_path=args.site)
        elif args.command == "qualify":
            from .qualification import qualify
            result = qualify(args.run, site_config(args.site), args.output)
        else:
            from .networks import analyze
            from .identity import identity
            result = analyze(args.config, args.output, runtime=identity(site_config(args.site)))
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (WorkflowError, OSError, subprocess.CalledProcessError, StopIteration) as error:
        print(f"felis-endpoint: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
