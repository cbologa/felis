from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .common import WorkflowError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Portable, additive workflows for pinned FELIS")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Freeze scientific inputs and construct the calculation graph")
    for option in ["campaign", "forcefield", "protocol", "output"]:
        plan.add_argument("--" + option, type=Path, required=True)
    plan.add_argument("--repo", type=Path)
    for name in ["prepare", "submit", "resume"]:
        p = commands.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--site", type=Path, required=True)
        if name != "prepare":
            p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("status")
    p.add_argument("--run", type=Path, required=True)
    p = commands.add_parser("analyze")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--corrections", type=Path)
    p = commands.add_parser("doctor")
    p.add_argument("--site", type=Path, required=True)
    p = commands.add_parser("verify-upstream")
    p.add_argument("--repo", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            from .planning import plan
            result = plan(args.campaign, args.forcefield, args.protocol, args.output, args.repo)
        elif args.command == "verify-upstream":
            from .integrity import verify_upstream
            from .planning import default_repo
            result = verify_upstream(args.repo or default_repo())
        elif args.command == "analyze":
            from .analysis import analyze
            result = analyze(args.run, args.corrections)
        else:
            from . import orchestration
            if args.command == "prepare":
                result = orchestration.prepare(args.run, args.site)
            elif args.command in {"submit", "resume"}:
                result = orchestration.execute(args.run, args.site, args.command == "resume", args.dry_run)
            elif args.command == "status":
                result = orchestration.status(args.run)
            else:
                result = orchestration.doctor(args.site)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (WorkflowError, OSError, subprocess.CalledProcessError) as error:
        print(f"felis-workflow: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
