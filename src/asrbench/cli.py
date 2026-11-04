"""The ``asrbench`` command.

``run`` writes one receipt and prints the console projection. ``compare``
refuses when the comparison key does not match unless ``--force`` and
``--reason`` are both present. The reason is copied into the output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from asrbench.canonical import CanonicalError
from asrbench.clips import ClipError
from asrbench.compare import compare_receipts
from asrbench.config import (
    ConfigError,
    add_override_arguments,
    load_json_object,
    overrides_from_args,
    resolve,
)
from asrbench.reducers import MeasurementError
from asrbench.reporters import (
    NotHomogeneousError,
    console_report,
    frontier_report,
    frontier_text,
    sustained_streams_report,
    vendor_tsv,
    vllm_report,
)
from asrbench.runner import run_benchmark
from asrbench.tracking import write_index


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand.

    Args:
        argv: Arguments, excluding the program name. ``None`` reads
            ``sys.argv``.

    Returns:
        A process exit code. Comparison refusal is 2. A failed cell is
        still 0: the failure is a result.
    """
    parser = argparse.ArgumentParser(prog="asrbench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run a preset or a receipt")
    run_parser.add_argument("--preset", type=Path)
    run_parser.add_argument("--from-receipt", type=Path, dest="from_receipt")
    add_override_arguments(run_parser)

    compare_parser = sub.add_parser("compare", help="compare two receipts")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--force", action="store_true")
    compare_parser.add_argument("--reason")

    index_parser = sub.add_parser("index", help="rebuild a receipt index")
    index_parser.add_argument("directory", type=Path)
    index_parser.add_argument("--out", type=Path, required=True)

    frontier_parser = sub.add_parser(
        "frontier", help="latency/accuracy frontier"
    )
    frontier_parser.add_argument("receipts", nargs="+", type=Path)
    frontier_parser.add_argument("--out", type=Path)

    sustained_parser = sub.add_parser(
        "sustained", help="sustained streams across a concurrency sweep"
    )
    sustained_parser.add_argument("receipts", nargs="+", type=Path)

    report_parser = sub.add_parser("report", help="project one receipt")
    report_parser.add_argument("receipt", type=Path)
    report_parser.add_argument(
        "--format",
        required=True,
        choices=("console", "vllm", "tsv"),
    )
    report_parser.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (
        ConfigError,
        ClipError,
        NotHomogeneousError,
        CanonicalError,
        MeasurementError,
    ) as error:
        print(f"asrbench: {error}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "run":
        return _run(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "index":
        write_index(args.directory, args.out)
        return 0
    if args.command == "frontier":
        return _frontier(args)
    if args.command == "sustained":
        return _sustained(args)
    if args.command == "report":
        return _report(args)
    raise ConfigError(f"unknown command {args.command}")


def _run(args: argparse.Namespace) -> int:
    import os

    env = dict(os.environ)
    config = resolve(
        preset_path=args.preset,
        from_receipt=args.from_receipt,
        env=env,
        overrides=overrides_from_args(args),
    )
    if config.run.output_path is None:
        raise ConfigError("--out is required")
    for item in config.reproduction_overrides:
        print(
            f"reproduction {item.path}: {item.receipt} -> {item.resolved}",
            file=sys.stderr,
        )
    receipt = asyncio.run(run_benchmark(config, env=env))
    destination = Path(config.run.output_path)
    _write_json(destination, receipt)
    sys.stdout.write(console_report(receipt))
    return 0


def _compare(args: argparse.Namespace) -> int:
    result = compare_receipts(
        load_json_object(args.left),
        load_json_object(args.right),
        force=args.force,
        reason=args.reason,
    )
    json.dump(result.to_json(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    if result.status == "refused":
        return 2
    return 0


def _frontier(args: argparse.Namespace) -> int:
    receipts = [load_json_object(path) for path in args.receipts]
    sources = [path.name for path in args.receipts]
    report = frontier_report(receipts, sources)
    if args.out is not None:
        _write_json(args.out, report)
    sys.stdout.write(frontier_text(report))
    return 0


def _sustained(args: argparse.Namespace) -> int:
    receipts = [load_json_object(path) for path in args.receipts]
    sources = [path.name for path in args.receipts]
    report = sustained_streams_report(receipts, sources)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _report(args: argparse.Namespace) -> int:
    receipt = load_json_object(args.receipt)
    if args.format == "console":
        text = console_report(receipt)
    elif args.format == "vllm":
        text = json.dumps(vllm_report(receipt), indent=2) + "\n"
    else:
        text = vendor_tsv(receipt)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
