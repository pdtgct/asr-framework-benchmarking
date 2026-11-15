"""The ``asrbench`` command.

``run`` writes one receipt and prints the console projection. ``sweep``
runs one ``run`` per stream count and writes an NVIDIA-layout table; a
failed cell is a result, so the sweep continues and still exits 0.
``compare``
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
from asrbench.corpora import ami, librispeech
from asrbench.manifest import ManifestError, load_manifest, verify_manifest
from asrbench.nim import load_nim_metadata, nim_overrides
from asrbench.reducers import MeasurementError
from asrbench.reporters import (
    NotHomogeneousError,
    console_report,
    frontier_report,
    frontier_text,
    nvidia_streaming_row,
    nvidia_table_tsv,
    perf_table_tsv,
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
    _add_nim_metadata_argument(run_parser)
    add_override_arguments(run_parser)

    sweep_parser = sub.add_parser(
        "sweep",
        help="run one receipt per stream count in performance_streams",
    )
    sweep_parser.add_argument("--preset", type=Path)
    sweep_parser.add_argument("--from-receipt", type=Path, dest="from_receipt")
    sweep_parser.add_argument(
        "--streams",
        type=_stream_list,
        help="comma-separated stream counts; default "
        "workload.performance_streams",
    )
    sweep_parser.add_argument("--out-dir", type=Path, required=True)
    sweep_parser.add_argument(
        "--prefix",
        default="",
        help="receipt name prefix; default c<chunk_ms>",
    )
    _add_nim_metadata_argument(sweep_parser)
    add_override_arguments(sweep_parser)

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

    nvidia_parser = sub.add_parser(
        "nvidia-table",
        help="one NVIDIA ASR NIM performance-table row per receipt",
    )
    nvidia_parser.add_argument("receipts", nargs="+", type=Path)
    nvidia_parser.add_argument("--out", type=Path)

    report_parser = sub.add_parser("report", help="project one receipt")
    report_parser.add_argument("receipt", type=Path)
    report_parser.add_argument(
        "--format",
        required=True,
        choices=("console", "vllm", "tsv"),
    )
    report_parser.add_argument("--out", type=Path)

    _add_corpus_parsers(sub)

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (
        ConfigError,
        ClipError,
        NotHomogeneousError,
        CanonicalError,
        MeasurementError,
        ManifestError,
    ) as error:
        print(f"asrbench: {error}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "run":
        return _run(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "sweep":
        return _sweep(args)
    if args.command == "index":
        write_index(args.directory, args.out)
        return 0
    if args.command == "frontier":
        return _frontier(args)
    if args.command == "sustained":
        return _sustained(args)
    if args.command == "report":
        return _report(args)
    if args.command == "nvidia-table":
        return _nvidia_table(args)
    if args.command == "corpus":
        return _corpus(args)
    raise ConfigError(f"unknown command {args.command}")


def _run(args: argparse.Namespace) -> int:
    import os

    env = dict(os.environ)
    config = resolve(
        preset_path=args.preset,
        from_receipt=args.from_receipt,
        env=env,
        overrides=_overrides(args),
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
    if config.run.emit_tsv:
        destination.with_suffix(".tsv").write_text(
            perf_table_tsv(receipt), encoding="utf-8"
        )
    sys.stdout.write(console_report(receipt))
    return 0


def _add_nim_metadata_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--nim-metadata",
        dest="nim_metadata",
        help="NIM /v1/metadata: a saved JSON file or the NIM HTTP URL; "
        "fills engine name and version, nim_profile_id, and nim_model_url. "
        "Explicit flags win; metadata wins over the preset and ASRBENCH_* "
        "variables for those fields",
    )


def _overrides(args: argparse.Namespace) -> dict[tuple[str, ...], Any]:
    """Flags the user passed, over values read from ``--nim-metadata``."""
    found: dict[tuple[str, ...], Any] = {}
    if args.nim_metadata:
        found.update(nim_overrides(load_nim_metadata(args.nim_metadata)))
    found.update(overrides_from_args(args))
    return found


def _stream_list(text: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in text.split(",") if item.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--streams is a comma-separated list of integers"
        ) from None
    if not counts or any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError(
            "--streams needs at least one count, each >= 1"
        )
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError(
            "--streams repeats a count; each would overwrite the last receipt"
        )
    return counts


def _sweep(args: argparse.Namespace) -> int:
    """Run one cell per stream count, one receipt each.

    Each count is resolved exactly as ``asrbench run --lanes N`` would
    be, so ``iterations_per_stream`` and a single-entry population follow
    the lane count. Counts run in order, one after another. A failed
    cell is a result and does not stop the sweep.
    """
    import os

    overrides = _overrides(args)
    for path, flag in (
        (("workload", "lanes"), "--lanes"),
        (("run", "output_path"), "--out"),
    ):
        if path in overrides:
            raise ConfigError(
                f"sweep sets {flag} per stream count; use --streams and "
                "--out-dir"
            )
    env = dict(os.environ)
    first = resolve(
        preset_path=args.preset,
        from_receipt=args.from_receipt,
        env=env,
        overrides=overrides,
    )
    streams = args.streams or first.workload.performance_streams
    if not streams:
        raise ConfigError(
            "no stream counts: set workload.performance_streams in the "
            "preset or pass --streams"
        )
    # --streams is checked by argparse; this covers performance_streams.
    if len(set(streams)) != len(streams):
        raise ConfigError(
            "stream counts repeat; each would overwrite the last receipt"
        )
    default_prefix = (
        "offline"
        if first.workload.mode == "offline"
        else f"c{first.workload.chunk_ms}"
    )
    prefix = args.prefix or default_prefix
    # Resolve every count before the first run, so a config error surfaces
    # before a long sweep has used any server time.
    cells = []
    for count in streams:
        destination = args.out_dir / f"{prefix}-n{count}.json"
        cells.append(
            (
                count,
                destination,
                resolve(
                    preset_path=args.preset,
                    from_receipt=args.from_receipt,
                    env=env,
                    overrides={
                        **overrides,
                        ("workload", "lanes"): count,
                        ("run", "output_path"): str(destination),
                    },
                ),
            )
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for count, destination, config in cells:
        print(f"sweep streams={count} -> {destination}", file=sys.stderr)
        for item in config.reproduction_overrides:
            print(
                f"reproduction {item.path}: {item.receipt} -> {item.resolved}",
                file=sys.stderr,
            )
        receipt = asyncio.run(run_benchmark(config, env=env))
        _write_json(destination, receipt)
        if config.run.emit_tsv:
            destination.with_suffix(".tsv").write_text(
                perf_table_tsv(receipt), encoding="utf-8"
            )
        destination.with_suffix(".console.txt").write_text(
            console_report(receipt), encoding="utf-8"
        )
        written.append(destination)
    rows = [
        nvidia_streaming_row(load_json_object(path), path.name)
        for path in written
    ]
    table = nvidia_table_tsv(rows)
    (args.out_dir / f"{prefix}-nvidia-layout.tsv").write_text(
        table, encoding="utf-8"
    )
    sys.stdout.write(table)
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


def _nvidia_table(args: argparse.Namespace) -> int:
    rows = [
        nvidia_streaming_row(load_json_object(path), str(path))
        for path in args.receipts
    ]
    text = nvidia_table_tsv(rows)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
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


def _add_corpus_parsers(sub: Any) -> None:
    """``asrbench corpus ...``: build and check manifests from local files."""
    corpus = sub.add_parser(
        "corpus",
        help="build or verify a corpus manifest from locally staged files",
    )
    kinds = corpus.add_subparsers(dest="corpus_command", required=True)

    urls = kinds.add_parser(
        "ami-urls", help="print the AMI files to stage on a connected host"
    )
    urls.add_argument("--recipe", default="ami-ihm-excerpts-v1")
    urls.add_argument(
        "--split",
        choices=sorted(ami.SPLITS),
        help="every headset of a split instead of the recipe's channels",
    )

    survey = kinds.add_parser(
        "ami-survey", help="measure pauses on staged headsets"
    )
    survey.add_argument("channels", nargs="+", metavar="MEETING:CHANNEL")
    survey.add_argument("--audio-root", type=Path, required=True)
    survey.add_argument("--min-silence-ms", type=int, default=800)
    survey.add_argument("--propose", type=int, default=3)
    survey.add_argument(
        "--windows", type=float, nargs="+", default=[20.16, 60.0]
    )

    excerpts = kinds.add_parser(
        "ami-excerpts", help="cut recipe windows into a manifest"
    )
    excerpts.add_argument("--recipe", default="ami-ihm-excerpts-v1")
    _ami_inputs(excerpts)
    excerpts.add_argument("--id")
    excerpts.add_argument("--min-silence-ms", type=int, default=800)

    segments = kinds.add_parser(
        "ami-segments", help="one clip per annotated headset segment"
    )
    _ami_inputs(segments)
    group = segments.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", choices=sorted(ami.SPLITS))
    group.add_argument("--meetings", nargs="+", metavar="MEETING[:N]")
    segments.add_argument("--id")
    segments.add_argument("--min-duration-s", type=float, default=0.0)
    segments.add_argument("--max-clips", type=int)

    libri = kinds.add_parser(
        "librispeech", help="one clip per LibriSpeech utterance"
    )
    libri.add_argument("--root", type=Path, required=True)
    libri.add_argument("--split", choices=librispeech.SPLITS, required=True)
    libri.add_argument("--out-dir", type=Path, required=True)
    libri.add_argument("--id")
    libri.add_argument("--limit", type=int)
    libri.add_argument("--speakers", nargs="+")

    verify = kinds.add_parser(
        "verify", help="re-hash every clip of a manifest against its pins"
    )
    verify.add_argument("manifest", type=Path)


def _ami_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="ami_public_manual_1.6.2.zip or its extracted directory",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--language", default="en-US")


def _corpus(args: argparse.Namespace) -> int:
    command = args.corpus_command
    if command == "ami-urls":
        if args.split:
            print(ami.ANNOTATIONS_URL)
            print(
                "# Headset-0..3 of each meeting; corpusResources/meetings.xml "
                "in the annotations lists any further headset"
            )
            for meeting in ami.SPLITS[args.split]:
                for channel in range(4):
                    print(ami.channel_url(meeting, channel))
            return 0
        recipe, _ = ami.load_recipe(args.recipe)
        for url in ami.mirror_urls(ami.recipe_channels(recipe)):
            print(url)
        return 0
    if command == "ami-survey":
        pairs = []
        for spec in args.channels:
            meeting, _, channel = spec.partition(":")
            pairs.append((meeting, int(channel or 0)))
        for row in ami.survey(
            channels=pairs,
            audio_root=args.audio_root,
            windows=args.windows,
            min_silence_ms=args.min_silence_ms,
            propose=args.propose,
        ):
            print(
                f"{row.meeting}:{row.channel}  {row.duration_s / 60:.1f} min  "
                f"pauses>={args.min_silence_ms}ms: {row.qualifying_pauses}  "
                f"longest={row.longest_gap_ms}ms"
            )
            for window, offsets in row.proposals.items():
                shown = ", ".join(f"{item:.3f}s" for item in offsets)
                print(f"  {window:>7.2f}s windows: {shown or 'none found'}")
        return 0
    if command == "ami-excerpts":
        result = ami.build_excerpts(
            recipe_spec=args.recipe,
            audio_root=args.audio_root,
            annotations=args.annotations,
            out_dir=args.out_dir,
            manifest_id=args.id,
            language=args.language,
            min_silence_ms=args.min_silence_ms,
        )
    elif command == "ami-segments":
        meetings = list(ami.SPLITS[args.split]) if args.split else args.meetings
        result = ami.build_segments(
            meetings=meetings,
            audio_root=args.audio_root,
            annotations=args.annotations,
            out_dir=args.out_dir,
            manifest_id=args.id or f"ami-ihm-{args.split or 'custom'}-segments",
            language=args.language,
            min_duration_s=args.min_duration_s,
            max_clips=args.max_clips,
        )
    elif command == "librispeech":
        result = librispeech.build(
            root=args.root,
            split=args.split,
            out_dir=args.out_dir,
            manifest_id=args.id,
            limit=args.limit,
            speakers=args.speakers,
        )
    elif command == "verify":
        loaded = load_manifest(args.manifest)
        problems = verify_manifest(loaded)
        for line in problems:
            print(f"FAIL {line}", file=sys.stderr)
        print(
            f"{loaded.id} {loaded.digest} {len(loaded.clips)} clips "
            f"{loaded.total_duration_s:.1f}s: "
            + ("ok" if not problems else f"{len(problems)} failed")
        )
        return 1 if problems else 0
    else:
        raise ConfigError(f"unknown corpus command {command}")
    for line in result.warnings:
        print(f"warning: {line}", file=sys.stderr)
    manifest = result.manifest
    print(
        f"{manifest.path}\n{manifest.id} {manifest.digest} "
        f"{len(manifest.clips)} clips {manifest.total_duration_s:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
