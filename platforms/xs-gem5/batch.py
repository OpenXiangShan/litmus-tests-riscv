#!/usr/bin/env python3
"""Sweep this repository's non-mixed-size RISC-V tests on XS-GEM5.

Uses msort7 and mselect7 for deduplication and thread selection. Tests with one
or two threads use the two-hart runtime; larger tests are reported as unsupported.
Per-test records and append-only progress survive interruption. Reusing an
output directory resumes completed test records.
A nonzero exit also reports coverage gaps (unsupported instructions or models),
so inspect the per-test status rather than treating every exclusion as a CPU bug.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import check


def execute(command, log, timeout=120, cwd=None):
    result = subprocess.run(
        list(map(str, command)),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {log}")
    return result.stdout


def classify_simulation(result, directory):
    if result["status"] == "pass":
        return "pass"
    if result["status"] == "forbidden":
        return "forbidden"
    text = (directory / result["name"] / "sim.log").read_text(errors="replace")
    if "panic:" in text or "Assertion " in text:
        return "simulator-panic"
    if "max instruction count" in text:
        return "instruction-limit"
    if "because simulate() limit reached" in text:
        return "tick-limit"
    if result["returncode"] is None:
        return "timeout"
    return "incomplete"


def one_test(args, number, source):
    # Each source has its own namespace, including identically named tests.
    directory = args.outdir / "cases" / f"{number:05d}-{source.stem}"
    record = directory / "result.json"
    if record.exists():
        return json.loads(record.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "source": str(source.relative_to(check.SUITE)),
        "source_sha256": check.sha256(source),
        "index": number,
    }
    options = SimpleNamespace(**vars(args))
    options.outdir = directory
    phase = "prepare"
    try:
        # prepare is deterministic; discard an interrupted preparation only.
        images = directory / "images"
        if images.exists():
            shutil.rmtree(images)
        image = check.prepare(options, source, False)
        phase = "simulation"
        results = []
        for topology in args.topologies:
            for swap in (0, 1):
                name = f"fixed-{topology}-swap{swap}-{source.stem}"
                saved = directory / name / "result.json"
                if saved.exists():
                    result = json.loads(saved.read_text())
                else:
                    run_dir = directory / name
                    if run_dir.exists():
                        shutil.rmtree(run_dir)
                    result = check.simulate(
                        options, image, topology, swap, False
                    )
                result["classification"] = classify_simulation(
                    result, directory
                )
                results.append(result)
        classes = [r["classification"] for r in results]
        status = "pass" if set(classes) == {"pass"} else "failed"
        identity.update(status=status, runs=results)
    except ValueError as error:
        identity.update(
            status="unsupported-adapter"
            if phase == "prepare"
            else "checker-error",
            phase=phase,
            reason=str(error),
        )
    except subprocess.TimeoutExpired as error:
        identity.update(
            status="model-or-generator-timeout"
            if phase == "prepare"
            else "tool-timeout",
            phase=phase,
            reason=str(error),
        )
    except RuntimeError as error:
        reason = str(error)
        status = (
            "model-or-generator-error" if phase == "prepare" else "build-error"
        )
        if phase == "simulation" and (
            "unrecognized opcode" in reason or "unrecognized CSR" in reason
        ):
            status = "unsupported-instruction"
        identity.update(status=status, phase=phase, reason=reason)
    except Exception as error:
        identity.update(
            status="driver-error",
            phase=phase,
            reason=f"{type(error).__name__}: {error}",
        )
    check.write_json(record, identity)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gem5-root", type=Path, required=True)
    parser.add_argument("--gem5", required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--tool-prefix", default="riscv64-linux-gnu-")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--limit", type=int, help="pilot only; omit for the full list"
    )
    parser.add_argument("--topology", action="append", choices=("smt", "spmt"))
    args = parser.parse_args()
    if min(args.jobs, args.trials, args.timeout) < 1:
        parser.error("trials, jobs and timeout must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    args.gem5_root = args.gem5_root.resolve()
    if not (args.gem5_root / "configs/example/smt_idealkmhv3.py").is_file():
        parser.error(
            "gem5-root does not contain configs/example/smt_idealkmhv3.py"
        )
    args.tools = args.tools.resolve()
    args.gem5 = check.executable(args.gem5)
    args.herd7 = check.executable(str(args.tools / "herd7"))
    args.litmus7 = check.executable(str(args.tools / "litmus7"))
    args.cc = check.executable(args.tool_prefix + "gcc")
    args.msort7 = check.executable(str(args.tools / "msort7"))
    args.mselect7 = check.executable(str(args.tools / "mselect7"))
    tools = check.tool_manifest(args, (args.msort7, args.mselect7))
    args.baseline = None
    args.topologies = args.topology or ["smt", "spmt"]
    if len(set(args.topologies)) != len(args.topologies):
        parser.error("duplicate topology")
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.gem5_head = check.invoke(
        ["git", "rev-parse", "HEAD"], cwd=args.gem5_root
    )[0].strip()
    args.suite_head = check.invoke(
        ["git", "rev-parse", "HEAD"], cwd=check.SUITE
    )[0].strip()
    args.binary_hashes = {str(args.gem5): check.sha256(args.gem5)}
    raw_sources = sorted(
        (check.SUITE / "tests/non-mixed-size").rglob("*.litmus")
    )
    raw_list = args.outdir / "@raw-sources"
    raw_list.write_text("\n".join(map(str, raw_sources)) + "\n")
    paths = args.outdir / "@nonmixed"
    sorted_list = execute(
        [args.msort7, "-keepbad", "false", raw_list],
        args.outdir / "msort.log",
        cwd=check.SUITE,
    )
    sources = [
        (check.SUITE / line).resolve()
        for line in sorted_list.splitlines()
        if line and not line.startswith("#")
    ]
    paths.write_text("\n".join(map(str, sources)) + "\n")
    selected_text = execute(
        [
            args.mselect7,
            "-arch",
            "RISCV",
            "-threads",
            "[1..2]",
            paths,
        ],
        args.outdir / "mselect.log",
        cwd=check.SUITE,
    )
    selected = [
        Path(line)
        for line in selected_text.splitlines()
        if line and not line.startswith("#")
    ]
    chosen = set(selected)
    skipped = [
        {
            "source": str(p.relative_to(check.SUITE)),
            "reason": "requires-more-harts-or-outside-thread-range",
        }
        for p in sources
        if p not in chosen
    ]
    diagnostics = (args.outdir / "msort.log").read_text()
    bad_sources = set(re.findall(r'Warning: File "([^"]+)"', diagnostics))
    canonical = set(sources)
    skipped.extend(
        {
            "source": str(p.relative_to(check.SUITE)),
            "reason": "model-parser-error"
            if str(p) in bad_sources
            else "duplicate-test",
        }
        for p in raw_sources
        if p not in canonical
    )
    skipped.extend(
        {
            "source": str(p.relative_to(check.SUITE)),
            "reason": "mixed-size-model-adapter-not-supported",
        }
        for p in sorted((check.SUITE / "tests/mixed-size").rglob("*.litmus"))
    )
    check.write_json(args.outdir / "skipped.json", skipped)
    if args.limit:
        selected = selected[: args.limit]
    identity = {
        "gem5": str(args.gem5),
        "binary_hashes": args.binary_hashes,
        "gem5_head": args.gem5_head,
        "suite_head": args.suite_head,
        **tools,
        "gem5_root": str(args.gem5_root),
        "timeout": args.timeout,
        "sources": {
            str(p.relative_to(check.SUITE)): check.sha256(p) for p in selected
        },
        "adapter_sources": {
            p.name: check.sha256(p)
            for p in check.HERE.iterdir()
            if p.suffix in (".py", ".c", ".S", ".ld")
        },
        "trials": args.trials,
        "topologies": args.topologies,
        "nonmixed_file_count": len(raw_sources),
        "unique_valid_nonmixed_count": len(sources),
        "selected_count": len(selected),
        "skipped_counts": dict(Counter(item["reason"] for item in skipped)),
        "pilot_limit": args.limit,
    }
    manifest = args.outdir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("execution identity changed; use another outdir")
    check.write_json(manifest, identity)
    print(
        f"Selected {len(selected)} tests, skips: {identity['skipped_counts']}",
        flush=True,
    )
    results = []
    started = time.monotonic()
    with (args.outdir / "progress.jsonl").open("a", buffering=1) as progress:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(one_test, args, i, p)
                for i, p in enumerate(selected)
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress.write(json.dumps(result) + "\n")
                counts = dict(Counter(r["status"] for r in results))
                check.write_json(
                    args.outdir / "progress.json",
                    {
                        "done": len(results),
                        "total": len(selected),
                        "counts": counts,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                if len(results) % 25 == 0 or result["status"] != "pass":
                    print(
                        f"{len(results)}/{len(selected)} {result['status']} {result['source']} {counts}",
                        flush=True,
                    )
    check.write_json(
        args.outdir / "results.json", sorted(results, key=lambda r: r["index"])
    )
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(error, file=sys.stderr)
        sys.exit(2)
