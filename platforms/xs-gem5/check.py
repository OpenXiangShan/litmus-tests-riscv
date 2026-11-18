#!/usr/bin/env python3
r"""Run this repository's RVWMO tests on XS-GEM5 SMT and SpMT.

Requires herdtools7 7.58 (herd7 and litmus7) and riscv64-linux-gnu tools.
Example:
  python3 platforms/xs-gem5/check.py --gem5-root /path/to/GEM5 \
    --herd7 /path/to/herd7 --litmus7 /path/to/litmus7 \
    --outdir /tmp/riscv-litmus --baseline /path/to/pre-fix/gem5.opt

litmus7 parses the original files and emits the tested thread functions.
The adapter supports one or two harts with integer and pointer observations.
Startup, UART, and exit use XS-GEM5 conventions; RTL integration needs its own
platform hooks. Generated instruction bodies and the RVWMO oracle are reusable.
For filter tests, the oracle additionally observes filter-only result registers.
All observed tuples must belong to herd7's RVWMO state set. Unobserved allowed
states are coverage information, never a test failure.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SUITE = HERE.parents[1]
DEFAULT_TESTS = (
    "tests/non-mixed-size/CO/CoRR.litmus",
    "tests/non-mixed-size/BASIC_2_THREAD/MP+fence.rw.rws.litmus",
    "tests/non-mixed-size/HAND/LB+amoadd-data-amoadds.litmus",
)
CORR = SUITE / DEFAULT_TESTS[0]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invoke(command, **kwargs):
    result = subprocess.run(
        [str(item) for item in command],
        capture_output=True,
        text=True,
        timeout=120,
        **kwargs,
    )
    if result.returncode:
        raise RuntimeError(f"{command}:\n{result.stdout}\n{result.stderr}")
    return result.stdout, result.stderr


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parse_states(text):
    match = re.search(r"^States (\d+)\n", text, re.MULTILINE)
    if not match or int(match[1]) == 0:
        raise ValueError("herd7 did not enumerate a nonempty state set")
    lines = text[match.end() :].splitlines()[: int(match[1])]
    states, locations = [], None
    pair = r"(\d+:x\d+|\[\w+\])=(-?\d+|[A-Za-z_]\w*);"
    for line in lines:
        pairs = re.findall(pair, line)
        if re.sub(pair + r"\s*", "", line):
            raise ValueError(f"unsupported herd state: {line}")
        names = [key for key, _ in pairs]
        if locations is None:
            locations = names
        if names != locations or len(set(names)) != len(names):
            raise ValueError("inconsistent observation locations")
        values = []
        for _, value in pairs:
            if re.fullmatch(r"-?\d+", value):
                value = int(value)
                if value < -(1 << 63) or value >= 1 << 64:
                    raise ValueError(
                        "only observations up to 64 bits are supported"
                    )
                if value >= 1 << 63:
                    value -= 1 << 64
            values.append(value)
        states.append(tuple(values))
    if len(states) != int(match[1]) or len(set(states)) != len(states):
        raise ValueError("incomplete or duplicate herd state list")
    if not locations:
        # Instruction-only tests have one empty state. A constant auxiliary
        # observation lets the same runtime check completion without zero VLAs.
        return ["__completion"], [(0,)]
    return locations, states


def extract_threads(text, locations, states, delayed):
    functions = re.findall(
        r"__attribute__\(\(noinline\)\) static void code(\d+)"
        r"\(([^)]*)\) \{\n(.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if [pid for pid, _, _ in functions] not in (["0"], ["0", "1"]):
        raise ValueError("expected one or two litmus7 thread functions")
    log = re.search(r"typedef struct \{\n([^{}]*)\} log_t;", text)
    scalar = r"(?:int|unsigned int|int32_t|uint32_t|int64_t|uint64_t)(?:\*)?"
    typed_fields = (
        re.findall(rf"^  ({scalar}) (\w+);$", log[1], re.MULTILINE)
        if log
        else []
    )
    fields = {name: kind for kind, name in typed_fields}
    if log is None or re.sub(rf"  {scalar} \w+;\n", "", log[1]):
        raise ValueError(
            "only scalar integer and pointer observations are supported"
        )
    outputs = set()
    var_struct = re.search(r"typedef struct \{\n([^{}]*)\} vars_t;", text)
    memory = (
        set(re.findall(r"\*(\w+)", var_struct[1])) if var_struct else set()
    )
    calls = []
    bodies = []
    types = dict(fields)
    for pid, params, body in functions:
        names = []
        for param in params.split(","):
            if param in ("", "void"):
                continue
            match = re.fullmatch(rf"({scalar}) \*(\w+)", param)
            if not match:
                raise ValueError(f"unsupported generated parameter: {param}")
            name = match[2]
            if name in types and types[name] != match[1]:
                raise ValueError(f"inconsistent parameter type for {name}")
            types[name] = match[1]
            if name.startswith("out_"):
                if not name.startswith(f"out_{pid}_"):
                    raise ValueError("output belongs to another thread")
                outputs.add(name)
            else:
                memory.add(name)
            names.append(name)
        if '"cc","memory"' not in body:
            raise ValueError("generated asm lacks expected memory clobber")
        if delayed and pid == "1":
            # Identity arithmetic delays only the older load address. No new
            # memory events or cross-thread dependencies are introduced. Both
            # addresses remain x, so the original CoRR state set still applies.
            old = '"lw %[x5],0(%[x6])\\n"'
            if body.count(old) != 1 or body.count(':[x6] "r" (x)') != 1:
                raise ValueError("CoRR probe no longer matches generated code")
            body = body.replace(old, '"lw %[x5],0(%[delayed])\\n"')
            body = body.replace(
                ':[x6] "r" (x)',
                ':[x6] "r" (x),[delayed] "r" ((int *)delayed_x)',
            )
            body = (
                "uintptr_t delayed_x = (uintptr_t)x;\n"
                "unsigned long one = 1;\n"
                '__asm__ volatile(".rept 64\\n\\tdivu %0,%0,%1\\n\\t.endr"\n'
                '    : "+&r"(delayed_x) : "r"(one));\n' + body
            )
        body = body.replace(".aq.rl ", ".aqrl ")
        bodies.append(
            f"__attribute__((noinline)) static void code{pid}({params}) {{\n{body}\n}}\n"
        )
        calls.append(
            f"case {pid}: code{pid}({','.join('&'+n for n in names)}); break;"
        )
    initial = {}
    for name in sorted(memory):
        values = re.findall(
            rf"^    \*{re.escape(name)} = ([^;]+);$", text, re.MULTILINE
        )
        if len(values) != 1:
            raise ValueError(f"unsupported initialization for {name}")
        value = values[0]
        for symbol in memory:
            value = re.sub(
                rf"(?<![\w&])(?:_vars->)?\b{re.escape(symbol)}\b",
                "&" + symbol,
                value,
            )
        initial[name] = value
        types.setdefault(name, fields.get(name, "uint64_t"))
    declarations = "\n".join(
        f"static {types[name]} {name} __attribute__((aligned(64)));"
        for name in sorted(memory | outputs)
    )
    reads = []
    for index, location in enumerate(locations):
        if location == "__completion":
            reads.append(f"case {index}: return 0;")
            continue
        name = (
            location[1:-1]
            if location.startswith("[")
            else "out_" + location.replace(":", "_")
        )
        if name not in memory | outputs:
            raise ValueError(f"missing generated observation: {location}")
        reads.append(f"case {index}: return (int64_t){name};")
    init = "\n".join(f"{name} = {value};" for name, value in initial.items())

    def literal(value):
        if isinstance(value, str):
            if value not in memory:
                raise ValueError(f"unknown symbolic outcome: {value}")
            return f"(int64_t)(uintptr_t)&{value}"
        return "INT64_MIN" if value == -(1 << 63) else f"INT64_C({value})"

    allowed = ",\n".join(
        "{" + ",".join(map(literal, state)) + "}" for state in states
    )
    # Preserve the generator's licence notice in the generated artifact.
    notice = text[: text.index("/* Parameters */")]
    header = f"""{notice}
#define NOUT {len(locations)}
#define NSTATES {len(states)}
{declarations}
{''.join(bodies)}
static void litmus_init(void) {{ {init} }}
static void litmus_run(unsigned role) {{ switch(role) {{ {' '.join(calls)} }} }}
static int64_t litmus_read(unsigned index) {{ switch(index) {{ {' '.join(reads)} }} return 0; }}
static int litmus_allowed(const int64_t *values) {{
    static const int64_t allowed[][NOUT] = {{ {allowed} }};
    for (unsigned i = 0; i < sizeof(allowed)/sizeof(allowed[0]); ++i) {{
        unsigned j = 0;
        while (j < NOUT && values[j] == allowed[i][j]) ++j;
        if (j == NOUT) return 1;
    }}
    return 0;
}}
"""
    return header, initial


def prepare(args, source, delayed):
    case = source.stem
    name = case + ("-address-delay" if delayed else "")
    directory = args.outdir / "images" / name
    directory.mkdir(parents=True, exist_ok=False)
    model_command = [
        args.herd7,
        "-model",
        args.model_path,
        "-checkfilter",
        "false",
        source,
    ]
    stdout, stderr = invoke(model_command)
    (directory / "herd.log").write_text(stdout + stderr)
    if stderr:
        raise ValueError(f"herd7 diagnostics: {stderr}")
    locations, states = parse_states(stdout)
    generated = directory / "upstream"
    generated.mkdir()
    generator_command = [
        args.litmus7,
        "-mode",
        "presi",
        "-ascall",
        "true",
        "-avail",
        "2",
        "-s",
        "1",
        "-r",
        "1",
        "-o",
        generated,
        source,
    ]
    stdout, stderr = invoke(generator_command)
    (directory / "litmus7.log").write_text(stdout + stderr)
    generated_text = (generated / f"{case}.c").read_text()
    oracle_source = source
    if "inline static int filter_cond(" in generated_text:
        # Include filter-only result registers in the joint state. Comparing
        # the unfiltered projection without these registers would lose the
        # intended LR/SC success constraint.
        observed = {
            loc[1:-1] if loc.startswith("[") else loc for loc in locations
        }
        observed.update(
            f"{pid}:{reg}"
            for pid, reg in re.findall(r"\bout_(\d+)_(x\d+)\b", generated_text)
        )
        clause = "locations [" + ";".join(sorted(observed)) + ";]\n"
        oracle_text = source.read_text()
        if re.search(r"\blocations\s*\[", oracle_text):
            oracle_text = re.sub(
                r"\blocations\s*\[[^]]*\]\s*;?", clause, oracle_text, count=1
            )
        else:
            oracle_text, changed = re.subn(
                r"(?m)^(?=\s*filter\b)", lambda _: clause, oracle_text, count=1
            )
            if changed != 1:
                raise ValueError("could not extend filter observations")
        oracle_source = directory / "oracle.litmus"
        oracle_source.write_text(oracle_text)
        model_command[-1] = oracle_source
        stdout, stderr = invoke(model_command)
        (directory / "herd.log").write_text(stdout + stderr)
        if stderr:
            raise ValueError(f"herd7 diagnostics: {stderr}")
        locations, states = parse_states(stdout)
    header, initial = extract_threads(
        generated_text, locations, states, delayed
    )
    (directory / "test.h").write_text(header)
    model = {
        "case": case,
        "probe": delayed,
        "checkfilter": False,
        "locations": locations,
        "allowed": states,
        "initial": initial,
        "source": str(source.relative_to(SUITE)),
        "source_sha256": sha256(source),
        "oracle_source": str(oracle_source),
        "oracle_sha256": sha256(oracle_source),
        "herd_command": list(map(str, model_command)),
        "litmus_command": list(map(str, generator_command)),
        "address_divisions": 64 if delayed else 0,
    }
    write_json(directory / "model.json", model)
    return name, directory, model


def signed_hex(value):
    n = int(value, 16)
    return n - (1 << 64) if n >= 1 << 63 else n


def check_observations(text, model, trials):
    observations = Counter()
    lines = text.splitlines()
    outcome_lines = [
        line for line in lines if line.startswith("LITMUS_OUTCOME")
    ]
    for line in outcome_lines:
        match = re.fullmatch(r"LITMUS_OUTCOME ([0-9a-f ]+)", line)
        if not match:
            raise ValueError("malformed UART observation")
        tokens = match[1].split()
        if len(tokens) != len(model["locations"]) + 1 or any(
            len(v) != 16 for v in tokens
        ):
            raise ValueError("malformed UART observation")
        count = int(tokens[0], 16)
        values = tuple(signed_hex(v) for v in tokens[1:])
        if count == 0 or values in observations:
            raise ValueError("zero count or duplicate outcome")
        observations[values] = count
    forbidden = sorted(
        set(observations) - {tuple(s) for s in model["allowed"]}
    )
    count = sum(observations.values())
    text = re.sub(
        r"(LITMUS_DONE [0-9a-f]{16})(?=Profile report: )", r"\1\n", text
    )
    lines = text.splitlines()
    done = [line for line in lines if line.startswith("LITMUS_DONE")]
    bad = [line for line in lines if line.startswith("LITMUS_FORBIDDEN")]
    complete = count == trials and done == [f"LITMUS_DONE {trials:016x}"]
    failed = (
        0 < count <= trials
        and len(forbidden) == 1
        and not done
        and bad == [f"LITMUS_FORBIDDEN {count:016x}"]
        and observations[forbidden[0]] == 1
    )
    status = "incomplete"
    if forbidden and failed:
        status = "forbidden"
    elif not forbidden and not bad and complete:
        status = "pass"
    return {
        "status": status,
        "trials": sum(observations.values()),
        "forbidden": forbidden,
        "outcomes": [
            {"values": values, "count": count}
            for values, count in sorted(observations.items())
        ],
    }


def simulate(args, image, topology, swap, baseline):
    name, prepared, model = image
    binary = args.baseline if baseline else args.gem5
    label = (
        f"{'baseline' if baseline else 'fixed'}-{topology}-swap{swap}-{name}"
    )
    directory = args.outdir / label
    directory.mkdir()
    elf = directory / "test.elf"
    raw = directory / "test.bin"
    command = [
        args.cc,
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-ffreestanding",
        "-fno-builtin",
        "-nostdlib",
        "-nostartfiles",
        "-static",
        "-fno-pic",
        "-fno-pie",
        "-mcmodel=medany",
        "-msmall-data-limit=0",
        "-mno-relax",
        "-march=rv64ima_zicsr_zifencei",
        "-mabi=lp64",
        "-Wl,--build-id=none",
        "-T",
        HERE / "link.ld",
        "-I",
        prepared,
        f"-DTRIALS={args.trials}",
        f"-DSWAP_HARTS={swap}",
        HERE / "start.S",
        HERE / "runtime.c",
        "-o",
        elf,
    ]
    stdout, stderr = invoke(command)
    (directory / "build.log").write_text(stdout + stderr)
    invoke([args.tool_prefix + "objcopy", "-O", "binary", elf, raw])
    disassembly, _ = invoke([args.tool_prefix + "objdump", "-d", elf])
    (directory / "test.dis").write_text(disassembly)
    run = [
        binary,
        "--listener-mode=off",
        f"--outdir={directory}/m5out",
        args.gem5_root / "configs/example/smt_idealkmhv3.py",
        "--raw-cpt",
        f"--generic-rv-cpt={raw}",
        "--disable-difftest",
        "--mem-type=SimpleMemory",
        "--mem-size=16GB",
        "--maxinsts=5000000",
        "--abs-max-tick=10000000000",
    ]
    if topology == "spmt":
        run.append("--enable-spmt")
    write_json(
        directory / "command.json",
        {
            "build": list(map(str, command)),
            "run": list(map(str, run)),
            "gem5_sha256": args.binary_hashes[str(binary)],
            "image_sha256": sha256(raw),
            "gem5_head": args.gem5_head,
            "suite_head": args.suite_head,
        },
    )
    with (directory / "sim.log").open("w") as log:
        try:
            completed = subprocess.run(
                list(map(str, run)),
                cwd=args.gem5_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = None
    text = (directory / "sim.log").read_text(errors="replace")
    concrete_model = dict(model)
    if any(isinstance(v, str) for row in model["allowed"] for v in row):
        symbols, _ = invoke([args.tool_prefix + "nm", "-n", elf])
        addresses = {
            name: int(address, 16)
            for address, name in re.findall(
                r"^([0-9a-f]+) [A-Za-z] (\w+)$", symbols, re.MULTILINE
            )
        }
        concrete_model["allowed"] = [
            [addresses[v] if isinstance(v, str) else v for v in row]
            for row in model["allowed"]
        ]
        write_json(directory / "concrete-model.json", concrete_model)
    result = check_observations(text, concrete_model, args.trials)
    if (
        "because m5_exit instruction encountered" not in text
        or returncode != 0
    ):
        result["status"] = "incomplete"
    result.update(
        name=label,
        baseline=baseline,
        returncode=returncode,
        locations=model["locations"],
    )
    write_json(directory / "result.json", result)
    print(
        f"{label}: {result['status']} ({result['trials']} trials)", flush=True
    )
    return result


def executable(name):
    found = shutil.which(name)
    if not found:
        raise ValueError(f"executable not found: {name}")
    return Path(found).resolve()


def tool_manifest(args, extra_tools=()):
    """Check the generated-C format version and fingerprint its dependencies."""
    versions = {}
    for tool in (args.herd7, args.litmus7):
        versions[tool.name] = invoke([tool, "-version"])[0].strip()
        if not re.fullmatch(r"7\.58(?:,.*)?", versions[tool.name]):
            raise ValueError(
                "the generated-C adapter requires herdtools7 7.58"
            )
    libdir = Path(invoke([args.herd7, "-libdir"])[0].strip()).resolve()
    args.model_path = libdir / "riscv.cat"
    if not args.model_path.is_file():
        raise ValueError("herd7's installed riscv.cat was not found")
    tools = [args.herd7, args.litmus7, args.cc, *extra_tools]
    tools.extend(
        executable(args.tool_prefix + name)
        for name in ("as", "ld", "objcopy", "objdump", "nm")
    )
    return {
        "versions": versions,
        "tools": {str(p): sha256(p) for p in tools},
        "compiler": invoke([args.cc, "--version"])[0].splitlines()[0],
        "model_files": {p.name: sha256(p) for p in libdir.glob("*.cat")},
        "gem5_config_sha256": sha256(
            args.gem5_root / "configs/example/smt_idealkmhv3.py"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--herd7", default="herd7")
    parser.add_argument("--litmus7", default="litmus7")
    parser.add_argument("--gem5-root", type=Path, required=True)
    parser.add_argument(
        "--gem5", help="default: GEM5_ROOT/build/RISCV/gem5.opt"
    )
    parser.add_argument("--baseline")
    parser.add_argument(
        "--corr-address-delay",
        action="store_true",
        help="add a CoRR address-delay probe; also enabled by --baseline",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--tool-prefix", default="riscv64-linux-gnu-")
    parser.add_argument(
        "--test",
        action="append",
        type=Path,
        help=".litmus path relative to this repository; repeatable",
    )
    parser.add_argument(
        "--topology",
        action="append",
        choices=("smt", "spmt"),
        help="default: both; repeatable",
    )
    args = parser.parse_args()
    if min(args.trials, args.jobs, args.timeout) < 1:
        parser.error("trials, jobs and timeout must be positive")
    args.gem5_root = args.gem5_root.resolve()
    config = args.gem5_root / "configs/example/smt_idealkmhv3.py"
    if not config.is_file():
        parser.error(
            "gem5-root does not contain configs/example/smt_idealkmhv3.py"
        )
    tests = [((SUITE / p).resolve()) for p in (args.test or DEFAULT_TESTS)]
    if len({p.stem for p in tests}) != len(tests):
        parser.error("selected tests must have distinct names")
    for test in tests:
        if (
            not test.is_relative_to(SUITE)
            or test.suffix != ".litmus"
            or not test.is_file()
        ):
            parser.error(f"expected a .litmus file in this repository: {test}")
        if not re.fullmatch(r"[A-Za-z0-9+._-]+", test.stem):
            parser.error(f"unsupported output file name: {test.stem}")
    if args.baseline and CORR not in tests:
        parser.error(
            "baseline detection requires tests/non-mixed-size/CO/CoRR.litmus"
        )
    args.corr_address_delay |= bool(args.baseline)
    if args.corr_address_delay and CORR not in tests:
        parser.error("the address-delay probe requires CoRR.litmus")
    args.topology = args.topology or ["smt", "spmt"]
    if len(set(args.topology)) != len(args.topology):
        parser.error("duplicate topology")
    args.outdir = args.outdir.resolve()
    if args.outdir.exists():
        parser.error("outdir must not exist; preserve earlier evidence")
    args.herd7, args.litmus7 = executable(args.herd7), executable(args.litmus7)
    args.gem5 = executable(
        args.gem5 or str(args.gem5_root / "build/RISCV/gem5.opt")
    )
    args.baseline = executable(args.baseline) if args.baseline else None
    args.cc = executable(args.tool_prefix + "gcc")
    tools = tool_manifest(args)
    args.gem5_head = invoke(["git", "rev-parse", "HEAD"], cwd=args.gem5_root)[
        0
    ].strip()
    args.suite_head = invoke(["git", "rev-parse", "HEAD"], cwd=SUITE)[
        0
    ].strip()
    args.binary_hashes = {
        str(p): sha256(p) for p in (args.gem5, args.baseline) if p
    }
    args.outdir.mkdir(parents=True)
    write_json(
        args.outdir / "manifest.json",
        {
            **tools,
            "binaries": args.binary_hashes,
            "gem5_head": args.gem5_head,
            "suite_head": args.suite_head,
            "suite_origin": invoke(
                ["git", "remote", "get-url", "origin"], cwd=SUITE
            )[0].strip(),
            "gem5_root": str(args.gem5_root),
            "tests": {str(p.relative_to(SUITE)): sha256(p) for p in tests},
            "topologies": args.topology,
            "sources": {
                str(p.relative_to(HERE)): sha256(p)
                for p in HERE.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            },
            "trials": args.trials,
            "corr_address_delay": args.corr_address_delay,
        },
    )
    images = [
        prepare(args, test, delayed)
        for test in tests
        for delayed in (
            (False, True)
            if test == CORR and args.corr_address_delay
            else (False,)
        )
    ]
    jobs = [
        (image, topology, swap, False)
        for image in images
        for topology in args.topology
        for swap in (0, 1)
    ]
    if args.baseline:
        jobs += [
            (image, topology, swap, True)
            for image in images
            if image[2]["probe"]
            for topology in args.topology
            for swap in (0, 1)
        ]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(simulate, args, *job) for job in jobs]
        results = [future.result() for future in futures]
    write_json(args.outdir / "results.json", results)
    ok = all(r["status"] == "pass" for r in results if not r["baseline"])
    baseline = [r for r in results if r["baseline"]]
    if baseline:
        ok &= all(
            r["status"] in ("pass", "forbidden") for r in baseline
        ) and any(r["status"] == "forbidden" for r in baseline)
    return 0 if ok else 1


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
