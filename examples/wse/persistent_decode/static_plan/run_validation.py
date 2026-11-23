# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reproduce the finite-trace numeric baseline and the static protocol experiment."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from run_deployment_checks import run as run_deployment_checks
from run_kv_quota_checks import run as run_kv_quota_checks
from run_probe import run as run_probe


def main(source_root, compiler, build):
    here = Path(__file__).resolve().parent
    build.mkdir(parents=True, exist_ok=True)
    output = build / "summary"
    output.mkdir(exist_ok=True)
    sources = [
        "src/wse/platform/sim/persistent/device.cpp",
        "src/wse/runtime/persistent/decode.cpp",
        "src/wse/runtime/persistent/admission.cpp",
        "src/wse/runtime/persistent/prefill_state.cpp",
        "examples/wse/persistent_decode/decode_sim.cpp",
    ]
    numeric = build / "wse_decode_current.exe"
    command = [str(compiler), "-std=c++17", "-O2"]
    for directory in (
        "src/wse/platform/sim/persistent",
        "src/wse/runtime/persistent",
        "src/wse/platform/include",
        "examples/wse/persistent_decode",
    ):
        command.append("-I" + str(source_root / directory))
    command += [str(source_root / source) for source in sources]
    command += [str(here / "finite_trace_input.cpp"), "-o", str(numeric)]
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    (build / "numeric_build.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(process.stderr)
    subprocess.run(
        [
            sys.executable,
            str(source_root / "examples/wse/persistent_decode/run_checks.py"),
            "--binary",
            str(numeric),
            "--output",
            str(build / "numeric"),
        ],
        check=True,
    )
    validation = json.loads((build / "numeric/validation.json").read_text())
    (output / "numeric_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    run_probe(source_root, compiler, build / "protocol", output)
    run_deployment_checks(
        here / "deployment.min.json", source_root, compiler, build / "deployment", output / "deployment_validation.json"
    )
    run_kv_quota_checks(source_root, compiler, build / "quota", output / "kv_quota_validation.json")
    inputs = {source: source_root / source for source in sources}
    inputs |= {
        "examples/wse/persistent_decode/" + name: source_root / "examples/wse/persistent_decode" / name
        for name in ("run_checks.py", "reference.py", "prefill.py")
    }
    inputs |= {
        name: here / name
        for name in (
            "static_plan_probe.cpp",
            "run_probe.py",
            "finite_trace_input.cpp",
            "run_validation.py",
            "validation_contract.json",
            "deployment.min.json",
            "program.manifest.json",
            "prepare_deployment.py",
            "deployment.schema.json",
            "run_deployment_checks.py",
            "kv_quota.h",
            "run_kv_quota_checks.py",
        )
    }
    evidence = {
        "date": "2026-11-23",
        "compiler": subprocess.check_output([str(compiler), "--version"], text=True).splitlines()[0],
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
        "numeric": {
            "positive": sum(case.get("independent_numpy_oracle") == "passed" for case in validation.values()),
            "negative": sum(case.get("rejected_as_expected", False) for case in validation.values()),
            "live_stdin_tested": False,
            "note": "Original finite-trace source; live stdin entry replaced by an explicitly throwing link shim",
        },
        "protocol": "static_validation.json",
        "deployment": "deployment_validation.json",
        "kv_quota": "kv_quota_validation.json",
        "not_verified": ["integrated full-model static codegen", "native WSE", "native batcher ABI", "hardware timing"],
    }
    (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root.resolve(), args.compiler.resolve(), args.build.resolve())
