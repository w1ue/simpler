# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the deployment input seam, including an actual Memory/Fabric execution."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from prepare_deployment import DeploymentError, prepare
from run_probe import check_run


def check_rejections(original, write_case):
    negative = {}

    def reject(name, mutate, expected):
        data = copy.deepcopy(original)
        mutate(data)
        path, directory = write_case(name, data)
        try:
            prepare(path, directory)
        except DeploymentError as error:
            assert expected in str(error), (name, str(error))
            assert not (directory / "prepared_deployment.json").exists()
            negative[name] = {"rejected_before_deployment": True, "diagnostic": str(error)}
        else:
            raise AssertionError(name + " should have been rejected")

    for field in original:
        reject("missing_" + field, lambda d, key=field: d.pop(key), "missing or unknown fields")
    reject("unknown_field", lambda d: d.update(requests=[]), "missing or unknown fields")
    reject(
        "unsupported_version", lambda d: d.update(format="simpler.wse.deployment.v99"), "unsupported deployment format"
    )
    reject("manifest_digest", lambda d: d["program"].update(sha256="0" * 64), "manifest digest mismatch")
    reject("missing_manifest", lambda d: d["program"].update(manifest="absent.json"), "manifest file missing")
    reject("home_out_of_range", lambda d: d["placement"]["home_core_by_slot"].__setitem__(0, 16), "Core range invalid")
    reject("expert_collision", lambda d: d["placement"]["routed_expert_core"].__setitem__(0, 2), "owner collision")
    reject("role_collision", lambda d: d["placement"]["routed_expert_core"].__setitem__(0, 0), "role owner collision")
    reject("non_home_bailu_owner", lambda d: d["placement"]["home_core_by_slot"].__setitem__(0, 1), "home must be")
    reject("zero_credit", lambda d: d["transport"].update(inbox_entries_per_core=0), "positive bounded integer")
    reject("bool_capacity", lambda d: d["capacity_bytes"].update(sram_per_core=True), "positive bounded integer")
    reject("unaligned_capacity", lambda d: d["capacity_bytes"].update(sram_per_core=262143), "128-byte aligned")
    reject("insufficient_sram", lambda d: d["capacity_bytes"].update(sram_per_core=128), "capacity exceeded: sram")
    reject("insufficient_bailu", lambda d: d["capacity_bytes"].update(bailu_per_chip=128), "capacity exceeded: bailu")
    reject("insufficient_host", lambda d: d["capacity_bytes"].update(host_per_chip=128), "capacity exceeded: host")
    reject(
        "insufficient_batcher",
        lambda d: d["capacity_bytes"].update(batcher_per_endpoint=128),
        "capacity exceeded: batcher",
    )
    reject(
        "quota_zero_initial", lambda d: d["kv_quota"].update(initial_pages_per_request=0), "positive bounded integer"
    )
    reject("quota_zero_refill", lambda d: d["kv_quota"].update(refill_pages=0), "positive bounded integer")
    reject("quota_watermark", lambda d: d["kv_quota"].update(low_watermark_pages=2), "below initial grant")
    reject("quota_pool_bound", lambda d: d["kv_quota"].update(pool_pages_per_chip=33), "positive bounded integer")
    reject("quota_initial_capacity", lambda d: d["kv_quota"].update(pool_pages_per_chip=3), "initial KV grants exceed")
    reject(
        "quota_control_queue",
        lambda d: d["kv_quota"].update(control_queue_entries_per_endpoint=0),
        "positive bounded integer",
    )
    reject(
        "quota_exhaustion_policy",
        lambda d: d["kv_quota"].update(exhaustion_policy="grow"),
        "unsupported exhaustion_policy",
    )

    manifest_path = Path(original["program"]["manifest"])
    manifest_original = json.loads(manifest_path.read_text())
    manifest_original["artifact"]["path"] = str(
        (manifest_path.parent / manifest_original["artifact"]["path"]).resolve()
    )
    for case in ("artifact_digest", "program_contract"):
        data = copy.deepcopy(original)
        program = copy.deepcopy(manifest_original)
        if case == "artifact_digest":
            program["artifact"]["sha256"] = "0" * 64
            expected = "artifact digest mismatch"
        else:
            program["bounds"]["layers"] = 41
            expected = "unsupported compiled program contract"
        path, directory = write_case(case, data)
        manifest = directory / "program.json"
        manifest.write_text(json.dumps(program))
        reject_data = copy.deepcopy(original)
        reject_data["program"] = dict(manifest=str(manifest), sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        path.write_text(json.dumps(reject_data))
        try:
            prepare(path, directory)
        except DeploymentError as error:
            assert expected in str(error)
            negative[case] = {"rejected_before_deployment": True, "diagnostic": str(error)}
        else:
            raise AssertionError(case + " should fail")
    return negative


def run(input_path, source_root, compiler, build, summary_path):
    build.mkdir(parents=True, exist_ok=True)
    original = json.loads(input_path.read_text(encoding="utf-8"))
    original["program"]["manifest"] = str((input_path.parent / original["program"]["manifest"]).resolve())
    positive = {}

    def write_case(name, data):
        directory = build / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "deployment.json"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path, directory

    def execute(name, data):
        path, directory = write_case(name, data)
        plan, artifact = prepare(path, directory)
        binary = directory / "probe.exe"
        command = [
            str(compiler),
            "-std=c++17",
            "-O2",
            "-I" + str(directory),
            "-I" + str(source_root / "src/wse/platform/sim/persistent"),
            "-I" + str(source_root / "src/wse/runtime/persistent"),
            "-I" + str(source_root / "src/wse/platform/include"),
            str(artifact),
            str(source_root / "src/wse/platform/sim/persistent/device.cpp"),
            "-o",
            str(binary),
        ]
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        (directory / "build.log").write_text(process.stdout + process.stderr)
        assert process.returncode == 0, process.stderr
        process = subprocess.run(
            [str(binary), "baseline", str(directory / "run")], check=False, capture_output=True, text=True, timeout=90
        )
        assert process.returncode == 0, process.stderr
        result = json.loads((directory / "run/results.json").read_text())
        assert result.get("deployment_digest") == plan["deployment_digest"], "deployment input was not consumed"
        observed = json.loads((directory / "run/allocations.json").read_text())
        expected = [{key: value for key, value in cell.items() if key != "name"} for cell in plan["allocations"]]
        assert observed == expected, "actual allocations differ from prepared layout"
        tiers = dict(sram=0, bailu=1, batcher=2, host=3)
        expected_regions = [
            {key: value for key, value in region.items() if key != "index"} for region in plan["regions"]
        ]
        for region in expected_regions:
            region["tier"] = tiers[region["tier"]]
        assert result["memory"] == expected_regions
        assert result["transport"] == data["transport"]
        checks = check_run(directory / "run", "baseline")
        events = [json.loads(row) for row in (directory / "run/events.jsonl").read_text().splitlines()]
        for event in events:
            if event["event"] == "LAYER_DONE":
                assert event["actor"] == data["placement"]["home_core_by_slot"][event["slot"]]
        observed_expert_owners = {event["actor"] for event in events if event["event"] == "EXPERT_COMPUTE"}
        expected_expert_owners = set(
            data["placement"]["routed_expert_core"] + data["placement"]["shared_expert_core_by_chip"]
        )
        assert observed_expert_owners == expected_expert_owners
        positive[name] = {
            "passed": True,
            "allocation_count": len(observed),
            "deployment_digest": plan["deployment_digest"],
            "ordered_transfers": checks["ordered_transfers"],
            "queue_stalls": checks["queue_stalls"],
            "credit_stalls": checks["credit_stalls"],
            "fixed_memory_and_oracle": "passed",
        }
        return result

    baseline = execute("baseline", original)
    remapped = copy.deepcopy(original)
    remapped["placement"]["home_core_by_slot"] = [8, 0, 8, 0]
    remapped["placement"]["routed_expert_core"] = [2, 1, 4, 3, 10, 9, 12, 11]
    remapped["placement"]["shared_expert_core_by_chip"] = [6, 14]
    remapped["capacity_bytes"]["sram_per_core"] = 131072
    changed = execute("remapped", remapped)

    def key(row):
        return row["slot"], row["generation"], row["position"]

    assert {key(r): r["token"] for r in baseline["outputs"]} == {key(r): r["token"] for r in changed["outputs"]}
    pressured = copy.deepcopy(original)
    pressured["transport"] = dict(inbox_entries_per_core=1, queue_entries_per_endpoint=2)
    changed = execute("transport_pressure", pressured)
    assert changed["queue_stalls"] > 0 and changed["credit_stalls"] > 0
    assert {key(r): r["token"] for r in baseline["outputs"]} == {key(r): r["token"] for r in changed["outputs"]}
    reordered = dict(reversed(list(original.items())))
    reordered["placement"] = dict(reversed(list(original["placement"].items())))
    path, directory = write_case("reordered_fields", reordered)
    plan, _ = prepare(path, directory)
    assert plan["deployment_digest"] == positive["baseline"]["deployment_digest"]
    positive["reordered_fields"] = {"passed": True, "same_canonical_digest": True}
    relative_plan, _ = prepare(input_path, build / "relative_reference")
    assert relative_plan["deployment_digest"] == positive["baseline"]["deployment_digest"]
    positive["reference_path_independence"] = {"passed": True, "same_canonical_digest": True}

    negative = check_rejections(original, write_case)
    summary = dict(
        positive=positive,
        negative=negative,
        scope="Host-side simulation input adapter + actual WSE Memory/Fabric probe",
        native_simpler_api=False,
        hardware_execution=False,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(positive)} deployment cases and {len(negative)} rejection cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "deployment.min.json")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    run(
        args.input.resolve(),
        args.source_root.resolve(),
        args.compiler.resolve(),
        args.build.resolve(),
        (args.summary or args.build / "deployment_validation.json").resolve(),
    )
