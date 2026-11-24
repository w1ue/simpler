# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build against WSE Memory and verify a bounded static protocol plan."""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from prepare_deployment import prepare


def check_scenario(events, result, case, golden_layers):
    if case == "baseline":
        assert golden_layers[(0, 1, 1, 0)] == 10143, "worked first-layer example differs"
    if case == "slot_reuse":
        assert result["binds"] == 5 and any(row["slot"] == 0 and row["generation"] == 2 for row in result["outputs"])
    if case == "independent_progress":
        closed = next(i for i, event in enumerate(events) if event["event"] == "FETCH_GATE_CLOSED")
        released = next(i for i, event in enumerate(events) if event["event"] == "FETCH_GATE_RELEASED")
        assert any(event["event"] == "LAYER_DONE" and event["slot"] == 1 for event in events[closed:released])
        assert any(
            event["event"] == "LAYER_DONE" and event["slot"] == 2 and event["actor"] == 0
            for event in events[closed:released]
        ), "same-Core request did not progress while R0 waited"
        assert not any(event["event"] == "LAYER_DONE" and event["slot"] == 0 for event in events[closed:released])
    if case == "output_backpressure":
        withheld = next(i for i, event in enumerate(events) if event["event"] == "OUTPUT_ACK_WITHHELD")
        drained = next(
            i for i, event in enumerate(events) if event["event"] == "REQUEST_DRAINED" and event["slot"] == 1
        )
        resumed = next(i for i, event in enumerate(events) if event["event"] == "OUTPUT_ACK" and event["slot"] == 0)
        assert withheld < drained < resumed
        assert not any(event["event"] == "OUTPUT_ACK" and event["slot"] == 0 for event in events[withheld:drained])
    if case == "slot_reuse":
        drained = next(
            i for i, event in enumerate(events) if event["event"] == "REQUEST_DRAINED" and event["slot"] == 0
        )
        rebound = next(
            i
            for i, event in enumerate(events)
            if event["event"] == "HOST_BIND" and event["slot"] == 0 and event["generation"] == 2
        )
        assert drained < rebound and result["binds"] == 5
    if case in {"one_credit", "small_queue"}:
        assert result["credit_stalls"] > 0
    if case == "small_queue":
        assert result["queue_stalls"] > 0
    if case == "token_boundary_cancel":
        assert any(event["event"] == "TOKEN_BOUNDARY_CANCEL" for event in events)
    if case == "host_silent_burst":
        for slot in range(4):
            trace = [event for event in events if event["slot"] == slot and event["generation"] == 1]
            first_ack = next(i for i, event in enumerate(trace) if event["event"] == "OUTPUT_ACK")
            assert sum(event["event"] == "TOKEN_PRODUCED_WITHOUT_HOST_ACK" for event in trace[:first_ack]) == 3
            assert sum(event["event"] == "LAYER_DONE" for event in trace[:first_ack]) == 120


def check_run(directory, case, expected_tokens=None):
    result = json.loads((directory / "results.json").read_text())
    events = [json.loads(row) for row in (directory / "events.jsonl").read_text().splitlines()]
    streams = {}
    golden_layers = {}
    for actual in result["outputs"]:
        key = (actual["slot"], actual["generation"])
        initial, previous_kv, previous_token, position = streams.get(key, (actual["initial"], 5, actual["initial"], 0))
        position += 1
        assert position == actual["position"]
        hidden = (previous_token + previous_kv) % 1000003
        for layer in range(40):
            x = (3 * hidden + position + layer) % 1000003
            experts = list(range(6)) if result["hotspot"] else [(x + layer + j) % 8 for j in range(6)]
            experts.append(8)
            hidden = (
                x
                + sum(
                    (lane + 1) * ((x * (expert + 2) + 17 * layer + expert) % 1000003)
                    for lane, expert in enumerate(experts)
                )
            ) % 1000003
            golden_layers[key + (position, layer)] = hidden
        assert actual["hidden"] == hidden and actual["token"] == hidden % 97, (case, actual, hidden)
        streams[key] = (initial, hidden, hidden % 97, position)
    for event in events:
        if event["event"] == "LAYER_DONE":
            key = tuple(event[name] for name in ("slot", "generation", "position", "layer"))
            assert event["value"] == golden_layers[key], (case, key)
    if expected_tokens is None:
        expected_tokens = (
            15
            if case == "context_boundary"
            else 14
            if case == "slot_reuse"
            else 11
            if case == "token_boundary_cancel"
            else 12
        )
    assert result["tokens"] == expected_tokens
    assert result["all_drained"] and result["launches"] == 1
    assert result["physical_allocations_during_service"] == result["physical_releases_during_service"] == 0
    assert result["per_layer_host_decisions"] == result["host_token_feedback"] == 0
    assert result["last_use_holds"] == expected_tokens * 40 * 7
    assert all(region["used"] <= region["capacity"] for region in result["memory"])
    sequences = defaultdict(list)
    for output in result["outputs"]:
        sequences[output["slot"], output["generation"]].append(output)
    for sequence in sequences.values():
        previous_token = sequence[0]["initial"]
        previous_kv = 5
        for position, output in enumerate(sequence, 1):
            hidden = (previous_token + previous_kv) % 1000003
            for layer in range(40):
                activation = (hidden * 3 + position + layer) % 1000003
                experts = (
                    list(range(6)) if case == "hotspot" else [(activation + layer + lane) % 8 for lane in range(6)]
                )
                values = [(activation * (expert + 2) + 17 * layer + expert) % 1000003 for expert in experts + [8]]
                hidden = (activation + sum((lane + 1) * value for lane, value in enumerate(values))) % 1000003
            assert output["position"] == position
            assert (output["hidden"], output["token"]) == (hidden, hidden % 97)
            previous_kv, previous_token = hidden, hidden % 97
    transfers = defaultdict(list)
    holds = {}
    for event in events:
        if event["transfer_id"]:
            transfers[event["transfer_id"]].append(event["event"])
        identity = tuple(event[key] for key in ("slot", "generation", "position", "layer", "lane"))
        if event["event"] == "INPUT_LEASE_ACQUIRED":
            assert identity not in holds
            holds[identity] = event["tick"]
        elif event["event"] == "INPUT_LAST_USE":
            assert holds.pop(identity) < event["tick"], "last use was collapsed into mailbox dequeue"
    assert not holds
    assert all(order == ["SEND", "COPY_BEGIN", "DELIVER", "PUBLISH", "CONSUMED"] for order in transfers.values())
    assert sum(event["event"] == "LAYER_DONE" for event in events) == expected_tokens * 40
    assert events[-1]["event"] == "SERVICE_DRAINED"
    assert sum(event["event"] == "HOST_BIND" for event in events) == result["binds"]
    for index, event in enumerate(events):
        if event["event"] == "HOST_BIND" and event["generation"] > 1:
            assert any(
                old["event"] == "REQUEST_DRAINED"
                and old["slot"] == event["slot"]
                and old["generation"] == event["generation"] - 1
                for old in events[:index]
            )
    check_scenario(events, result, case, golden_layers)
    assert sum(event["event"] == "LOCAL_TOKEN_FEEDBACK" for event in events) == expected_tokens
    return {
        key: result[key]
        for key in (
            "tokens",
            "binds",
            "physical_allocations_before_launch",
            "queue_stalls",
            "credit_stalls",
            "last_use_holds",
        )
    } | {
        "passed": True,
        "ordered_transfers": len(transfers),
        "layer_completions": expected_tokens * 40,
        "independent_integer_oracle": "passed",
        "fixed_memory_usage": "passed",
        "events": len(events),
    }


def run(source_root, compiler, build, output):
    build.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    prepared, artifact = prepare(here / "deployment.min.json", build)
    device = source_root / "src/wse/platform/sim/persistent/device.cpp"
    binary = build / "static_plan_probe.exe"
    command = [
        str(compiler),
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-I" + str(build),
        "-I" + str(source_root / "src/wse/platform/sim/persistent"),
        "-I" + str(source_root / "src/wse/runtime/persistent"),
        "-I" + str(source_root / "src/wse/platform/include"),
        str(artifact),
        str(device),
        "-o",
        str(binary),
    ]
    compiled = subprocess.run(command, check=False, capture_output=True, text=True)
    (build / "build.log").write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise RuntimeError(compiled.stderr)
    contract = json.loads((here / "validation_contract.json").read_text())
    summary: dict = {"positive": {}, "negative": {}}
    for case in contract["positive_cases"]:
        directory = build / case
        process = subprocess.run(
            [str(binary), case, str(directory)], check=False, capture_output=True, text=True, timeout=90
        )
        assert process.returncode == 0, (case, process.stderr)
        actual = json.loads((directory / "results.json").read_text())
        assert actual["deployment_digest"] == prepared["deployment_digest"]
        summary["positive"][case] = check_run(directory, case)
        print(case, summary["positive"][case]["tokens"], "tokens: passed", flush=True)
    diagnostics = {
        "context_overflow": "context exceeds static profile",
        "slot_busy": "request slot still busy",
        "overwrite_inflight": "write while transfer owns allocation",
        "premature_reclaim": "reclaim before transfer consumed",
        "stale_completion": "stale result identity",
        "duplicate_completion": "duplicate result identity",
        "late_allocate": "physical allocation after launch",
        "overwrite_after_dequeue": "input lease still owned",
    }
    assert set(diagnostics) == set(contract["negative_cases"])
    for case, expected in diagnostics.items():
        process = subprocess.run(
            [str(binary), case, str(build / case)], check=False, capture_output=True, text=True, timeout=30
        )
        assert process.returncode != 0 and expected in process.stderr, (case, process.stderr)
        summary["negative"][case] = {"rejected_as_expected": True, "diagnostic": process.stderr.strip()}
    summary["source_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in {
            "static_plan_probe.cpp": here / "static_plan_probe.cpp",
            "run_probe.py": Path(__file__),
            "kv_quota.h": here / "kv_quota.h",
            "hardware_transport.h": here / "hardware_transport.h",
            "device.cpp": device,
            "device.h": device.with_suffix(".h"),
            "program.h": source_root / "src/wse/runtime/persistent/program.h",
            "deployment.min.json": here / "deployment.min.json",
            "program.manifest.json": here / "program.manifest.json",
            "prepare_deployment.py": here / "prepare_deployment.py",
        }.items()
    }
    summary["limits"] = contract["not_claimed"]
    summary["scope"] = (
        "Protocol scalar workload over actual WSE Memory and a bounded hardware-path model; no AICore interpreter"
    )
    summary["deployment_digest"] = prepared["deployment_digest"]
    summary["generated_header_sha256"] = hashlib.sha256((build / "deployment_config.h").read_bytes()).hexdigest()
    (output / "static_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Passed {len(summary['positive'])} positive and {len(summary['negative'])} negative static-plan cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(
        args.source_root.resolve(),
        args.compiler.resolve(),
        args.build.resolve(),
        (args.output or args.build / "summary").resolve(),
    )
