# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check hardware-path contention, one-way delivery and bounded background work."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from prepare_deployment import HARDWARE, prepare
from run_probe import check_run


def rows(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def check_contexts(trace):
    contexts, data = Counter(), Counter()
    for event in trace:
        if event["path"] != 2:
            continue
        ep = event["endpoint"]
        delta = 1 if event["event"] == "BATCHER_ISSUED" else -1 if event["event"] == "RECEIVER_VISIBLE" else 0
        contexts[ep] += delta
        data[ep] += delta * (event["kind"] < 100)
        assert 0 <= contexts[ep] <= HARDWARE["batcher_wait_entries"]
        assert 0 <= data[ep] < HARDWARE["batcher_wait_entries"]
    assert not any(contexts.values())


def check_paths(directory):
    trace = rows(directory / "transport_events.jsonl")
    streams = defaultdict(list)
    for event in trace:
        if event["transfer_id"]:
            streams[event["transfer_id"]].append(event)
    imports = []
    for events in streams.values():
        accepted = events[0]
        assert accepted["event"] == "TRANSPORT_ACCEPTED"
        path, kind = accepted["path"], accepted["kind"]
        names = [event["event"] for event in events]
        if kind in {8, 9}:
            assert path == 0 and "DIRECT_IMPORT_VISIBLE" in names
            assert "BATCHER_BUDGET_USED" not in names
            imports.append(next(event for event in events if event["event"] == "DIRECT_IMPORT_VISIBLE"))
        if kind in {1, 5, 6, 7, 10, 100, 101, 102, 103, 104}:
            assert path == 2, (kind, path)
        if path == 1:
            sent = next(event for event in events if event["event"] == "CORE_PUSH_SUBMITTED")
            visible = next(event for event in events if event["event"] == "RECEIVER_VISIBLE")
            assert visible["tick"] - sent["tick"] == HARDWARE["noc_latency_ticks"]
            assert "BATCHER_BUDGET_USED" not in names
    used = Counter((e["tick"], e["endpoint"]) for e in trace if e["event"] == "BATCHER_BUDGET_USED")
    assert used and max(used.values()) == 1, "data/control duplicated a physical batcher budget"
    check_contexts(trace)
    service = Counter()
    for event in trace:
        if event["event"] == "EXTERNAL_SERVICE" and event["bank"] != 2**32 - 1:
            service[event["tick"], event["endpoint"] // 4, event["bank"]] += event["bytes"]
    assert service and max(service.values()) <= HARDWARE["bailu_bytes_per_tick"]
    links = Counter()
    for event in trace:
        if event["event"] == "HOST_LINK_SERVICE":
            direction = (
                0 if event["kind"] in {1, 8, 9, 7, 101, 103} else 1 if event["kind"] in {6, 10, 100, 102, 104} else 2
            )
            links[event["tick"], event["endpoint"] // 4, direction] += event["bytes"]
    assert links and max(links.values()) <= HARDWARE["external_bytes_per_tick"]
    data = rows(directory / "events.jsonl")
    for event in data:
        if event["event"] == "ACTIVATION_INSTALLED":
            assert any(
                e["slot"] == event["slot"] and e["generation"] == event["generation"] and e["tick"] < event["tick"]
                for e in imports
            )
    return trace, data, len(streams)


def check_overlap(directory, case, trace, data):
    if case == "hw_overlap":
        second = next(e for e in data if e["event"] == "LOCAL_TOKEN_FEEDBACK" and e["slot"] == 0 and e["position"] == 2)
        for kind in (5, 6):
            done = next(
                e
                for e in trace
                if e["event"] == "SOURCE_READ_DONE" and e["kind"] == kind and e["slot"] == 0 and e["position"] == 1
            )
            assert second["tick"] < done["tick"], "second token waited for previous writeback/output"
        assert any(e["event"] == "KV_LOCAL_HIT" and e["slot"] == 0 and e["position"] == 2 for e in data)
    if case in {"hw_dirty_pressure", "hw_output_pressure"}:
        name = "DIRTY_RING_BLOCKED" if case == "hw_dirty_pressure" else "OUTPUT_RING_BLOCKED"
        blocked = next(e for e in data if e["event"] == name and e["slot"] == 0)
        assert blocked["position"] == 5
        assert any(
            e["event"] == "LOCAL_TOKEN_FEEDBACK"
            and e["slot"] == 0
            and e["position"] == 6
            and e["tick"] > blocked["tick"]
            for e in data
        )
    if case == "hw_push":
        for layer in range(40):
            completed = next(
                e
                for e in data
                if e["event"] == "LAYER_DONE" and e["slot"] == 0 and e["position"] == 1 and e["layer"] == layer
            )
            assert any(
                e["event"] == "CREDIT_RETURNED"
                and e["slot"] == 0
                and e["position"] == 1
                and e["layer"] == layer
                and e["tick"] > completed["tick"]
                for e in trace
            )
    if case == "hw_cancel_import":
        closed = next(e for e in trace if e["event"] == "IMPORT_CANCEL_GATE_CLOSED")
        visible = next(e for e in trace if e["event"] == "DIRECT_IMPORT_VISIBLE" and e["slot"] == 0)
        drained = next(e for e in data if e["event"] == "REQUEST_DRAINED" and e["slot"] == 0)
        rebound = next(e for e in data if e["event"] == "HOST_BIND" and e["slot"] == 0 and e["generation"] == 2)
        assert closed["tick"] < visible["tick"] <= drained["tick"] <= rebound["tick"]
        assert not any(e["event"] == "ACTIVATION_INSTALLED" and e["slot"] == 0 and e["generation"] == 1 for e in data)
        quota = json.loads((directory / "kv_quota.json").read_text())
        assert quota["outcomes"][0]["reason"] == "cancelled_before_activation"


def check_contention(case, trace, data):
    closed = next(e for e in trace if e["event"] == "BAILU_READ_GATE_CLOSED")
    opened = next(e for e in trace if e["event"] == "BAILU_READ_GATE_OPENED")
    during = [e for e in data if closed["tick"] <= e["tick"] < opened["tick"]]
    peer = sum(e["event"] == "CORE_COMPUTE_TILE" and e["slot"] == 2 and e["actor"] == 0 for e in during)
    assert (peer == 0) if case == "hw_direct_blocking" else (peer > 0)
    assert any(e["event"] == "LAYER_DONE" and e["slot"] == 1 for e in during)
    background = [e for e in trace if e["kind"] == 9 and e["event"] == "EXTERNAL_SERVICE"]
    assert background and all(e["bank"] == closed["bank"] for e in background)
    assert any(
        e["kind"] == 101 and e["event"] == "RECEIVER_VISIBLE" and closed["tick"] < e["tick"] < opened["tick"]
        for e in trace
    )
    return peer


def run(source_root, compiler, build, summary):
    here = Path(__file__).resolve().parent
    build.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    prepare(here / "deployment.min.json", build)
    binary = build / "probe.exe"
    process = subprocess.run(
        [
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
            str(here / "static_plan_probe.cpp"),
            str(source_root / "src/wse/platform/sim/persistent/device.cpp"),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    (build / "build.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stderr
    positive = {}
    for case in (
        "hw_paths",
        "hw_push",
        "hw_contention",
        "hw_direct_blocking",
        "hw_overlap",
        "hw_dirty_pressure",
        "hw_output_pressure",
        "hw_cancel_import",
        "hw_background_stream",
    ):
        directory = build / case
        process = subprocess.run(
            [str(binary), case, str(directory)], check=False, capture_output=True, text=True, timeout=60
        )
        assert process.returncode == 0, (case, process.stderr)
        count = 24 if case in {"hw_dirty_pressure", "hw_output_pressure"} else 11 if case == "hw_cancel_import" else 12
        checks = check_run(directory, case, count)
        trace, data, transfers = check_paths(directory)
        check_overlap(directory, case, trace, data)
        if case == "hw_background_stream":
            background = [e for e in trace if e["kind"] == 9]
            assert any(e["event"] == "HOST_LINK_SERVICE" for e in background)
            assert any(e["event"] == "DIRECT_IMPORT_VISIBLE" for e in background)
            assert not any(e["event"] == "BATCHER_BUDGET_USED" for e in background)
        if case in {"hw_contention", "hw_direct_blocking"}:
            checks["same_core_peer_tiles_during_read_gate"] = check_contention(case, trace, data)
        positive[case] = checks | {"shared_resource_budgets": "passed", "path_transfers": transfers}
        print(case + ": passed", flush=True)
    negative = {}
    for case, diagnostic in {
        "hw_rebind_during_import": "request slot still busy",
        "hw_reclaim_during_import": "reclaim before transfer consumed",
    }.items():
        process = subprocess.run(
            [str(binary), case, str(build / case)], check=False, capture_output=True, text=True, timeout=30
        )
        assert process.returncode != 0 and diagnostic in process.stderr, (case, process.stderr)
        negative[case] = {"rejected_as_expected": True, "diagnostic": process.stderr.strip()}
    names = (
        "static_plan_probe.cpp",
        "kv_quota.h",
        "hardware_transport.h",
        "run_hardware_checks.py",
        "prepare_deployment.py",
        "deployment.min.json",
        "program.manifest.json",
    )
    result = dict(
        positive=positive,
        negative=negative,
        source_sha256={n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in names},
        hardware_execution=False,
        timing="declared logical service quanta; no hardware timing calibration",
    )
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Passed {len(positive)} hardware-path scenarios and {len(negative)} rejection scenarios")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    run(
        args.source_root.resolve(),
        args.compiler.resolve(),
        args.build.resolve(),
        (args.summary or args.build / "hardware_validation.json").resolve(),
    )
