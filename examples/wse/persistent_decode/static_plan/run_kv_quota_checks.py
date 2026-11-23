# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise quota control messages and fixed-pool decode using explicit gates."""

import argparse
import copy
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from prepare_deployment import prepare
from run_probe import check_run


def verify_delayed(case, trace, data):
    if case in {"kv_delayed_host", "kv_delayed_install"}:
        gate = "KV_HOST_GATE" if case == "kv_delayed_host" else "KV_INSTALL_GATE"
        closed = next(e for e in trace if e["event"] == gate + "_CLOSED")
        released = next(e for e in trace if e["event"] == gate + "_RELEASED")
        pause = next(e for e in trace if e["event"] == "KV_HARD_PAUSE" and e["slot"] == 0)
        assert closed["tick"] <= pause["tick"] <= released["tick"]
        assert any(
            e["event"] == "LAYER_DONE"
            and e["slot"] == 2
            and e["actor"] == 0
            and pause["tick"] <= e["tick"] <= released["tick"]
            for e in data
        )
        assert any(e["event"] == "KV_RESUMED" and e["slot"] == 0 for e in trace)
        if case == "kv_delayed_install":
            first = trace.index(closed)
            last = trace.index(released)
            held = [e for e in trace[first : last + 1] if e["slot"] == 0]
            assert any(e["host_owned"] > e["granted"] for e in held)
            assert all(e["granted"] == 2 for e in held), "pending Host grant became usable before installation"


def verify(directory, case, config):
    result = json.loads((directory / "results.json").read_text())
    quota = json.loads((directory / "kv_quota.json").read_text())
    trace = [json.loads(row) for row in (directory / "kv_events.jsonl").read_text().splitlines()]
    data = [json.loads(row) for row in (directory / "events.jsonl").read_text().splitlines()]
    control = [json.loads(row) for row in (directory / "control_events.jsonl").read_text().splitlines()]
    expected = (
        24
        if case == "kv_watermark_edge"
        else 14
        if case == "kv_stale_generation"
        else 4
        if case == "kv_pool_deny"
        else 12
    )
    checks = check_run(directory, case, expected)
    assert quota["free_pages"] == [config["pool_pages_per_chip"]] * 2
    assert quota["low_sent"] > 0 and quota["claims"] == expected
    assert all(event["claimed"] <= event["granted"] <= 16 for event in trace)
    transfers = defaultdict(list)
    for event in control:
        if event["transfer_id"]:
            transfers[event["transfer_id"]].append(event["event"])
    assert transfers and all(
        events == ["SEND", "COPY_BEGIN", "DELIVER", "PUBLISH", "CONSUMED"] for events in transfers.values()
    )
    by_request = defaultdict(list)
    for event in trace:
        by_request[event["slot"], event["generation"]].append(event)
    for identity, events in by_request.items():
        installed = [e for e in events if e["event"] == "KV_GRANT_INSTALLED"]
        assert len(installed) == len({e["sequence"] for e in installed}), "grant installed twice"
        low = [e for e in events if e["event"] == "KV_LOW_SENT"]
        assert len(low) == len({e["sequence"] for e in low}), "low-water notification storm"
        assert any(e["event"] == "KV_PAGES_RETURNED" for e in events)
        data_trace = [e for e in data if (e["slot"], e["generation"]) == identity]
        assert any(e["event"] == "REQUEST_DRAINED" for e in data_trace)
    if case == "kv_watermark_edge":
        assert quota["low_sent"] == 8
        assert all(e["granted"] - e["claimed"] == 1 for e in trace if e["event"] == "KV_LOW_SENT")
    verify_delayed(case, trace, data)
    if case == "kv_hard_pause":
        assert quota["hard_pauses"] > 0
        assert all(e["claimed"] == e["granted"] for e in trace if e["event"] == "KV_HARD_PAUSE")
    if case == "kv_duplicate_event":
        assert quota["duplicate_events_dropped"] > 0
    if case == "kv_duplicate_grant":
        assert quota["duplicate_grants_dropped"] > 0
    if case == "kv_old_grant":
        assert quota["stale_grants_dropped"] > 0 and any(e["event"] == "KV_OLD_GRANT_DROPPED" for e in trace)
    if case in {"kv_stale_generation", "kv_stale_epoch"}:
        assert quota["stale_events_dropped"] > 0 and quota["stale_grants_dropped"] > 0
        for e in trace:
            if e["event"] == "KV_STALE_GRANT_DROPPED":
                assert (
                    e["packet_generation"] != e["generation"]
                    if case == "kv_stale_generation"
                    else e["packet_epoch"] == 0
                )
    if case == "kv_pool_wait":
        assert quota["pool_waits"] > 0 and quota["denials"] == 0
    if case == "kv_pool_deny":
        assert quota["denials"] == 4
        assert sum(e["event"] == "REQUEST_QUOTA_STOP" for e in data) == 4
        assert len(quota["outcomes"]) == 4 and all(o["reason"] == "capacity_denied" for o in quota["outcomes"])
    if case == "kv_control_pressure":
        assert result["queue_stalls"] > 0 and quota["control_queue_stalls"] > 0
    return checks | quota | {"control_ordered_transfers": len(transfers), "page_ownership_and_bounds": "passed"}


def run(source_root, compiler, build, summary):
    here = Path(__file__).resolve().parent
    base = json.loads((here / "deployment.min.json").read_text())
    base["program"]["manifest"] = str(here / base["program"]["manifest"])
    cases = [
        "kv_refill",
        "kv_watermark_edge",
        "kv_delayed_host",
        "kv_delayed_install",
        "kv_hard_pause",
        "kv_duplicate_event",
        "kv_duplicate_grant",
        "kv_old_grant",
        "kv_stale_generation",
        "kv_stale_epoch",
        "kv_pool_wait",
        "kv_pool_deny",
        "kv_control_pressure",
    ]
    outputs, cache = {}, {}
    for case in cases:
        directory = build / case
        directory.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(base)
        q = config["kv_quota"]
        if case == "kv_watermark_edge":
            q.update(initial_pages_per_request=3, refill_pages=3)
        if case == "kv_hard_pause":
            q.update(initial_pages_per_request=1, low_watermark_pages=0)
        if case in {"kv_pool_wait", "kv_pool_deny"}:
            q.update(
                pool_pages_per_chip=6 if case == "kv_pool_wait" else 4,
                refill_pages=2,
                exhaustion_policy="wait" if case == "kv_pool_wait" else "deny",
            )
        if case == "kv_control_pressure":
            config["transport"] = dict(inbox_entries_per_core=1, queue_entries_per_endpoint=2)
            q["control_queue_entries_per_endpoint"] = 1
        path = directory / "deployment.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        plan, artifact = prepare(path, directory)
        digest = plan["deployment_digest"]
        if digest not in cache:
            binary = directory / "probe.exe"
            command = [
                str(compiler),
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
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
            cache[digest] = binary
        process = subprocess.run(
            [str(cache[digest]), case, str(directory / "run")], check=False, capture_output=True, text=True, timeout=90
        )
        assert process.returncode == 0, (case, process.stderr)
        outputs[case] = verify(directory / "run", case, q)
        print(case + ": passed", flush=True)
    config = copy.deepcopy(base)
    config["kv_quota"].update(pool_pages_per_chip=4, refill_pages=2, exhaustion_policy="wait")
    directory = build / "kv_wait_no_progress"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "deployment.json"
    path.write_text(json.dumps(config, indent=2))
    plan, artifact = prepare(path, directory)
    binary = directory / "probe.exe"
    process = subprocess.run(
        [
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
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    process = subprocess.run(
        [str(binary), "kv_wait_no_progress", str(directory / "run")],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode != 0 and "no runnable page owner" in process.stderr
    negatives = {"kv_wait_no_progress": {"rejected_as_expected": True, "diagnostic": process.stderr.strip()}}
    for case, message in {
        "kv_invalid_grant_bound": "grant exceeds compiled page table",
        "kv_conflicting_grant": "conflicting duplicate grant",
    }.items():
        process = subprocess.run(
            [str(next(iter(cache.values()))), case, str(build / case)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert process.returncode != 0 and message in process.stderr, (case, process.stderr)
        negatives[case] = {"rejected_as_expected": True, "diagnostic": process.stderr.strip()}
    hashes = {
        name: hashlib.sha256((here / name).read_bytes()).hexdigest()
        for name in (
            "static_plan_probe.cpp",
            "kv_quota.h",
            "prepare_deployment.py",
            "run_kv_quota_checks.py",
            "deployment.min.json",
            "program.manifest.json",
        )
    }
    summary.write_text(
        json.dumps(
            dict(
                cases=outputs,
                negative=negatives,
                source_sha256=hashes,
                hardware_execution=False,
                scope="Fixed-pool quota and control-plane protocol over actual Memory/Fabric",
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Passed {len(outputs)} KV quota scenarios and {len(negatives)} rejection scenarios")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--compiler", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--summary", type=Path)
    a = p.parse_args()
    run(
        a.source_root.resolve(),
        a.compiler.resolve(),
        a.build.resolve(),
        (a.summary or a.build / "kv_quota_validation.json").resolve(),
    )
