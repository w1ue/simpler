# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run numeric, programmable-entry and lifecycle checks against the C++ backend."""

import argparse
import json
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from prefill import envelope, generate
from reference import verify


def trace_checks(directory):
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    transfers = defaultdict(list)
    for event in events:
        if event["transfer_id"]:
            transfers[event["transfer_id"]].append(event["event"])
    expected = ["SEND", "COPY_BEGIN", "DELIVER", "PUBLISH", "CONSUMED"]
    assert transfers and all(order == expected for order in transfers.values()), "transfer ordering violation"
    result = json.loads((directory / "results.json").read_text())
    assert not any(event["event"] == "PREFILL_INPUT" for event in events), "WSE executed prompt computation"
    for request in result["requests"]:
        assert len(request["layer_checksums"]) == 40 * len(request["generated"]), "WSE computed prefill layers"
        trace = [
            event
            for event in events
            if event["slot"] == request["slot"] and event["generation"] == request["generation"]
        ]
        ready = next(i for i, event in enumerate(trace) if event["event"] == "PREFILL_IMPORTED")
        compute = [i for i, event in enumerate(trace) if event["event"] == "DECODE_INPUT"]
        assert compute and min(compute) > ready
        assert len(compute) == len(request["generated"])
        assert all(
            event["position"] >= len(request["prompt"])
            for event in trace
            if event["event"] in {"DECODE_INPUT", "LAYER_DONE", "EXPERT_COMPUTE", "LOOKUP_COMPUTE"}
        )
        assert next(event["position"] for event in trace if event["event"] == "DECODE_INPUT") == len(request["prompt"])
    expected_imports = sum(3 * (len(request["prompt"]) // 2) + len(request["prompt"]) for request in result["requests"])
    assert result["wse_prefill_compute_positions"] == 0
    assert result["prefill_kv_rows_imported"] == expected_imports
    expected_rows = sum(sum(len(rows) for rows in request["cache_rows"]) for request in result["requests"])
    assert result["kv_rows_committed"] + expected_imports == expected_rows
    assert result["prefill_import_bytes"] > 0
    assert result["bytes_transferred"] == result["prefill_import_bytes"] + result["decode_bytes_transferred"]
    for region in result["memory"]:
        assert 0 <= region["used"] <= region["peak"] <= region["capacity"]
        if region["tier"] in (1, 3):
            assert region["used"] == 0, "KV or Host import buffers leaked after request drain"
    assert len([event for event in events if event["event"] == "SERVICE_LAUNCH"]) == 1
    assert events[-1]["event"] == "SERVICE_DRAINED"
    if len(result["requests"]) == 4:
        a_drained = next(
            i
            for i, event in enumerate(events)
            if event["event"] == "REQUEST_DRAINED" and event["slot"] == 0 and event["generation"] == 1
        )
        c2_admit = next(
            i
            for i, event in enumerate(events)
            if event["event"] == "ADMIT" and event["slot"] == 0 and event["generation"] == 2
        )
        assert a_drained < c2_admit
        a_outputs = [
            i
            for i, event in enumerate(events)
            if event["event"] == "OUTPUT_TOKEN" and event["slot"] == 0 and event["generation"] == 1
        ]
        c1_admit = next(i for i, event in enumerate(events) if event["event"] == "ADMIT" and event["slot"] == 2)
        assert a_outputs[1] < c1_admit < a_drained
        assert result["slot_reuses"] == 1
    return {
        "ordered_transfers": len(transfers),
        "events": len(events),
        "capacity_and_drain": "passed",
        "decode_only_trace": "passed",
        "prefill_import_order": "passed",
    }


def snapshot_rejections(invoke, output, snapshots):
    raw = (snapshots / "A.wsepf").read_bytes()
    payload = bytearray(raw[24:])
    length = struct.unpack_from("<I", payload, 48)[0]
    first_rows = 56 + length * 4
    first_window = first_rows + sum(4 + count * 12 for count in [length // 2] * 3 + [length])
    bad_model, bad_rows, bad_window, bad_token, nonfinite = (bytearray(payload) for _ in range(5))
    struct.pack_into("<I", bad_model, 4, 999)
    struct.pack_into("<I", bad_rows, first_rows, 999)
    struct.pack_into("<I", bad_window, first_window, 999)
    struct.pack_into("<I", bad_token, 52, 999)
    struct.pack_into("<d", nonfinite, first_window + 4, float("nan"))
    cases = {
        "snapshot_truncated": (raw[:-1], "truncated payload"),
        "snapshot_checksum": (raw[:-1] + bytes([raw[-1] ^ 1]), "checksum mismatch"),
        "snapshot_model": (envelope(bad_model), "model contract mismatch"),
        "snapshot_rows": (envelope(bad_rows), "row count mismatch"),
        "snapshot_window": (envelope(bad_window), "window row count mismatch"),
        "snapshot_token": (envelope(bad_token), "next token outside vocabulary"),
        "snapshot_state": (envelope(payload[:-8]), "truncated dense state"),
        "snapshot_nonfinite": (envelope(nonfinite), "nonfinite dense state"),
        "snapshot_version": (raw[:8] + struct.pack("<I", 99) + raw[12:], "unsupported version"),
    }
    summary = {}
    for name, (content, diagnostic) in cases.items():
        directory = output / name
        directory.mkdir(exist_ok=True)
        (directory / "A.wsepf").write_bytes(content)
        process = invoke(["--single"], directory / "run", directory)
        assert process.returncode != 0 and diagnostic in process.stderr, (name, process.stderr)
        summary[name] = {"rejected_as_expected": True, "diagnostic": process.stderr.strip()}
    return summary


def run_all(binary, output, wsl=False):
    output.mkdir(parents=True, exist_ok=True)
    snapshot_directory = output / "external_prefill"
    generate(snapshot_directory)

    def invoke(arguments, directory, snapshots_path=snapshot_directory):
        if wsl:

            def linux_path(path):
                text = str(path.resolve()).replace("\\", "/")
                return "/mnt/" + text[0].lower() + text[2:]

            command = ["wsl", "-d", "Ubuntu-26.04", "--", linux_path(binary)]
            destination = linux_path(directory)
        else:
            command, destination = [str(binary.resolve())], str(directory.resolve())
        snapshots = linux_path(snapshots_path) if wsl else str(snapshots_path.resolve())
        return subprocess.run(
            command + ["--prefill-dir", snapshots] + arguments + ["--output", destination],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    scenarios = {
        "single": ["--single"],
        "multi": ["--multi"],
        "reverse": ["--multi", "--reverse-batcher"],
        "batcher_program": ["--multi", "--custom-batcher"],
        "aicore_program": ["--single", "--custom-core"],
        "cancel": ["--multi", "--cancel"],
        "one_credit": ["--multi", "--inbox-capacity", "1"],
        "max_boundary": ["--single", "--max-context", "11"],
        "repage": ["--multi", "--page-rows", "2"],
    }
    summary, results = {}, {}
    for name, arguments in scenarios.items():
        directory = output / name
        process = invoke(arguments, directory)
        if process.returncode:
            raise RuntimeError(f"{name}: {process.stdout}\n{process.stderr}")
        checks = verify(directory)
        checks.update(trace_checks(directory))
        (directory / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        summary[name] = checks
        results[name] = json.loads((directory / "results.json").read_text())

    baseline = results["multi"]["requests"]
    for name in ("reverse", "batcher_program", "one_credit", "repage"):
        for expected_request, actual_request in zip(baseline, results[name]["requests"]):
            assert expected_request["generated"] == actual_request["generated"]
            assert expected_request["cache_rows"] == actual_request["cache_rows"]
            np.testing.assert_allclose(expected_request["logits"], actual_request["logits"], atol=1e-10, rtol=0)
    assert results["batcher_program"]["custom_batcher_steps"] > 0
    assert not np.allclose(
        results["single"]["requests"][0]["logits"], results["aicore_program"]["requests"][0]["logits"]
    )
    cancelled = next(request for request in results["cancel"]["requests"] if request["name"] == "B")
    assert cancelled["cancelled"] and len(cancelled["generated"]) == 2
    assert results["single"]["bailu_fetches"] > 0

    failures = {
        "publish_before_delivery": (["--invalid-batcher"], "publication before delivery"),
        "service_send_before_compute": (["--invalid-service"], "service send outside computed result state"),
        "maximum_context_overrun": (["--single", "--max-context", "10"], "maximum context exceeded"),
        "decode_before_import": (["--invalid-core"], "decode before prefill import completed"),
    }
    for name, (arguments, message) in failures.items():
        process = invoke(arguments, output / name)
        assert process.returncode != 0 and message in process.stderr, (name, process.stderr)
        summary[name] = {"rejected_as_expected": True, "diagnostic": process.stderr.strip()}
    summary.update(snapshot_rejections(invoke, output, snapshot_directory))
    for name, tokens in (("even_page_boundary", [1, 4, 7, 3, 2, 8, 5, 6]), ("short_prefix", [3])):
        snapshots = output / (name + "_prefill")
        generate(snapshots, {"A": tokens})
        directory = output / name
        process = invoke(["--single"], directory, snapshots)
        assert process.returncode == 0, process.stderr
        summary[name] = verify(directory) | trace_checks(directory)
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    positive = sum(check.get("independent_numpy_oracle") == "passed" for check in summary.values())
    negative = sum(check.get("rejected_as_expected", False) for check in summary.values())
    print(f"Passed {positive} numeric scenarios and {negative} rejection scenarios")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wsl", action="store_true", help="Run a Linux executable from Windows using Ubuntu-26.04")
    arguments = parser.parse_args()
    run_all(arguments.binary, arguments.output, arguments.wsl)
