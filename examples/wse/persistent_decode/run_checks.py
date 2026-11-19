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
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
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
    for region in result["memory"]:
        assert 0 <= region["used"] <= region["peak"] <= region["capacity"]
        if region["tier"] == 1:
            assert region["used"] == 0, "KV pages leaked after request drain"
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
    return {"ordered_transfers": len(transfers), "events": len(events), "capacity_and_drain": "passed"}


def run_all(binary, output, wsl=False):
    output.mkdir(parents=True, exist_ok=True)

    def invoke(arguments, directory):
        if wsl:

            def linux_path(path):
                text = str(path.resolve()).replace("\\", "/")
                return "/mnt/" + text[0].lower() + text[2:]

            command = ["wsl", "-d", "Ubuntu-26.04", "--", linux_path(binary)]
            destination = linux_path(directory)
        else:
            command, destination = [str(binary.resolve())], str(directory.resolve())
        return subprocess.run(
            command + arguments + ["--output", destination], capture_output=True, text=True, timeout=120, check=False
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
    for name in ("reverse", "batcher_program", "one_credit"):
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
    }
    for name, (arguments, message) in failures.items():
        process = invoke(arguments, output / name)
        assert process.returncode != 0 and message in process.stderr, (name, process.stderr)
        summary[name] = {"rejected_as_expected": True, "diagnostic": process.stderr.strip()}
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Passed {len(scenarios)} numeric scenarios and {len(failures)} rejection scenarios")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wsl", action="store_true", help="Run a Linux executable from Windows using Ubuntu-26.04")
    arguments = parser.parse_args()
    run_all(arguments.binary, arguments.output, arguments.wsl)
