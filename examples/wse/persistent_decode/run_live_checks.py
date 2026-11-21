# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate live stdin submission using protocol events and an independent numerical oracle."""

import argparse
import json
import queue
import subprocess
import threading
from pathlib import Path

from prefill import generate
from reference import verify


class Session:
    def __init__(self, binary, directory, *arguments):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.stderr = (directory / "stderr.log").open("w")
        self.process = subprocess.Popen(
            [str(binary), "--serve", "--output", str(directory), *arguments],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        self.queue = queue.Queue()
        self.events = []
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.process.stdout.close()
        self.stderr.close()

    def read(self):
        try:
            for line in self.process.stdout:
                self.queue.put(json.loads(line))
        except Exception as error:
            self.queue.put(error)
        finally:
            self.queue.put(None)

    def send(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def until(self, predicate):
        while True:
            event = self.queue.get(timeout=30)
            assert event is not None, "service exited before expected event"
            if isinstance(event, Exception):
                raise event
            self.events.append(event)
            if predicate(event):
                return event

    def finish(self, eof=True):
        if eof:
            self.process.stdin.close()
        code = self.process.wait(timeout=30)
        self.reader.join(timeout=5)
        assert not self.reader.is_alive()
        while not self.queue.empty():
            event = self.queue.get_nowait()
            if isinstance(event, Exception):
                raise event
            if event is not None:
                self.events.append(event)
        self.stderr.close()
        assert code == 0, (self.directory / "stderr.log").read_text()
        (self.directory / "live-output.jsonl").write_text("".join(json.dumps(e) + "\n" for e in self.events))
        return json.loads((self.directory / "results.json").read_text())


def check_budget(result):
    assert result["launches"] == 1 and result["all_drained"] and result["input_closed"]
    for region in result["memory"]:
        assert region["reserved"] == 0
        assert region["peak"] <= region["committed_peak"] <= region["capacity"]


def run(binary, api_binary, output):
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "external_prefill"
    generate(snapshots)
    api = subprocess.run(
        [str(api_binary), str(snapshots), str(output)], capture_output=True, text=True, timeout=60, check=False
    )
    assert api.returncode == 0, api.stdout + api.stderr
    summary = {}
    for name in ("api", "short_contract"):
        summary[name] = verify(output / name)
        result = json.loads((output / name / "results.json").read_text())
        check_budget(result)
    api_events = [json.loads(line) for line in (output / "api/events.jsonl").read_text().splitlines()]
    a_first = next(i for i, e in enumerate(api_events) if e["event"] == "OUTPUT_TOKEN" and e["slot"] == 0)
    b_admit = next(i for i, e in enumerate(api_events) if e["event"] == "ADMIT" and e["slot"] == 1)
    assert a_first < b_admit
    a_drain = next(
        i
        for i, e in enumerate(api_events)
        if e["event"] == "REQUEST_DRAINED" and e["slot"] == 0 and e["generation"] == 1
    )
    c1_admit = next(i for i, e in enumerate(api_events) if e["event"] == "ADMIT" and e["slot"] == 2)
    assert a_drain < c1_admit

    with Session(binary, output / "stdin") as live:
        live.until(lambda e: e["event"] == "SERVICE_LAUNCH")
        live.send(f'SUBMIT A auto 0 128 4 "{snapshots / "A.wsepf"}"')
        live.until(lambda e: e["event"] == "REQUEST_DRAINED")
        live.send(f'SUBMIT C2 auto 0 128 2 "{snapshots / "C2.wsepf"}"')
        live.until(lambda e: e["event"] == "INPUT_ACCEPTED" and e["name"] == "C2")
        result = live.finish()
    check_budget(result)
    assert [(r["slot"], r["generation"]) for r in result["requests"]] == [(0, 1), (0, 2)]
    summary["stdin"] = verify(output / "stdin")
    emitted = [e["token"] for e in live.events if e["event"] == "OUTPUT_TOKEN"]
    assert emitted == [token for request in result["requests"] for token in request["generated"]]

    with Session(binary, output / "reject_then_continue", "--bailu-bytes", "1024") as reject:
        reject.until(lambda e: e["event"] == "SERVICE_LAUNCH")
        reject.send(f'SUBMIT A auto 0 128 4 "{snapshots / "A.wsepf"}"')
        event = reject.until(lambda e: e["event"] == "INPUT_REJECTED")
        assert "cannot fit device capacity" in event["reason"]
        reject.send(f'SUBMIT C1 auto 0 5 2 "{snapshots / "C1.wsepf"}"')
        reject.send("CLOSE")
        result = reject.finish(eof=False)
    check_budget(result)
    assert [r["name"] for r in result["requests"]] == ["C1"]
    summary["reject_then_continue"] = verify(output / "reject_then_continue")

    full_snapshots = output / "max_context_prefill"
    generate(full_snapshots, {"A": [index % 32 for index in range(124)]})
    with Session(binary, output / "max_context", "--bailu-bytes", "10240") as full:
        full.until(lambda e: e["event"] == "SERVICE_LAUNCH")
        full.send(f'SUBMIT A auto 0 128 4 "{full_snapshots / "A.wsepf"}"')
        result = full.finish()
    check_budget(result)
    assert result["requests"][0]["processed_length"] == 128
    assert [len(rows) for rows in result["requests"][0]["cache_rows"]] == [64, 64, 64, 128]
    summary["max_context"] = verify(output / "max_context")
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "Passed live API/stdin, ingress backpressure, capacity delay, rejection, generation reuse and numerical checks"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--api-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.binary.resolve(), args.api_binary.resolve(), args.output.resolve())
