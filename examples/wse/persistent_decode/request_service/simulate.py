# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run an isolated request-level persistent decode protocol, not the product stack."""

import argparse
import json
from collections import Counter
from pathlib import Path

from device import DeviceProgram
from model import GROUPS, CachePool, Layout, Serving, SimplerHost, oracle
from transport import Memory, Transport, Violation, require


def default_requests():
    return [
        dict(
            request_id=name,
            prompt_context=prompt,
            seed=seed,
            max_new_tokens=budget,
            eos_token_id=eos,
            sampling="scalar_mod97",
            policy_epoch=1,
            cache_groups=["window", "c2a", "c1a"],
        )
        for name, prompt, seed, budget, eos in (
            ("A", 127, 11, 4, 25),
            ("B", 255, 18, 4, None),
            ("C", 127, 97, 2, None),
        )
    ]


def read_requests(path):
    data = json.loads(path.read_text())
    require(set(data) == {"format", "program", "requests"}, "unknown service input fields")
    require(
        data["format"] == "wse.request_service.functional.v1",
        "unsupported request service format",
    )
    require(
        data["program"]
        == dict(
            id="scalar40.twochips.twoslots",
            request_slots=2,
            max_context=512,
            cache_group_source_tokens=GROUPS,
        ),
        "unsupported compiled profile",
    )
    require(
        isinstance(data["requests"], list) and 0 < len(data["requests"]) <= 16,
        "bounded input trace required",
    )
    return data["requests"]


class World:
    def __init__(self, scenario="normal", specs=None):
        self.scenario = scenario
        self.tick = 0
        self.events = []
        self.memory = Memory()
        self.transport = Transport(self.memory, self.emit, self.gate)
        self.pool = CachePool(self.memory, self.emit)
        self.layout = Layout(self.memory, self.pool)
        self.simpler = self.host = SimplerHost(self)
        self.serving = Serving(self, specs or default_requests())
        self.device = DeviceProgram(self)
        self.cancel_issued = False
        self.unloaded = False
        self.launches = 0
        self.specs = specs or default_requests()

    def emit(self, event, component, **data):
        self.events.append(
            dict(
                index=len(self.events),
                tick=self.tick,
                event=event,
                component=component,
                **data,
            )
        )

    def gate(self, job):
        if self.scenario == "cancel_import" and job.kind == "prefill" and job.identity.get("request") == "A":
            return not any(e["event"] == "LAYER_COMMITTED" and e.get("request") == "B" for e in self.events)
        if job.kind == "writeback" and job.identity.get("request") == "B":
            slot = self.device.slots[1]
            if self.scenario == "slow_writeback":
                return not slot.dirty_blocked
            return not any(
                e.get("request") == "B"
                and (
                    e["event"] == "GENERATION_FINISHED"
                    or (e["event"] == "TOKEN_LOCAL_FEEDBACK" and e.get("sequence") == 2)
                )
                for e in self.events
            )
        return False

    def request_cancel(self):
        if self.cancel_issued:
            return
        request = "A" if self.scenario == "cancel_import" else "B"
        trigger = "PREFILL_PART_SUBMITTED" if self.scenario == "cancel_import" else "TOKEN_LOCAL_FEEDBACK"
        if self.scenario not in {"cancel_import", "cancel_running"}:
            return
        if any(e["event"] == trigger and e.get("request") == request for e in self.events):
            self.emit("CLIENT_CANCEL", "Serving.RequestScheduler", request=request)
            self.serving.cancelled.add(request)
            self.host.cancel(request)
            self.cancel_issued = True

    def run(self, limit=50000):
        self.emit(
            "PROGRAM_PROFILE_BOUND",
            "ModelAdapter",
            program="scalar40.twochips.twoslots",
            implementation="handwritten compiled-control-flow surrogate",
        )
        self.memory.sealed = True
        self.emit(
            "FIXED_LAYOUT_SEALED",
            "SimplerHost.Loader",
            allocations=len(self.memory.cells),
        )
        self.emit(
            "ENDPOINTS_CONFIGURED",
            "SimplerHost.Transport",
            batcher_endpoints=8,
            contexts_per_endpoint=4,
            data_context_limit=3,
        )
        self.launches += 1
        self.emit(
            "PERSISTENT_KERNEL_LAUNCH",
            "SimplerHost.Session",
            allocations=len(self.memory.cells),
        )
        self.emit("SESSION_READY", "CoreProgram")
        for tick in range(limit):
            self.tick = tick
            self.transport.advance(tick)
            self.request_cancel()
            self.serving.step()
            if not self.device.exited:
                self.device.step()
            self.host.step()
            if self.device.exited and self.transport.idle() and not self.unloaded:
                require(
                    not self.pool.tables and all(v is None for v in self.pool.owner.values()),
                    "unload with outstanding cache lease",
                )
                require(
                    all(not self.memory.busy(i) for i in range(len(self.memory.cells))),
                    "unload with DMA or delivery ownership",
                )
                self.memory.released = True
                self.unloaded = True
                self.emit("SESSION_UNLOADED", "SimplerHost.Lifecycle")
            if self.unloaded and not self.serving.network and len(self.serving.terminal_sent) == len(self.specs):
                return self.report()
        pending = [(s.key, s.phase, s.sequence, s.layer, len(s.pending_control)) for s in self.device.slots]
        live = [(j.kind, j.stage, j.identity.get("key")) for j in self.transport.jobs if j is not None]
        raise Violation(f"no drain within logical bound: {pending}; {live}")

    def report(self):
        requests = {}
        for spec in self.specs:
            name = spec["request_id"]
            output = self.serving.received.get(name, [])
            expected = oracle(spec["seed"], len(output))
            require(
                [(p["token"], p["hidden"]) for p in output] == expected,
                "independent integer oracle mismatch",
            )
            finish = self.serving.finished[name]
            require(
                finish["emitted"] == len(output),
                "GenerationFinished/output ownership mismatch",
            )
            require(name in self.serving.released, "request not physically drained")
            requests[name] = dict(
                tokens=[p["token"] for p in output],
                hidden=[p["hidden"] for p in output],
                **finish,
            )
        occupancy = Counter()
        for c in self.memory.cells:
            occupancy[c.tier + ":" + str(c.owner)] += c.size
        return dict(
            scenario=self.scenario,
            launches=self.launches,
            requests=requests,
            allocations_before_launch=len(self.memory.cells),
            physical_allocations_during_service=0,
            per_token_serving_commands=0,
            session_unloaded=self.unloaded,
            all_cache_leases_returned=not self.pool.tables,
            events=len(self.events),
            last_logical_tick=self.tick,
            fixed_bytes_by_owner=dict(occupancy),
            limits=[
                "isolated proposed protocol; not real serving/simpler integration",
                "integer scalar decode; no full DeepSeek weights/attention/sampling",
                "logical service model; no native WSE ISA, ABI or timing calibration",
                "CPU metadata shadows; allowances do not prove native control layout",
            ],
        )

    def write(self, directory, result):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        with (directory / "events.jsonl").open("w") as stream:
            for event in self.events:
                stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        (directory / "memory_layout.json").write_text(
            json.dumps(
                [
                    dict(handle=i, tier=c.tier, owner=c.owner, name=c.name, bytes=c.size)
                    for i, c in enumerate(self.memory.cells)
                ],
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="normal",
        choices=[
            "normal",
            "slow_output",
            "slow_writeback",
            "slow_client",
            "cancel_import",
            "cancel_running",
            "deny_refill",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("requests.json"))
    args = parser.parse_args()
    world = World(args.scenario, read_requests(args.input))
    result = world.run()
    world.write(args.output, result)
    print(json.dumps(result))
