# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify component boundaries, mathematical outputs and lifetime rejection cases."""

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from model import ModelAdapter, oracle
from simulate import World, default_requests, read_requests
from transport import Violation


def select(events, name, request=None):
    return [e for e in events if e["event"] == name and (request is None or e.get("request") == request)]


def validate(world, result):
    events = world.events
    assert result["launches"] == 1 and result["allocations_before_launch"] == 246
    assert result["session_unloaded"] and result["all_cache_leases_returned"]
    assert len(select(events, "SUBMIT_REQUEST")) == len(world.specs) == 3
    assert len(select(events, "REQUEST_ACCEPTED")) == len(select(events, "REQUEST_DRAINED")) == 3
    assert not any(e["event"] in {"STEP_COMMAND", "HOST_LAYER_DISPATCH"} for e in events)
    tokens = select(events, "TOKEN_LOCAL_FEEDBACK")
    assert len(select(events, "LAYER_COMMITTED")) == 40 * len(tokens)
    for name, entry in result["requests"].items():
        generated = select(events, "GENERATION_FINISHED_OBSERVED", name)[0]
        drained = select(events, "REQUEST_DRAINED", name)[0]
        assert generated["index"] < drained["index"]
        terminal = select(events, "CLIENT_GENERATION_FINISHED", name)
        assert len(terminal) == 1 and generated["index"] < terminal[0]["index"]
        assert all(
            e["index"] < terminal[0]["index"]
            for e in select(events, "TOKEN_CHUNK_OWNED", name) + select(events, "CLIENT_TOKEN_SENT", name)
        )
        for token in select(events, "TOKEN_LOCAL_FEEDBACK", name):
            assert token["sequence"] <= world.host.records[name].spec["max_new_tokens"]
        spec = world.host.records[name].spec
        actual_layers = [(e["sequence"], e["layer"], e["hidden"]) for e in select(events, "LAYER_COMMITTED", name)]
        assert actual_layers == oracle(spec["seed"], entry["emitted"], return_layers=True)
        if entry["emitted"]:
            imports = select(events, "PREFILL_PART_VISIBLE", name)
            active = select(events, "REQUEST_ACTIVE", name)[0]
            assert max(e["index"] for e in imports) < active["index"]
            assert all(e["index"] > active["index"] for e in select(events, "TOKEN_START", name))
    assert select(events, "REQUEST_DRAINED", "A")[0]["index"] < select(events, "REQUEST_ACCEPTED", "C")[0]["index"]
    assert world.host.records["A"].generation == 1 and world.host.records["C"].generation == 2
    assert select(events, "SESSION_DRAINED")[0]["index"] < select(events, "PERSISTENT_KERNEL_EXIT")[0]["index"]
    assert select(events, "PERSISTENT_KERNEL_EXIT")[0]["index"] < select(events, "SESSION_UNLOADED")[0]["index"]

    transfers = validate_transport(events)
    validate_scenario(world, result, events)
    return dict(
        passed=True,
        tokens=len(tokens),
        layers=40 * len(tokens),
        transfers=transfers,
        oracle="passed",
        budgets="passed",
        lifecycle="passed",
        request_protocol="passed",
    )


def validate_transport(events):
    streams = defaultdict(list)
    actions, services = Counter(), Counter()
    contexts, data = Counter(), Counter()
    for event in events:
        if "transfer" not in event:
            continue
        streams[event["transfer"]].append(event)
        if event["path"] == "Batcher" and event["event"] in {
            "TRANSFER_ISSUED",
            "DATA_VISIBLE",
        }:
            ep = event["endpoint"]
            actions[event["tick"], ep] += 1
            delta = 1 if event["event"] == "TRANSFER_ISSUED" else -1
            control = event["kind"] in {
                "activation",
                "grant",
                "low",
                "grant_ack",
                "finished",
                "drained",
                "cancel",
                "output_credit",
                "session_stop",
            }
            contexts[ep] += delta
            data[ep] += delta * (not control)
            assert 0 <= data[ep] <= 3 and 0 <= contexts[ep] <= 4
        if event["event"] == "SHARED_SERVICE":
            services[event["tick"], event["endpoint"] // 4, event["domain"]] += event["bytes"]
    assert max(actions.values()) == 1 and max(services.values()) <= 64
    assert not any(contexts.values()) and not any(data.values())
    for stream in streams.values():
        names = [
            e["event"]
            for e in stream
            if e["event"]
            in {
                "TRANSFER_ACCEPTED",
                "TRANSFER_ISSUED",
                "SOURCE_READ_DONE",
                "DATA_VISIBLE",
            }
        ]
        assert names == [
            "TRANSFER_ACCEPTED",
            "TRANSFER_ISSUED",
            "SOURCE_READ_DONE",
            "DATA_VISIBLE",
        ]
        first = stream[0]
        if first["kind"] == "prefill":
            assert first["path"] == "DirectImport"
        if first["kind"] in {
            "grant",
            "activation",
            "low",
            "grant_ack",
            "finished",
            "drained",
            "output",
            "output_credit",
            "cancel",
        }:
            assert first["path"] == "Batcher"
        if first["path"] == "CorePush":
            issued = next(e for e in stream if e["event"] == "TRANSFER_ISSUED")
            visible = next(e for e in stream if e["event"] == "DATA_VISIBLE")
            assert visible["tick"] - issued["tick"] == 2
    return len(streams)


def validate_scenario(world, result, events):
    if world.scenario in {"normal", "slow_output", "slow_writeback"}:
        assert result["requests"]["A"]["tokens"] == [61, 25]
        assert result["requests"]["A"]["reason"] == "eos"
        assert result["requests"]["B"]["emitted"] == 4
        second = select(events, "TOKEN_LOCAL_FEEDBACK", "B")[1]
        first_write = next(
            e
            for e in events
            if e["event"] == "SOURCE_READ_DONE"
            and e.get("request") == "B"
            and e.get("kind") == "writeback"
            and e.get("sequence") == 1
        )
        assert second["index"] < first_write["index"]
        grants = select(events, "GROUP_GRANT_INSTALLED", "B")
        assert any(e["counts"] == {"window": 3, "c2a": 2, "c1a": 3} for e in grants)
    if world.scenario == "slow_output":
        assert select(events, "OUTPUT_RING_FULL", "B")[0]["sequence"] == 3
    if world.scenario == "slow_writeback":
        assert select(events, "DIRTY_RING_FULL", "B")[0]["sequence"] == 3
    if world.scenario == "cancel_import":
        assert not select(events, "ACTIVATION_INSTALLED", "A")
        assert result["requests"]["A"]["emitted"] == 0
        assert (
            max(e["index"] for e in select(events, "PREFILL_PART_VISIBLE", "A"))
            < select(events, "REQUEST_DRAINED", "A")[0]["index"]
        )
    if world.scenario == "cancel_running":
        assert result["requests"]["B"]["reason"] == "cancelled"
        assert select(events, "CANCEL_INSTALLED", "B")
        assert select(events, "CANCELLED_OUTPUT_DISCARDED", "B")
    if world.scenario == "slow_client":
        assert (
            select(events, "OUTPUT_CREDIT_INSTALLED", "B")[0]["index"]
            < select(events, "CLIENT_TOKEN_SENT", "B")[0]["index"]
        )
    if world.scenario == "deny_refill":
        assert all(e["reason"] == "capacity_denied" for e in result["requests"].values())


def rejections():
    results = {}

    def reject(name, action, text):
        try:
            action()
        except Violation as error:
            assert text in str(error), (name, str(error))
            results[name] = dict(rejected=True, diagnostic=str(error))
        else:
            raise AssertionError(name + " did not reject")

    spec = default_requests()[0]
    reject(
        "unsupported_sampling",
        lambda: ModelAdapter.lower(dict(spec, sampling="top_p")),
        "sampling capability",
    )
    reject(
        "generation_over_budget",
        lambda: ModelAdapter.lower(dict(spec, max_new_tokens=7)),
        "generation budget",
    )
    reject(
        "incomplete_groups",
        lambda: ModelAdapter.lower(dict(spec, cache_groups=["window"])),
        "grouped cache",
    )
    world = World()
    world.host.input_closed = True
    reject("submit_after_close", lambda: world.host.submit(spec), "CloseInput")
    world = World()
    world.memory.sealed = True
    reject(
        "runtime_allocation",
        lambda: world.memory.allocate("SRAM", 0, "late"),
        "allocation after launch",
    )
    world = World()
    world.host.submit(spec)
    reject("duplicate_request", lambda: world.host.submit(spec), "duplicate request")
    world.host.step()
    binding = world.host.records["A"]
    reject(
        "return_lease_with_import_inflight",
        lambda: world.pool.release(binding.key),
        "before DMA drain",
    )
    reject(
        "overwrite_import_source",
        lambda: world.memory.write(binding.layout["snapshot_host"], 0, "Host"),
        "DMA or delivery",
    )
    world = World()
    world.host.submit(spec)
    slot, binding = world.device.slots[0], world.host.records["A"]
    slot.key = binding.key
    world.memory.write(slot.layout["credit_core"], dict(key="A:0", sequence=1, index=0), slot.home)
    reject("old_generation_output_credit", slot.credit, "stale output credit generation")
    initial = {g: tuple(p) for g, p in binding.tables.items()}
    payload = dict(
        key=binding.key,
        generation=1,
        sequence=1,
        tables=initial,
        authorized_context=128,
    )
    world.memory.write(slot.layout["grant_core"], payload, slot.home)
    slot.install_grant()
    larger = world.pool.reserve(binding.key, 0, 300)
    assert larger is not None
    bad = copy.deepcopy(larger)
    bad["window"] = (bad["window"][1], bad["window"][0], *bad["window"][2:])
    world.memory.write(
        slot.layout["grant_core"],
        dict(payload, sequence=2, tables=bad, authorized_context=300),
        slot.home,
    )
    reject(
        "replace_live_page_prefix",
        slot.install_grant,
        "replaces committed group prefix",
    )
    slot.phase = "compute"
    reject("exit_with_active_request", world.device.stop, "exit before request drain")
    before = copy.deepcopy(world.pool.tables)
    assert world.pool.reserve(binding.key, 0, 4096) is None and world.pool.tables == before
    results["atomic_reservation_exhaustion"] = dict(
        rejected=True,
        diagnostic="reserve returned None without partial ownership changes",
    )
    return results


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    assert read_requests(Path(__file__).with_name("requests.json")) == default_requests()
    positive = {}
    for scenario in (
        "normal",
        "slow_output",
        "slow_writeback",
        "slow_client",
        "cancel_import",
        "cancel_running",
        "deny_refill",
    ):
        world = World(scenario)
        result = world.run()
        positive[scenario] = validate(world, result)
        world.write(output / scenario, result)
        print(scenario + ": passed", flush=True)
    negative = rejections()
    source = Path(__file__).resolve().parent
    summary = dict(
        format="wse.request_service.functional.v1",
        positive=positive,
        negative=negative,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")},
        input_sha256=hashlib.sha256((source / "requests.json").read_bytes()).hexdigest(),
        real_serving_integration=False,
        full_deepseek_model=False,
        native_hardware=False,
    )
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Passed {len(positive)} request-service scenarios and {len(negative)} rejection checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
