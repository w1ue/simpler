# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host components and a single-owner physical cache pool for request-level service."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from transport import require

GROUPS = {"window": 128, "c2a": 256, "c1a": 128}
EXPERT_OWNERS = [1, 2, 3, 4, 9, 10, 11, 12]
MODULUS = 1000003


def page_counts(context):
    return {group: (context + size - 1) // size for group, size in GROUPS.items()}


def oracle(seed, tokens, return_layers=False):
    """Independent scalar reference for 40-layer integer decode."""
    history, last = 5, seed
    result = []
    layers = []
    for position in range(1, tokens + 1):
        hidden = (last + history) % MODULUS
        for layer in range(40):
            x = (3 * hidden + position + layer) % MODULUS
            experts = [(x + layer + lane) % 8 for lane in range(6)] + [8]
            hidden = (
                x
                + sum(
                    (lane + 1) * ((x * (expert + 2) + 17 * layer + expert) % MODULUS)
                    for lane, expert in enumerate(experts)
                )
            ) % MODULUS
            layers.append((position, layer, hidden))
        last, history = hidden % 97, hidden
        result.append((last, history))
    return layers if return_layers else result


class ModelAdapter:
    @staticmethod
    def lower(spec):
        require(
            set(spec)
            == {
                "request_id",
                "prompt_context",
                "seed",
                "max_new_tokens",
                "eos_token_id",
                "sampling",
                "policy_epoch",
                "cache_groups",
            },
            "unknown request fields",
        )
        require(
            isinstance(spec["request_id"], str) and bool(spec["request_id"]),
            "invalid request ID",
        )
        require(
            all(type(spec[k]) is int for k in ("prompt_context", "seed", "max_new_tokens", "policy_epoch")),
            "integer request field required",
        )
        require(spec["sampling"] == "scalar_mod97", "unsupported sampling capability")
        require(0 < spec["max_new_tokens"] <= 6, "generation budget outside profile")
        require(
            0 < spec["prompt_context"] and spec["prompt_context"] + spec["max_new_tokens"] <= 512,
            "context outside compiled profile",
        )
        require(
            set(spec["cache_groups"]) == set(GROUPS),
            "incomplete grouped cache contract",
        )
        require(spec["policy_epoch"] == 1, "unsupported policy version")
        return dict(spec)


class CachePool:
    """Host simpler is the sole allocator in this proposed request-level variant."""

    def __init__(self, memory, emit):
        self.memory, self.emit = memory, emit
        self.pools = {
            (chip, group): [memory.allocate("Bailu", chip * 8, f"pool.{chip}.{group}.{i}") for i in range(8)]
            for chip in range(2)
            for group in GROUPS
        }
        self.owner = {handle: None for pool in self.pools.values() for handle in pool}
        self.tables = {}

    def reserve(self, key, chip, context):
        previous = self.tables.get(key, {g: [] for g in GROUPS})
        counts = page_counts(context)
        free = {g: [h for h in self.pools[chip, g] if self.owner[h] is None] for g in GROUPS}
        if any(counts[g] - len(previous[g]) > len(free[g]) for g in GROUPS):
            return None
        tables = {g: list(previous[g]) for g in GROUPS}
        for group in GROUPS:
            for handle in free[group][: max(0, counts[group] - len(tables[group]))]:
                self.owner[handle] = key
                tables[group].append(handle)
        self.tables[key] = tables
        self.emit(
            "GROUP_LEASE_RESERVED",
            "SimplerHost.CachePool",
            key=key,
            context=context,
            counts={g: len(p) for g, p in tables.items()},
        )
        return {g: tuple(p) for g, p in tables.items()}

    def validate(self, key, tables):
        require(set(tables) == set(GROUPS), "partial group grant")
        require(
            all(self.owner.get(h) == key for pages in tables.values() for h in pages),
            "cache lease belongs to another request",
        )

    def release(self, key):
        tables = self.tables[key]
        self.validate(key, tables)
        require(
            all(not self.memory.busy(h) for p in tables.values() for h in p),
            "lease release before DMA drain",
        )
        for pages in tables.values():
            for handle in pages:
                self.owner[handle] = None
        del self.tables[key]


class Layout:
    def __init__(self, memory, pool):
        self.slots = []
        for slot, home in enumerate((0, 8)):
            fixed: dict[str, Any] = {"home": home, "slot": slot}
            for name in ("snapshot", "activation", "grant", "credit", "cancel"):
                fixed[name + "_host"] = memory.allocate(
                    "Host",
                    home,
                    f"slot{slot}.{name}.host",
                    512 if name in {"activation", "grant"} else 128,
                )
                fixed[name + "_core"] = memory.allocate(
                    "SRAM",
                    home,
                    f"slot{slot}.{name}.core",
                    512 if name in {"activation", "grant"} else 128,
                )
            for name in ("low", "grant_ack", "finished", "drained"):
                fixed[name + "_core"] = memory.allocate(
                    "SRAM",
                    home,
                    f"slot{slot}.{name}.core",
                    512 if name in {"activation", "grant"} else 128,
                )
                fixed[name + "_host"] = memory.allocate(
                    "Host",
                    home,
                    f"slot{slot}.{name}.host",
                    512 if name in {"activation", "grant"} else 128,
                )
            fixed["fetch"] = memory.allocate("SRAM", home, f"slot{slot}.fetch")
            fixed["frame"] = memory.allocate("SRAM", home, f"slot{slot}.frame", 512)
            fixed["dirty"] = [
                {g: memory.allocate("SRAM", home, f"slot{slot}.dirty{i}.{g}") for g in GROUPS} for i in range(2)
            ]
            fixed["output"] = [memory.allocate("SRAM", home, f"slot{slot}.output{i}") for i in range(2)]
            fixed["output_host"] = [memory.allocate("Host", home, f"slot{slot}.output{i}.host") for i in range(2)]
            fixed["banks"] = []
            for bank in range(2):
                lanes = [
                    (
                        memory.allocate("SRAM", home, f"slot{slot}.bank{bank}.send{lane}"),
                        memory.allocate("SRAM", home, f"slot{slot}.bank{bank}.result{lane}"),
                    )
                    for lane in range(7)
                ]
                experts = []
                for expert in range(9):
                    owner = EXPERT_OWNERS[expert] if expert < 8 else home + 5
                    experts.append(
                        (
                            owner,
                            memory.allocate("SRAM", owner, f"slot{slot}.bank{bank}.e{expert}.input"),
                            memory.allocate("SRAM", owner, f"slot{slot}.bank{bank}.e{expert}.output"),
                        )
                    )
                fixed["banks"].append((lanes, experts))
            self.slots.append(fixed)
        self.stop_host = memory.allocate("Host", 0, "session.stop.host")
        self.stop_core = memory.allocate("SRAM", 0, "session.stop.core")
        for ep in range(8):
            memory.allocate("Batcher", ep, f"batcher{ep}.control_allowance", 8192)
        self.physical_count = len(memory.cells)


@dataclass
class Binding:
    spec: dict
    layout: dict
    generation: int
    tables: dict
    imports: list = field(default_factory=list)
    imports_done: set = field(default_factory=set)
    activated: bool = False
    activation_sent: bool = False
    grant_sequence: int = 0
    grant_pending: bool = False
    grant_to_send: bool = True
    cancelled: bool = False
    cancel_sent: bool = False
    generation_done: bool = False
    drain_seen: bool = False
    next_output: int = 1
    grant_context: int = 0
    low_wait: Optional[dict] = None
    denial_to_send: bool = False
    outputs_ready: dict = field(default_factory=dict)

    @property
    def key(self):
        return self.spec["request_id"] + ":" + str(self.generation)

    def identity(self, **extra):
        return dict(
            key=self.key,
            request=self.spec["request_id"],
            slot=self.layout["slot"],
            generation=self.generation,
            **extra,
        )


class SimplerHost:
    def __init__(self, world):
        self.w = world
        self.bindings: list[Optional[Binding]] = [None, None]
        self.generations = [0, 0]
        self.records = {}
        self.input_closed = False
        self.session_drained = False
        self.stop_sent = False

    def submit(self, spec):
        require(not self.input_closed, "submit after CloseInput")
        spec = ModelAdapter.lower(spec)
        require(spec["request_id"] not in self.records, "duplicate request registration")
        slot = next((i for i, binding in enumerate(self.bindings) if binding is None), None)
        if slot is None:
            return False
        layout = self.w.layout.slots[slot]
        self.generations[slot] += 1
        key = spec["request_id"] + ":" + str(self.generations[slot])
        tables = self.w.pool.reserve(key, layout["home"] // 8, spec["prompt_context"])
        require(tables is not None, "initial grouped lease unavailable")
        binding = Binding(spec, layout, self.generations[slot], tables)
        binding.grant_context = min(
            *(len(tables[g]) * GROUPS[g] for g in GROUPS),
            spec["prompt_context"] + spec["max_new_tokens"],
        )
        binding.imports = [h for pages in tables.values() for h in pages]
        self.bindings[slot] = binding
        self.records[spec["request_id"]] = binding
        self.w.memory.write(
            layout["snapshot_host"],
            {"history": 5, "position": spec["prompt_context"]},
            "Host",
        )
        self.w.memory.write(
            layout["activation_host"],
            dict(spec, key=binding.key, generation=binding.generation),
            "Host",
        )
        self.w.emit(
            "REQUEST_ACCEPTED",
            "SimplerHost.Admission",
            **binding.identity(),
            max_new_tokens=spec["max_new_tokens"],
            prompt_context=spec["prompt_context"],
        )
        return True

    def cancel(self, request):
        binding = self.records[request]
        binding.cancelled = True
        self.w.emit("CANCEL_ACCEPTED", "SimplerHost.Control", **binding.identity())

    def low(self, binding, payload):
        if binding.cancelled or binding.generation_done:
            return
        if binding.grant_pending or binding.grant_to_send:
            binding.low_wait = payload
            return
        if self.w.scenario == "deny_refill":
            binding.denial_to_send = binding.grant_to_send = True
            self.w.emit("REFILL_DENIED", "SimplerHost.CachePool", **binding.identity())
            return
        target = min(
            payload["claimed_context"] + 2,
            binding.spec["prompt_context"] + binding.spec["max_new_tokens"],
        )
        tables = self.w.pool.reserve(binding.key, binding.layout["home"] // 8, target)
        if tables is not None:
            binding.tables, binding.grant_context, binding.grant_to_send = (
                tables,
                target,
                True,
            )
        else:
            self.w.emit("WAIT_PHYSICAL_POOL", "SimplerHost.CachePool", **binding.identity())

    def receive_control(self, binding, kind):
        require(binding is not None, "control for unbound slot")
        payload = self.w.memory.read(binding.layout[kind + "_host"], "Host")
        require(payload["key"] == binding.key, "control outside bound generation")
        if kind == "low":
            self.low(binding, payload)
        elif kind == "grant_ack":
            require(payload["sequence"] == binding.grant_sequence, "stale Grant ACK")
            binding.grant_pending = False
            self.w.emit(
                "GRANT_TRANSACTION_ACKED",
                "SimplerHost.CachePool",
                **binding.identity(),
                sequence=payload["sequence"],
            )
        elif kind == "finished":
            binding.generation_done = True
            self.w.serving.generation_finished(binding, dict(reason=payload["reason"], emitted=payload["emitted"]))
        elif kind == "drained":
            binding.drain_seen = True

    def receive_control_slot(self, index, kind):
        self.receive_control(self.bindings[index], kind)

    def output_ready(self, index, sequence, key):
        binding = self.bindings[index]
        require(binding is not None and binding.key == key, "output for unbound generation")
        payload = self.w.memory.read(binding.layout["output_host"][(sequence - 1) % 2], "Host")
        require(
            payload["key"] == key and payload["sequence"] == sequence,
            "output identity mismatch",
        )
        binding.outputs_ready[sequence] = (sequence - 1) % 2
        self.w.emit(
            "HOST_OUTPUT_VISIBLE",
            "SimplerHost.Output",
            **binding.identity(sequence=sequence),
        )

    def send_control(self, binding, name, payload, callback):
        source, target = binding.layout[name + "_host"], binding.layout[name + "_core"]
        if self.w.memory.busy(source) or self.w.memory.busy(target):
            return False
        self.w.memory.write(source, payload, "Host")
        return self.w.transport.send(source, target, name, binding.identity(), callback, control=True)

    def step(self):
        for index, binding in enumerate(list(self.bindings)):
            if binding is None:
                continue
            slot = self.w.device.slots[index]
            if binding.low_wait is not None and not binding.grant_pending and not binding.grant_to_send:
                payload, binding.low_wait = binding.low_wait, None
                self.low(binding, payload)
            self.drive_import_and_activation(binding, slot)
            if binding.cancelled and binding.activation_sent and not binding.cancel_sent:
                if self.send_control(binding, "cancel", dict(key=binding.key), lambda s=slot: s.cancel()):
                    binding.cancel_sent = True
            self.consume_output(binding, slot)
            no_jobs = not self.w.transport.outstanding(binding.key)
            if binding.cancelled and not binding.activation_sent and no_jobs:
                binding.generation_done = binding.drain_seen = True
                self.w.serving.generation_finished(binding, dict(reason="cancelled_before_activation", emitted=0))
            if binding.drain_seen and not binding.grant_pending and no_jobs:
                self.w.pool.release(binding.key)
                self.w.emit("REQUEST_DRAINED", "SimplerHost.Lifecycle", **binding.identity())
                self.bindings[index] = None
                self.w.serving.released.add(binding.spec["request_id"])
        if self.input_closed and not any(self.bindings) and self.w.transport.idle() and not self.stop_sent:
            self.session_drained = True
            self.w.emit("SESSION_DRAINED", "SimplerHost.Session")
            self.w.memory.write(self.w.layout.stop_host, 1, "Host")
            if self.w.transport.send(
                self.w.layout.stop_host,
                self.w.layout.stop_core,
                "session_stop",
                {},
                self.w.device.stop,
                control=True,
            ):
                self.stop_sent = True

    def drive_import_and_activation(self, binding, slot):
        if not binding.cancelled:
            for target in binding.imports:
                if target in binding.imports_done or any(
                    j is not None and j.target == target for j in self.w.transport.jobs
                ):
                    continue
                if self.w.transport.send(
                    binding.layout["snapshot_host"],
                    target,
                    "prefill",
                    binding.identity(),
                    lambda b=binding, h=target: self.import_visible(b, h),
                ):
                    self.w.emit(
                        "PREFILL_PART_SUBMITTED",
                        "SimplerHost.Loader",
                        **binding.identity(),
                        target=target,
                    )
        imported = len(binding.imports_done) == len(binding.imports)
        if imported and not binding.cancelled:
            if binding.grant_to_send and not binding.grant_pending:
                sequence = binding.grant_sequence + 1
                self.w.pool.validate(binding.key, binding.tables)
                payload = dict(
                    key=binding.key,
                    request=binding.spec["request_id"],
                    generation=binding.generation,
                    sequence=sequence,
                    tables=binding.tables,
                    authorized_context=binding.grant_context,
                    denied=binding.denial_to_send,
                )
                if self.send_control(binding, "grant", payload, lambda s=slot: s.install_grant()):
                    binding.grant_sequence = sequence
                    binding.grant_pending, binding.grant_to_send = True, False
                    self.w.emit(
                        "GRANT_SENT",
                        "SimplerHost.CachePool",
                        **binding.identity(),
                        sequence=sequence,
                    )
            if not binding.activation_sent:
                if self.w.transport.send(
                    binding.layout["activation_host"],
                    binding.layout["activation_core"],
                    "activation",
                    binding.identity(),
                    lambda s=slot: s.activate(),
                    control=True,
                ):
                    binding.activation_sent = True

    def import_visible(self, binding, target):
        binding.imports_done.add(target)
        self.w.emit("PREFILL_PART_VISIBLE", "DirectDMA", **binding.identity(), target=target)

    def consume_output(self, binding, slot):
        sequence = binding.next_output
        index = (sequence - 1) % 2
        if sequence not in binding.outputs_ready:
            return
        if (
            self.w.scenario == "slow_output"
            and binding.spec["request_id"] == "B"
            and not any(e["event"] == "OUTPUT_RING_FULL" and e.get("key") == binding.key for e in self.w.events)
        ):
            return
        if len(self.w.serving.network) >= 4:
            return
        source, target = binding.layout["credit_host"], binding.layout["credit_core"]
        if self.w.memory.busy(source) or self.w.memory.busy(target):
            return
        payload = self.w.memory.read(binding.layout["output_host"][index], "Host")
        self.w.memory.write(source, dict(key=binding.key, sequence=sequence, index=index), "Host")
        if self.w.transport.send(
            source,
            target,
            "output_credit",
            binding.identity(sequence=sequence),
            lambda s=slot: s.credit(),
            control=True,
        ):
            del binding.outputs_ready[sequence]
            binding.next_output += 1
            self.w.serving.tokens(binding, payload)


class Serving:
    def __init__(self, world, specs):
        self.w = world
        self.pending = deque(specs)
        self.released = set()
        self.network = deque()
        self.received, self.finished = {}, {}
        self.terminal_sent = set()
        self.cancelled = set()
        self.backpressure_seen = set()
        self.submitted = set()

    def step(self):
        if self.pending:
            spec = self.pending[0]
            if spec["request_id"] not in self.submitted:
                self.submitted.add(spec["request_id"])
                self.w.emit(
                    "SUBMIT_REQUEST",
                    "Serving.RequestScheduler",
                    request=spec["request_id"],
                    max_new_tokens=spec["max_new_tokens"],
                    policy_epoch=spec["policy_epoch"],
                )
            if self.w.host.submit(spec):
                self.pending.popleft()
            elif spec["request_id"] not in self.backpressure_seen:
                self.backpressure_seen.add(spec["request_id"])
                self.w.emit(
                    "REQUEST_QUEUED",
                    "Serving.RequestScheduler",
                    request=spec["request_id"],
                    reason="no_free_slot",
                )
        if not self.pending and not self.w.host.input_closed:
            self.w.host.input_closed = True
            self.w.emit("CLOSE_INPUT", "Serving.RequestScheduler")
        client_blocked = self.w.scenario == "slow_client" and not any(
            e["event"] == "OUTPUT_CREDIT_INSTALLED" and e.get("request") == "B" for e in self.w.events
        )
        if self.network and self.w.tick % 11 == 0 and not client_blocked:
            request, payload = self.network.popleft()
            self.w.emit(
                "CLIENT_TOKEN_SENT",
                "Serving.Streamer",
                request=request,
                sequence=payload["sequence"],
            )
        for request, finish in self.finished.items():
            if (
                request not in self.terminal_sent
                and len(self.received.get(request, [])) == finish["emitted"]
                and not any(r == request for r, _ in self.network)
            ):
                self.terminal_sent.add(request)
                self.w.emit(
                    "CLIENT_GENERATION_FINISHED",
                    "Serving.Streamer",
                    request=request,
                    **finish,
                )

    def tokens(self, binding, payload):
        request = binding.spec["request_id"]
        self.received.setdefault(request, []).append(payload)
        if request not in self.cancelled:
            self.network.append((request, payload))
        else:
            self.w.emit(
                "CANCELLED_OUTPUT_DISCARDED",
                "Serving.Worker",
                request=request,
                sequence=payload["sequence"],
            )
        self.w.emit(
            "TOKEN_CHUNK_OWNED",
            "Serving.Worker",
            **binding.identity(sequence=payload["sequence"]),
            token=payload["token"],
            hidden=payload["hidden"],
        )

    def generation_finished(self, binding, payload):
        request = binding.spec["request_id"]
        require(request not in self.finished, "duplicate GenerationFinished")
        self.finished[request] = payload
        self.w.emit(
            "GENERATION_FINISHED_OBSERVED",
            "Serving.RequestScheduler",
            **binding.identity(),
            **payload,
        )
