# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generated bounded Core control flow; no generic on-device runtime/allocator."""

from collections import deque
from typing import Optional, cast

from model import GROUPS, MODULUS
from transport import require


class DeviceSlot:
    def __init__(self, world, layout):
        self.w, self.layout = world, layout
        self.home = layout["home"]
        self.key = None
        self.spec: dict = {}
        self.generation = 0
        self.phase = "idle"
        self.grant_key = None
        self.grant_sequence = 0
        self.tables = {g: () for g in GROUPS}
        self.authorized_context = 0
        self.claimed_context = 0
        self.low_notified = 0
        self.pending_control = deque()
        self.dirty: list[Optional[dict]] = [None, None]
        self.outputs: list[Optional[dict]] = [None, None]
        self.work: list[list[Optional[dict]]] = [[None] * 9 for _ in range(2)]
        self.hot = None
        self.last_token = 0
        self.sequence = 1
        self.layer = 0
        self.hidden = self.input_value = 0
        self.routes = []
        self.sent = 0
        self.received = set()
        self.stop_requested = False
        self.deny_next = False
        self.quota_blocked = self.output_blocked = self.dirty_blocked = False
        for lanes, experts in layout["banks"]:
            for _, result in lanes:
                world.transport.track(result)
            for _, source, _ in experts:
                world.transport.track(source)

    def identity(self, **extra):
        return dict(
            key=self.key,
            request=self.spec["request_id"] if self.spec else "",
            slot=self.layout["slot"],
            generation=self.generation,
            sequence=self.sequence,
            layer=self.layer,
            **extra,
        )

    def queue_control(self, name, payload):
        require(len(self.pending_control) < 4, "compiled control queue bound exceeded")
        self.pending_control.append((name, dict(key=self.key) | payload))

    def activate(self):
        payload = self.w.memory.read(self.layout["activation_core"], self.home)
        require(
            self.phase in {"idle", "retired"} and self.resources_free(),
            "activate before old request drain",
        )
        require(payload["generation"] > self.generation, "stale activation generation")
        self.spec, self.key, self.generation = (
            payload,
            payload["key"],
            payload["generation"],
        )
        self.phase, self.sequence, self.layer = "await_grant", 1, 0
        self.claimed_context = payload["prompt_context"]
        self.low_notified = 0
        self.hot, self.last_token = None, payload["seed"]
        self.stop_requested = self.deny_next = False
        self.output_blocked = self.dirty_blocked = self.quota_blocked = False
        self.w.emit("ACTIVATION_INSTALLED", "CoreProgram", **self.identity())

    def install_grant(self):
        payload = self.w.memory.read(self.layout["grant_core"], self.home)
        require(set(payload["tables"]) == set(GROUPS), "partial group grant")
        require(payload["generation"] >= self.generation, "stale grant generation")
        require(
            all(len(p) <= 8 for p in payload["tables"].values()),
            "grant exceeds fixed page table",
        )
        if self.phase not in {"idle", "retired"}:
            require(payload["key"] == self.key, "grant outside active generation")
        if payload["key"] != self.grant_key:
            require(
                self.phase in {"idle", "retired", "await_grant"},
                "new grant overwrites active generation",
            )
            self.grant_key, self.grant_sequence = payload["key"], 0
            self.tables = {g: () for g in GROUPS}
        require(payload["sequence"] == self.grant_sequence + 1, "grant sequence mismatch")
        require(
            all(tuple(payload["tables"][g][: len(self.tables[g])]) == self.tables[g] for g in GROUPS),
            "grant replaces committed group prefix",
        )
        self.tables = {g: tuple(p) for g, p in payload["tables"].items()}
        self.grant_sequence = payload["sequence"]
        self.authorized_context = payload["authorized_context"]
        self.deny_next = bool(payload.get("denied", False))
        self.queue_control("grant_ack", dict(key=payload["key"], sequence=self.grant_sequence))
        self.w.emit(
            "GROUP_GRANT_INSTALLED",
            "CoreProgram",
            key=payload["key"],
            request=payload.get("request", payload["key"].rsplit(":", 1)[0]),
            slot=self.layout["slot"],
            generation=payload["generation"],
            grant_sequence=self.grant_sequence,
            context=self.authorized_context,
            counts={g: len(p) for g, p in self.tables.items()},
        )

    def cancel(self):
        payload = self.w.memory.read(self.layout["cancel_core"], self.home)
        require(payload["key"] == self.key, "stale cancel generation")
        self.stop_requested = True
        self.w.emit("CANCEL_INSTALLED", "CoreProgram", **self.identity())

    def credit(self):
        payload = self.w.memory.read(self.layout["credit_core"], self.home)
        require(payload["key"] == self.key, "stale output credit generation")
        index = payload["index"]
        entry = cast(dict, self.outputs[index])
        require(
            entry is not None and entry["sequence"] == payload["sequence"],
            "stale output credit sequence",
        )
        self.outputs[index] = None
        self.w.emit(
            "OUTPUT_CREDIT_INSTALLED",
            "CoreProgram",
            **self.identity(),
            output_sequence=payload["sequence"],
        )

    def low(self):
        if self.low_notified == self.grant_sequence:
            return
        self.low_notified = self.grant_sequence
        self.queue_control("low", dict(claimed_context=self.claimed_context))
        self.w.emit(
            "LOW_QUEUED",
            "CoreProgram",
            **self.identity(),
            claimed_context=self.claimed_context,
        )

    def finish(self, reason):
        if self.phase in {"draining", "retired"}:
            return
        self.phase = "draining"
        self.queue_control("finished", dict(reason=reason, emitted=self.sequence - 1))
        self.w.emit(
            "GENERATION_FINISHED",
            "CoreProgram",
            **self.identity(),
            reason=reason,
            emitted=self.sequence - 1,
        )

    def resources_free(self):
        if any(self.dirty) or any(self.outputs) or any(any(bank) for bank in self.work):
            return False
        handles = [
            h for lanes, experts in self.layout["banks"] for h in [*[r for _, r in lanes], *[s for _, s, _ in experts]]
        ]
        return all(self.w.transport.payload_free[h] for h in handles)

    def drive_background(self):
        if self.key is None:
            return
        if self.pending_control:
            name, payload = self.pending_control[0]
            source, target = self.layout[name + "_core"], self.layout[name + "_host"]
            if not self.w.memory.busy(source) and not self.w.memory.busy(target):
                self.w.memory.write(source, payload, self.home)
                identity = self.identity()
                request, generation = payload["key"].rsplit(":", 1)
                identity.update(key=payload["key"], request=request, generation=int(generation))
                if self.w.transport.send(
                    source,
                    target,
                    name,
                    identity,
                    lambda i=self.layout["slot"], k=name: self.w.simpler.receive_control_slot(i, k),
                    control=True,
                ):
                    self.pending_control.popleft()
        for index, entry in enumerate(self.dirty):
            if entry is None:
                continue
            for group in GROUPS:
                if group in entry["sent"]:
                    continue
                if any(
                    j is not None and j.endpoint == self.home // 2 and j.kind == "writeback"
                    for j in self.w.transport.jobs
                ):
                    continue
                context = self.spec["prompt_context"] + entry["sequence"]
                target = self.tables[group][(context - 1) // GROUPS[group]]
                source = self.layout["dirty"][index][group]
                identity = dict(self.identity(), sequence=entry["sequence"], group=group)
                if self.w.transport.send(
                    source,
                    target,
                    "writeback",
                    identity,
                    lambda i=index, g=group, seq=entry["sequence"]: self.writeback_done(i, g, seq),
                ):
                    entry["sent"].add(group)
        for index, entry in enumerate(self.outputs):
            if entry is not None and not entry["sent"]:
                if any(
                    j is not None and j.endpoint == self.home // 2 and j.kind == "output" for j in self.w.transport.jobs
                ):
                    continue
                identity = dict(self.identity(), sequence=entry["sequence"])
                if self.w.transport.send(
                    self.layout["output"][index],
                    self.layout["output_host"][index],
                    "output",
                    identity,
                    lambda i=self.layout["slot"], seq=entry["sequence"], key=self.key: self.w.simpler.output_ready(
                        i, seq, key
                    ),
                ):
                    entry["sent"] = True

    def writeback_done(self, index, group, sequence):
        entry = cast(dict, self.dirty[index])
        require(
            entry is not None and entry["sequence"] == sequence,
            "writeback completion for reused entry",
        )
        entry["done"].add(group)
        self.w.emit(
            "KV_WRITEBACK_VISIBLE",
            "CoreProgram",
            **self.identity(),
            written_sequence=sequence,
            group=group,
        )
        if len(entry["done"]) == len(GROUPS):
            self.dirty[index] = None

    def receive_expert(self, bank, expert):
        owner, source, _ = self.layout["banks"][bank][1][expert]
        payload = self.w.memory.read(source, owner)
        require(self.work[bank][expert] is None, "expert work slot overwritten")
        require(
            payload["key"] == self.key and payload["layer"] == self.layer,
            "expert identity mismatch",
        )
        self.work[bank][expert] = dict(payload, state="compute")

    def expert_step(self, used):
        for bank in range(2):
            for expert, task in enumerate(self.work[bank]):
                if task is None:
                    continue
                owner, source, result = self.layout["banks"][bank][1][expert]
                if owner in used:
                    continue
                used.add(owner)
                if task["state"] == "compute":
                    data = self.w.memory.read(source, owner)
                    value = (data["value"] * (expert + 2) + 17 * data["layer"] + expert) % MODULUS
                    self.w.memory.write(result, value, owner)
                    task["state"] = "send"
                    self.w.emit(
                        "EXPERT_COMPUTE_LAST_USE",
                        "ExpertCore",
                        **self.identity(),
                        expert=expert,
                        core=owner,
                    )
                elif task["state"] == "send":
                    lane = task["lane"]
                    target = self.layout["banks"][bank][0][lane][1]
                    if self.w.transport.send(
                        result,
                        target,
                        "expert_result",
                        self.identity(lane=lane),
                        lambda b=bank, e=expert, ln=lane: self.result_ready(b, e, ln),
                    ):
                        task["state"] = "await_retirement"

    def result_ready(self, bank, expert, lane):
        task = self.work[bank][expert]
        require(
            task is not None and task["layer"] == self.layer and lane not in self.received,
            "stale or duplicate expert result",
        )
        self.received.add(lane)
        source = self.layout["banks"][bank][1][expert][1]
        self.w.transport.return_payload(source, self.identity(lane=lane))
        self.work[bank][expert] = None

    def step(self):
        self.drive_background()
        if self.phase in {"idle", "retired"}:
            return
        if self.phase == "await_grant":
            if self.grant_key == self.key:
                self.phase = "fetch"
        elif self.phase == "fetch":
            source = self.tables["window"][(self.spec["prompt_context"] - 1) // 128]
            if self.w.transport.send(source, self.layout["fetch"], "fetch", self.identity(), self.fetch_ready):
                self.phase = "fetch_wait"
        elif self.phase == "start":
            self.start_token()
        elif self.phase == "compute":
            self.prepare_layer()
        elif self.phase == "dispatch":
            bank, lane = self.layer % 2, self.sent
            expert = self.routes[lane]
            source = self.layout["banks"][bank][0][lane][0]
            target = self.layout["banks"][bank][1][expert][1]
            if self.w.transport.send(
                source,
                target,
                "expert_input",
                self.identity(lane=lane),
                lambda b=bank, e=expert: self.receive_expert(b, e),
            ):
                self.sent += 1
                if self.sent == 7:
                    self.phase = "collect"
        elif self.phase == "collect":
            self.collect_layer()
        elif self.phase == "commit_token":
            self.commit_token()
        elif self.phase == "draining":
            self.drain_request()

    def start_token(self):
        if self.stop_requested:
            self.finish("cancelled")
            return
        next_context = self.spec["prompt_context"] + self.sequence
        if next_context > self.authorized_context:
            if not self.quota_blocked:
                self.w.emit(
                    "KV_HARD_PAUSE",
                    "CoreProgram",
                    **self.identity(),
                    needed_context=next_context,
                )
            self.quota_blocked = True
            if self.deny_next:
                self.finish("capacity_denied")
            else:
                self.low()
            return
        index = (self.sequence - 1) % 2
        if self.dirty[index] is not None:
            if not self.dirty_blocked:
                self.w.emit("DIRTY_RING_FULL", "CoreProgram", **self.identity())
            self.dirty_blocked = True
            return
        if self.outputs[index] is not None:
            if not self.output_blocked:
                self.w.emit("OUTPUT_RING_FULL", "CoreProgram", **self.identity())
            self.output_blocked = True
            return
        self.claimed_context = next_context
        if (
            self.sequence < self.spec["max_new_tokens"]
            and self.authorized_context < self.spec["prompt_context"] + self.spec["max_new_tokens"]
            and self.authorized_context - next_context <= 1
        ):
            self.low()
        history = self.w.memory.read(self.hot, self.home)["hidden"] if self.hot is not None else self.initial_history
        if self.hot is not None:
            self.w.emit("KV_LOCAL_HIT", "CoreProgram", **self.identity())
        self.hidden = (self.last_token + history) % MODULUS
        self.layer, self.phase = 0, "compute"
        self.w.emit("TOKEN_START", "CoreProgram", **self.identity(), context=next_context)

    def prepare_layer(self):
        bank = self.layer % 2
        lanes, _ = self.layout["banks"][bank]
        if any(self.w.memory.busy(source) for source, _ in lanes):
            return
        self.input_value = (3 * self.hidden + self.sequence + self.layer) % MODULUS
        self.routes = [(self.input_value + self.layer + i) % 8 for i in range(6)] + [8]
        for lane, (source, _) in enumerate(lanes):
            self.w.memory.write(
                source,
                dict(
                    key=self.key,
                    value=self.input_value,
                    layer=self.layer,
                    lane=lane,
                ),
                self.home,
            )
        self.sent, self.received, self.phase = 0, set(), "dispatch"

    def collect_layer(self):
        if len(self.received) == 7:
            bank = self.layer % 2
            result = self.input_value
            for lane, (_, target) in enumerate(self.layout["banks"][bank][0]):
                result += (lane + 1) * self.w.memory.read(target, self.home)
                self.w.transport.return_payload(target, self.identity(lane=lane))
            self.hidden = result % MODULUS
            self.w.emit(
                "LAYER_COMMITTED",
                "CoreProgram",
                **self.identity(),
                hidden=self.hidden,
            )
            self.layer += 1
            self.phase = "compute" if self.layer < 40 else "commit_token"

    def drain_request(self):
        if self.resources_free() and not self.pending_control and not self.w.transport.outstanding(self.key):
            source, target = (
                self.layout["drained_core"],
                self.layout["drained_host"],
            )
            self.w.memory.write(source, dict(key=self.key), self.home)
            if self.w.transport.send(
                source,
                target,
                "drained",
                self.identity(),
                lambda i=self.layout["slot"]: self.w.simpler.receive_control_slot(i, "drained"),
                control=True,
            ):
                self.phase = "retired"

    def fetch_ready(self):
        payload = self.w.memory.read(self.layout["fetch"], self.home)
        self.initial_history, self.phase = payload["history"], "start"
        self.w.emit("REQUEST_ACTIVE", "CoreProgram", **self.identity())

    def commit_token(self):
        index = (self.sequence - 1) % 2
        require(
            self.dirty[index] is None and self.outputs[index] is None,
            "token commit overwrites live ring",
        )
        token = self.hidden % 97
        for group, handle in self.layout["dirty"][index].items():
            self.w.memory.write(
                handle,
                dict(
                    hidden=self.hidden,
                    context=self.claimed_context,
                    key=self.key,
                    group=group,
                ),
                self.home,
            )
        self.w.memory.write(
            self.layout["output"][index],
            dict(token=token, hidden=self.hidden, sequence=self.sequence, key=self.key),
            self.home,
        )
        self.dirty[index] = dict(sequence=self.sequence, sent=set(), done=set())
        self.outputs[index] = dict(sequence=self.sequence, sent=False, ready=False)
        self.hot, self.last_token = self.layout["dirty"][index]["window"], token
        self.w.emit(
            "TOKEN_LOCAL_FEEDBACK",
            "CoreProgram",
            **self.identity(),
            token=token,
            hidden=self.hidden,
        )
        self.sequence += 1
        if token == self.spec["eos_token_id"]:
            self.finish("eos")
        elif self.sequence > self.spec["max_new_tokens"]:
            self.finish("length")
        elif self.stop_requested:
            self.finish("cancelled")
        else:
            self.phase = "start"


class DeviceProgram:
    def __init__(self, world):
        self.w = world
        self.slots = [DeviceSlot(world, layout) for layout in world.layout.slots]
        self.exited = False

    def step(self):
        used = set()
        for slot in self.slots:
            slot.expert_step(used)
        for slot in self.slots:
            slot.step()
            self.w.memory.write(
                slot.layout["frame"],
                dict(
                    key=slot.key,
                    generation=slot.generation,
                    phase=slot.phase,
                    sequence=slot.sequence,
                    layer=slot.layer,
                    authorized_context=slot.authorized_context,
                    claimed_context=slot.claimed_context,
                ),
                slot.home,
            )

    def stop(self):
        require(
            all(s.phase in {"idle", "retired"} and s.resources_free() for s in self.slots),
            "persistent kernel exit before request drain",
        )
        self.exited = True
        self.w.emit("PERSISTENT_KERNEL_EXIT", "CoreProgram")
