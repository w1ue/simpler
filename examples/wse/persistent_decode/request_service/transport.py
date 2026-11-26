# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded functional transport and ownership model, with logical service units."""

import copy
from dataclasses import dataclass
from typing import Optional, cast


class Violation(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Violation(message)


@dataclass
class Cell:
    tier: str
    owner: int
    name: str
    value: object = 0
    readers: int = 0
    writers: int = 0
    deliveries: int = 0
    visible: bool = True
    size: int = 128


class Memory:
    def __init__(self):
        self.cells = []
        self.sealed = False
        self.released = False

    def allocate(self, tier, owner, name, size=128):
        require(not self.sealed, "device allocation after launch")
        self.cells.append(Cell(tier, owner, name, size=size))
        return len(self.cells) - 1

    def busy(self, handle):
        c = self.cells[handle]
        return bool(c.readers or c.writers or c.deliveries)

    def access(self, handle, actor):
        c = self.cells[handle]
        require(not self.released, "access after unload")
        require(
            (actor == "Host" and c.tier == "Host")
            or (isinstance(actor, int) and c.tier == "SRAM" and c.owner == actor),
            "invalid local memory owner",
        )
        return c

    def write(self, handle, value, actor):
        c = self.access(handle, actor)
        require(not self.busy(handle), "overwrite while DMA or delivery owns buffer")
        c.value = copy.deepcopy(value)
        c.visible = True

    def read(self, handle, actor):
        c = self.access(handle, actor)
        require(c.visible and not c.writers, "read before publication")
        return copy.deepcopy(c.value)

    def begin(self, source, target):
        s, t = self.cells[source], self.cells[target]
        require(
            source != target and s.visible and not s.writers and not self.busy(target),
            "invalid DMA ownership",
        )
        require(s.size == t.size, "transfer size mismatch")
        s.readers += 1
        t.writers += 1
        t.deliveries += 1
        t.visible = False

    def deliver(self, source, target):
        s, t = self.cells[source], self.cells[target]
        require(s.readers > 0 and t.writers == 1, "completion without DMA ownership")
        t.value = copy.deepcopy(s.value)
        s.readers -= 1
        t.writers -= 1

    def publish(self, target):
        c = self.cells[target]
        require(not c.writers and c.deliveries == 1 and not c.visible, "invalid publication")
        c.visible = True

    def consume(self, target):
        c = self.cells[target]
        require(c.visible and c.deliveries == 1, "consume before publication")
        c.deliveries -= 1


@dataclass
class Job:
    number: int
    source: int
    target: int
    path: str
    kind: str
    endpoint: int
    chip: int
    control: bool
    identity: dict
    callback: object
    stage: str = "queued"
    due: int = 0
    remaining: int = 128
    ingress: int = 0


class Transport:
    def __init__(self, memory, emit, gate):
        self.memory, self.emit, self.gate = memory, emit, gate
        self.jobs: list[Optional[Job]] = [None] * 256
        self.tick = 0
        self.next_number = 1
        self.contexts = [0] * 8
        self.data_contexts = [0] * 8
        self.cursor = [0] * 8
        self.turn = [False] * 8
        self.service_cursor = {}
        self.reserved_ports = set()
        self.payload_free = {}
        self.returns = []

    def track(self, target):
        self.payload_free[target] = True

    def event(self, job, name, **extra):
        self.emit(
            name,
            "CoreLink" if job.path == "CorePush" else "DirectDMA" if job.path == "DirectImport" else "Batcher",
            **job.identity,
            transfer=job.number,
            path=job.path,
            kind=job.kind,
            endpoint=job.endpoint,
            source=job.source,
            target=job.target,
            **extra,
        )

    def send(self, source, target, kind, identity, callback, control=False):
        s, t = self.memory.cells[source], self.memory.cells[target]
        if isinstance(s.value, dict) and "key" in s.value:
            require(
                identity.get("key") == s.value["key"],
                "transfer identity differs from source payload",
            )
        path = (
            "DirectImport"
            if s.tier == "Host" and t.tier == "Bailu"
            else "CorePush"
            if s.tier == t.tier == "SRAM" and s.owner // 8 == t.owner // 8
            else "Batcher"
        )
        owner = s.owner if s.tier == "SRAM" else t.owner
        ep = owner // 2
        capacity = 8 if path == "DirectImport" else 16 if path == "CorePush" else 4 if control else 16
        count = sum(
            j is not None and j.path == path and j.control == control and (path == "DirectImport" or j.endpoint == ep)
            for j in self.jobs
        )
        if count >= capacity or self.memory.busy(target) or self.memory.cells[source].writers:
            return False
        if target in self.payload_free and not self.payload_free[target]:
            return False
        index = next((i for i, j in enumerate(self.jobs) if j is None), None)
        if index is None:
            return False
        self.memory.begin(source, target)
        if target in self.payload_free:
            self.payload_free[target] = False
        job = Job(
            self.next_number,
            source,
            target,
            path,
            kind,
            ep,
            owner // 8,
            control,
            dict(identity),
            callback,
        )
        job.remaining = s.size
        if path == "DirectImport":
            job.ingress = s.size
        self.next_number += 1
        self.jobs[index] = job
        self.event(job, "TRANSFER_ACCEPTED")
        return True

    def return_payload(self, target, identity):
        require(
            target in self.payload_free and not self.payload_free[target],
            "duplicate payload release",
        )
        require(not any(h == target for _, h, _ in self.returns), "duplicate pending credit")
        self.returns.append((self.tick + 2, target, dict(identity)))
        self.emit("PAYLOAD_LAST_USE", "CoreProgram", **identity, target=target)

    def publish(self, index):
        job = cast(Job, self.jobs[index])
        self.memory.publish(job.target)
        self.event(job, "DATA_VISIBLE")
        self.memory.consume(job.target)
        if job.path == "Batcher":
            self.contexts[job.endpoint] -= 1
            self.data_contexts[job.endpoint] -= not job.control
        self.jobs[index] = None
        job.callback()

    def deliver(self, job, written):
        target = self.memory.cells[job.target]
        if target.tier == "SRAM":
            if target.owner in written:
                return False
            written.add(target.owner)
        self.memory.deliver(job.source, job.target)
        job.stage = "delivered"
        self.event(job, "SOURCE_READ_DONE")
        return True

    def advance(self, tick):
        self.tick = tick
        for due, handle, identity in list(self.returns):
            if due <= tick:
                self.payload_free[handle] = True
                self.returns.remove((due, handle, identity))
                self.emit("PAYLOAD_CREDIT_RETURNED", "CoreLink", **identity, target=handle)
        written = set()
        for index, job in enumerate(self.jobs):
            if job is not None and job.path == "CorePush" and job.stage == "active" and job.due == tick:
                require(self.deliver(job, written), "reserved CorePush SRAM port conflict")
                self.publish(index)
        self.reserved_ports = {(core, due) for core, due in self.reserved_ports if due > tick}

        self.issue_direct(tick)
        self.advance_batcher(tick)
        self.service_external(tick, written)
        for index, job in enumerate(self.jobs):
            if job is not None and job.path == "DirectImport" and job.stage == "delivered":
                self.publish(index)
        require(all(0 <= n <= 4 for n in self.contexts), "batcher context bound")
        require(
            all(0 <= n <= 3 for n in self.data_contexts),
            "data consumed reserved control context",
        )

    def issue_direct(self, tick):
        direct_used, noc_used = set(), set()
        for job in self.jobs:
            if job is None or job.stage != "queued" or job.path == "Batcher":
                continue
            if job.path == "CorePush":
                port = (self.memory.cells[job.target].owner, tick + 2)
                if job.endpoint in noc_used or port in self.reserved_ports:
                    continue
                self.reserved_ports.add(port)
                noc_used.add(job.endpoint)
            elif job.chip in direct_used:
                continue
            else:
                direct_used.add(job.chip)
            job.stage = "active"
            job.due = tick + (2 if job.path == "CorePush" else 3)
            self.event(job, "TRANSFER_ISSUED")

    def advance_batcher(self, tick):
        # One physical batcher action per endpoint; data leaves a control context.
        for ep in range(8):
            for channel in (self.turn[ep], not self.turn[ep]):
                chosen = None
                for offset in range(len(self.jobs)):
                    index = (self.cursor[ep] + offset) % len(self.jobs)
                    job = self.jobs[index]
                    if job is None or job.path != "Batcher" or job.endpoint != ep or job.control != channel:
                        continue
                    if job.stage == "delivered" or (
                        job.stage == "queued" and self.contexts[ep] < (4 if channel else 3)
                    ):
                        chosen = index
                        break
                if chosen is None:
                    continue
                job = cast(Job, self.jobs[chosen])
                self.event(job, "BATCHER_BUDGET_USED")
                if job.stage == "delivered":
                    self.publish(chosen)
                else:
                    job.stage, job.due = "active", tick + 3
                    self.contexts[ep] += 1
                    self.data_contexts[ep] += not channel
                    self.event(job, "TRANSFER_ISSUED")
                self.cursor[ep], self.turn[ep] = (
                    (chosen + 1) % len(self.jobs),
                    not channel,
                )
                break

    def service_external(self, tick, written):
        for chip in range(2):
            for domain in ("bank0", "bank1", "h2d", "d2h", "cross"):
                key = (chip, domain)
                start = self.service_cursor.get(key, 0)
                for offset in range(len(self.jobs)):
                    job = self.jobs[(start + offset) % len(self.jobs)]
                    if (
                        job is None
                        or job.stage != "active"
                        or job.path == "CorePush"
                        or job.due > tick
                        or job.chip != chip
                    ):
                        continue
                    s, t = self.memory.cells[job.source], self.memory.cells[job.target]
                    bailu_handle = job.source if s.tier == "Bailu" else job.target if t.tier == "Bailu" else None
                    selected = (
                        "h2d"
                        if job.ingress
                        else "bank" + str(bailu_handle % 2)
                        if bailu_handle is not None
                        else "h2d"
                        if s.tier == "Host"
                        else "d2h"
                        if t.tier == "Host"
                        else "cross"
                    )
                    if selected != domain or self.gate(job):
                        continue
                    count = min(64, job.ingress if job.ingress else job.remaining)
                    self.event(job, "SHARED_SERVICE", domain=domain, bytes=count)
                    if job.ingress:
                        job.ingress -= count
                    else:
                        job.remaining -= count
                        if job.remaining == 0:
                            self.deliver(job, written)
                    self.service_cursor[key] = (start + offset + 1) % len(self.jobs)
                    break

    def idle(self):
        return not any(self.jobs) and not self.returns and all(self.payload_free.values())

    def outstanding(self, key):
        return any(j is not None and j.identity.get("key") == key for j in self.jobs)
