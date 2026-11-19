/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#pragma once

#include "program.h"
#include "common/hardware_spec.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace wse::persistent {

using Bytes = std::vector<uint8_t>;
using Values = std::vector<double>;
using Handle = uint64_t;

enum class Tier { Sram, Bailu, Batcher, Host };
struct Region {
    Tier tier;
    uint32_t owner;
    uint64_t capacity;
    uint64_t used = 0;
    uint64_t peak = 0;
};

class Memory {
public:
    uint32_t add_region(Tier tier, uint32_t owner, uint64_t capacity);
    Handle allocate(uint32_t region, size_t bytes);
    void release(Handle handle);
    void write_local(uint32_t core, Handle handle, const Bytes &bytes);
    const Bytes &read_local(uint32_t core, Handle handle) const;
    void load_before_launch(Handle handle, const Bytes &bytes);
    Bytes inspect_after_drain(Handle handle) const;
    uint32_t region_of(Handle handle) const;
    const std::vector<Region> &regions() const { return regions_; }
    size_t size(Handle handle) const;
    bool in_use(Handle handle) const;
    uint64_t live_transfers() const { return live_transfers_; }
    void begin_transfer(Handle source, Handle destination);
    void complete_transfer(Handle source, Handle destination);
    void publish_transfer(Handle destination);
    void consume_transfer(Handle destination);

private:
    struct Allocation {
        uint32_t region;
        Bytes bytes;
        uint32_t readers = 0;
        uint32_t writers = 0;
        uint32_t deliveries = 0;
        bool visible = true;
        uint64_t charged = 0;
    };
    Allocation &get(Handle handle);
    const Allocation &get(Handle handle) const;
    std::vector<Region> regions_;
    std::map<Handle, Allocation> allocations_;
    Handle next_ = 1;
    uint64_t live_transfers_ = 0;
};

Bytes encode_values(const Values &values);
Values decode_values(const Bytes &bytes);

struct Message {
    Identity key;
    uint32_t source_core;
    uint32_t destination_core;
    uint32_t expert = 0;
    uint32_t kind = 0;
    Handle source = 0;
    Handle destination = 0;
    uint64_t transfer_id = 0;
};
static_assert(std::is_trivially_copyable_v<Message> && std::is_standard_layout_v<Message>);

struct TraceEvent {
    uint64_t tick;
    std::string event;
    uint32_t actor;
    Identity key;
    uint64_t bytes;
    uint64_t transfer_id;
};

enum class BatcherOperation : uint32_t { Acquire = 1, Copy = 2, AwaitDelivery = 3, Publish = 4 };

class Fabric {
public:
    Fabric(
        Memory &memory, uint32_t cores, uint32_t cores_per_chip, uint32_t inbox_capacity = 2,
        uint32_t queue_capacity = 32
    );
    void set_program(Program program);
    void register_operation(uint32_t id, std::function<StepResult(const Message &, uint32_t)> operation);
    bool send(Message message);
    bool step_batcher(uint32_t endpoint);
    bool deliver_due();
    bool has_message(uint32_t core) const;
    Message receive(uint32_t core);
    bool idle() const;
    bool outstanding(uint32_t slot, uint64_t generation) const;
    uint64_t tick() const { return tick_; }
    void advance_tick() { ++tick_; }
    uint64_t bytes_transferred() const { return bytes_transferred_; }
    uint64_t credit_stalls() const { return credit_stalls_; }
    uint64_t custom_program_steps() const { return custom_program_steps_; }
    uint32_t endpoint_count() const { return queues_.size(); }
    const std::vector<TraceEvent> &events() const { return events_; }
    void
    record(const std::string &event, uint32_t actor, Identity key = {}, uint64_t bytes = 0, uint64_t transfer_id = 0);
    const Program &program() const { return program_; }

private:
    struct Job {
        Message message;
        Continuation continuation;
        bool acquired = false;
        bool submitted = false;
        bool delivered = false;
        bool published = false;
        uint64_t due = 0;
        uint64_t bytes = 0;
    };
    Memory &memory_;
    uint32_t cores_per_chip_;
    uint32_t inbox_capacity_;
    uint32_t queue_capacity_;
    std::vector<std::deque<Job>> queues_;
    std::vector<std::deque<Message>> inboxes_;
    std::vector<uint32_t> credits_;
    std::map<uint32_t, std::function<StepResult(const Message &, uint32_t)>> custom_operations_;
    std::vector<TraceEvent> events_;
    Program program_;
    uint64_t tick_ = 0;
    uint64_t bytes_transferred_ = 0;
    uint64_t credit_stalls_ = 0;
    uint64_t custom_program_steps_ = 0;
    uint64_t next_transfer_id_ = 1;
};

}  // namespace wse::persistent
