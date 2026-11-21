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
#include "device.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace wse::persistent {

uint32_t Memory::add_region(Tier tier, uint32_t owner, uint64_t capacity) {
    if (!grants_.empty()) throw std::runtime_error("cannot change regions with live reservations");
    regions_.push_back({tier, owner, capacity});
    return regions_.size() - 1;
}

Memory::Allocation &Memory::get(Handle handle) {
    const auto found = allocations_.find(handle);
    if (found == allocations_.end())
        throw std::runtime_error("stale or unallocated memory handle " + std::to_string(handle));
    return found->second;
}
const Memory::Allocation &Memory::get(Handle handle) const {
    const auto found = allocations_.find(handle);
    if (found == allocations_.end())
        throw std::runtime_error("stale or unallocated memory handle " + std::to_string(handle));
    return found->second;
}
Reservation Memory::try_reserve(const std::vector<uint64_t> &bytes) {
    if (bytes.size() != regions_.size()) throw std::invalid_argument("reservation region count mismatch");
    for (size_t i = 0; i < bytes.size(); ++i) {
        const auto &region = regions_[i];
        if (bytes[i] % wse::SRAM_ALIGNMENT) throw std::invalid_argument("unaligned reservation");
        if (bytes[i] > region.capacity - region.used - region.reserved) return 0;
    }
    const auto id = next_grant_++;
    grants_.emplace(id, Grant{bytes, std::vector<uint64_t>(bytes.size())});
    for (size_t i = 0; i < bytes.size(); ++i) {
        auto &region = regions_[i];
        region.reserved += bytes[i];
        region.committed_peak = std::max(region.committed_peak, region.used + region.reserved);
    }
    return id;
}
void Memory::release_reservation(Reservation reservation) {
    const auto &grant = grants_.at(reservation);
    if (std::any_of(grant.used.begin(), grant.used.end(), [](auto used) {
            return used != 0;
        }))
        throw std::runtime_error("reservation still owns allocations");
    for (size_t i = 0; i < regions_.size(); ++i)
        regions_[i].reserved -= grant.limits[i];
    grants_.erase(reservation);
}
Handle Memory::allocate(uint32_t region, size_t bytes, Reservation reservation) {
    auto &target = regions_.at(region);
    if (!bytes || bytes > UINT64_MAX - wse::SRAM_ALIGNMENT) throw std::runtime_error("invalid allocation size");
    const auto charged = (bytes + wse::SRAM_ALIGNMENT - 1) / wse::SRAM_ALIGNMENT * wse::SRAM_ALIGNMENT;
    if (reservation) {
        const auto &grant = grants_.at(reservation);
        if (charged > grant.limits.at(region) - grant.used.at(region))
            throw std::runtime_error("request allocation exceeds reserved capacity");
    } else if (charged > target.capacity - target.used - target.reserved) {
        throw std::runtime_error("device region capacity exhausted");
    }
    const auto handle = next_++;
    Allocation allocation{region, Bytes(bytes)};
    allocation.charged = charged;
    allocation.reservation = reservation;
    allocations_.emplace(handle, std::move(allocation));
    if (reservation) {
        grants_.at(reservation).used[region] += charged;
        target.reserved -= charged;
    }
    target.used += charged;
    target.peak = std::max(target.peak, target.used);
    target.committed_peak = std::max(target.committed_peak, target.used + target.reserved);
    return handle;
}
void Memory::release(Handle handle) {
    const auto &allocation = get(handle);
    if (allocation.readers || allocation.writers || allocation.deliveries)
        throw std::runtime_error("reclaim before transfer consumed");
    regions_.at(allocation.region).used -= allocation.charged;
    if (allocation.reservation) {
        grants_.at(allocation.reservation).used[allocation.region] -= allocation.charged;
        regions_.at(allocation.region).reserved += allocation.charged;
    }
    allocations_.erase(handle);
}
void Memory::write_local(uint32_t core, Handle handle, const Bytes &bytes) {
    auto &allocation = get(handle);
    const auto &region = regions_.at(allocation.region);
    if (region.tier != Tier::Sram || region.owner != core) throw std::runtime_error("nonlocal SRAM write");
    if (allocation.readers || allocation.writers || allocation.deliveries)
        throw std::runtime_error("write while transfer owns allocation");
    if (allocation.bytes.size() != bytes.size()) throw std::runtime_error("write size mismatch");
    allocation.bytes = bytes;
}
const Bytes &Memory::read_local(uint32_t core, Handle handle) const {
    const auto &allocation = get(handle);
    const auto &region = regions_.at(allocation.region);
    if (region.tier != Tier::Sram || region.owner != core) throw std::runtime_error("nonlocal SRAM read");
    if (allocation.writers || !allocation.visible) throw std::runtime_error("read before transfer publication");
    return allocation.bytes;
}
void Memory::stage_host(Handle handle, const Bytes &bytes) {
    auto &allocation = get(handle);
    if (regions_.at(allocation.region).tier != Tier::Host || allocation.readers || allocation.writers ||
        allocation.deliveries || allocation.bytes.size() != bytes.size())
        throw std::runtime_error("invalid Host staging write");
    allocation.bytes = bytes;
}
void Memory::load_before_launch(Handle handle, const Bytes &bytes) {
    auto &allocation = get(handle);
    if (allocation.readers || allocation.writers || allocation.deliveries || allocation.bytes.size() != bytes.size())
        throw std::runtime_error("invalid boot load");
    allocation.bytes = bytes;
}
Bytes Memory::inspect_after_drain(Handle handle) const {
    const auto &allocation = get(handle);
    if (allocation.readers || allocation.writers || allocation.deliveries)
        throw std::runtime_error("inspection before drain");
    return allocation.bytes;
}
uint32_t Memory::region_of(Handle handle) const { return get(handle).region; }
size_t Memory::size(Handle handle) const { return get(handle).bytes.size(); }
bool Memory::in_use(Handle handle) const {
    return get(handle).readers || get(handle).writers || get(handle).deliveries;
}
void Memory::begin_transfer(Handle source, Handle destination) {
    auto &src = get(source);
    auto &dst = get(destination);
    if (source == destination || src.bytes.size() != dst.bytes.size() || src.writers || !src.visible || dst.readers ||
        dst.writers || dst.deliveries)
        throw std::runtime_error("invalid transfer allocation state");
    ++src.readers;
    ++dst.writers;
    ++dst.deliveries;
    dst.visible = false;
    ++live_transfers_;
}
void Memory::complete_transfer(Handle source, Handle destination) {
    auto &src = get(source);
    auto &dst = get(destination);
    if (!src.readers || dst.writers != 1) throw std::runtime_error("transfer completion without ownership");
    dst.bytes = src.bytes;
    --src.readers;
    --dst.writers;
    --live_transfers_;
}
void Memory::publish_transfer(Handle destination) {
    auto &dst = get(destination);
    if (dst.writers || dst.deliveries != 1 || dst.visible)
        throw std::runtime_error("publication before delivery or duplicate publication");
    dst.visible = true;
}
void Memory::consume_transfer(Handle destination) {
    auto &dst = get(destination);
    if (!dst.visible || dst.deliveries != 1) throw std::runtime_error("consume before publication");
    --dst.deliveries;
}
Bytes encode_values(const Values &values) {
    Bytes bytes(values.size() * sizeof(double));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}
Values decode_values(const Bytes &bytes) {
    if (bytes.size() % sizeof(double)) throw std::runtime_error("invalid dense numeric payload");
    Values values(bytes.size() / sizeof(double));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

Fabric::Fabric(
    Memory &memory, uint32_t cores, uint32_t cores_per_chip, uint32_t inbox_capacity, uint32_t queue_capacity
) :
    memory_(memory),
    cores_per_chip_(cores_per_chip),
    inbox_capacity_(inbox_capacity),
    queue_capacity_(queue_capacity),
    queues_((cores + 1) / 2),
    inboxes_(cores),
    credits_(cores, inbox_capacity) {
    if (!cores || !cores_per_chip || !inbox_capacity || !queue_capacity)
        throw std::invalid_argument("invalid fabric shape");
    set_program(
        {"ordered_push",
         {{InstructionKind::Invoke, 1, 0},
          {InstructionKind::Invoke, 2, 0},
          {InstructionKind::Invoke, 3, 0},
          {InstructionKind::Invoke, 4, 0},
          {InstructionKind::Halt, 0, 0}}}
    );
}
void Fabric::set_program(Program program) {
    if (!idle()) throw std::runtime_error("cannot replace live batcher program");
    program.validate();
    program_ = std::move(program);
}
void Fabric::register_operation(uint32_t id, std::function<StepResult(const Message &, uint32_t)> operation) {
    if (id <= 4 || !operation || !custom_operations_.emplace(id, std::move(operation)).second)
        throw std::invalid_argument("invalid custom batcher operation");
}
bool Fabric::send(Message message) {
    if (message.source_core >= inboxes_.size() || message.destination_core >= inboxes_.size())
        throw std::runtime_error("invalid message endpoint");
    if (memory_.regions().at(memory_.region_of(message.source)).owner != message.source_core)
        throw std::runtime_error("message source capability mismatch");
    const auto &destination = memory_.regions().at(memory_.region_of(message.destination));
    if (destination.owner != message.destination_core || destination.tier == Tier::Batcher)
        throw std::runtime_error("message destination capability mismatch");
    auto &queue = queues_.at(message.source_core / 2);
    if (queue.size() >= queue_capacity_) return false;
    memory_.begin_transfer(message.source, message.destination);
    message.transfer_id = next_transfer_id_++;
    Job job{};
    job.message = message;
    job.bytes = memory_.size(message.source);
    queue.push_back(std::move(job));
    record("SEND", message.source_core, message.key, memory_.size(message.source), message.transfer_id);
    return true;
}
bool Fabric::step_batcher(uint32_t endpoint) {
    auto &queue = queues_.at(endpoint);
    if (queue.empty()) return false;
    auto job = std::move(queue.front());
    queue.pop_front();
    OperationRegistry operations;
    operations.add(1, [&](uint32_t) {
        if (job.acquired) throw std::runtime_error("duplicate credit acquisition");
        auto &credit = credits_.at(job.message.destination_core);
        if (!credit) {
            ++credit_stalls_;
            return StepResult::Waiting;
        }
        --credit;
        job.acquired = true;
        return StepResult::Progress;
    });
    operations.add(2, [&](uint32_t) {
        if (!job.acquired || job.submitted) throw std::runtime_error("copy without unique credit");
        job.submitted = true;
        job.due = tick_ + 2 + (job.message.key.lane % 3) +
                  3 * (job.message.source_core / cores_per_chip_ != job.message.destination_core / cores_per_chip_);
        record("COPY_BEGIN", endpoint, job.message.key, job.bytes, job.message.transfer_id);
        return StepResult::Progress;
    });
    operations.add(3, [&](uint32_t) {
        return job.delivered ? StepResult::Progress : StepResult::Waiting;
    });
    operations.add(4, [&](uint32_t) {
        if (!job.delivered || job.published)
            throw std::runtime_error("publication before delivery or duplicate publication");
        memory_.publish_transfer(job.message.destination);
        inboxes_.at(job.message.destination_core).push_back(job.message);
        job.published = true;
        record("PUBLISH", job.message.destination_core, job.message.key, job.bytes, job.message.transfer_id);
        return StepResult::Progress;
    });
    for (const auto &custom : custom_operations_) {
        const auto callback = custom.second;
        operations.add(custom.first, [&, callback](uint32_t argument) {
            const auto result = callback(job.message, argument);
            if (result == StepResult::Progress) ++custom_program_steps_;
            return result;
        });
    }
    const auto result = job.continuation.step(program_, operations);
    if (result == StepResult::Halted) {
        if (!job.published) throw std::runtime_error("batcher halted without publication");
    } else {
        queue.push_back(std::move(job));
    }
    return result != StepResult::Waiting;
}
bool Fabric::deliver_due() {
    bool progress = false;
    for (auto &queue : queues_)
        for (auto &job : queue)
            if (job.submitted && !job.delivered && job.due <= tick_) {
                memory_.complete_transfer(job.message.source, job.message.destination);
                job.delivered = true;
                bytes_transferred_ += memory_.size(job.message.source);
                record("DELIVER", job.message.destination_core, job.message.key, job.bytes, job.message.transfer_id);
                progress = true;
            }
    return progress;
}
bool Fabric::has_message(uint32_t core) const { return !inboxes_.at(core).empty(); }
Message Fabric::receive(uint32_t core) {
    auto &inbox = inboxes_.at(core);
    if (inbox.empty()) throw std::runtime_error("receive on empty mailbox");
    auto message = inbox.front();
    inbox.pop_front();
    auto &credit = credits_.at(core);
    if (++credit > inbox_capacity_) throw std::runtime_error("credit over-return");
    record("CONSUMED", core, message.key, memory_.size(message.destination), message.transfer_id);
    memory_.consume_transfer(message.destination);
    return message;
}
bool Fabric::idle() const {
    return !memory_.live_transfers() && std::all_of(queues_.begin(), queues_.end(), [](const auto &q) {
        return q.empty();
    }) && std::all_of(inboxes_.begin(), inboxes_.end(), [](const auto &q) {
        return q.empty();
    }) && std::all_of(credits_.begin(), credits_.end(), [&](auto c) {
        return c == inbox_capacity_;
    });
}
bool Fabric::outstanding(uint32_t slot, uint64_t generation) const {
    const auto matches = [&](const Message &message) {
        return message.key.slot == slot && message.key.generation == generation;
    };
    for (const auto &queue : queues_)
        for (const auto &job : queue)
            if (matches(job.message)) return true;
    for (const auto &inbox : inboxes_)
        for (const auto &message : inbox)
            if (matches(message)) return true;
    return false;
}
void Fabric::record(const std::string &event, uint32_t actor, Identity key, uint64_t bytes, uint64_t transfer_id) {
    events_.push_back({tick_, event, actor, key, bytes, transfer_id});
}

}  // namespace wse::persistent
