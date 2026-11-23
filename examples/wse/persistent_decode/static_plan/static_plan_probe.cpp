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
#include "kv_quota.h"
#include <deployment_config.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace wp = wse::persistent;
namespace {
constexpr uint32_t kSlots = 4, kBanks = 2, kExperts = 8, kLanes = 7, kLayers = 40, kContext = 16;
constexpr uint64_t kModulus = 1000003;
constexpr size_t kPacketBytes = 128;
enum class Phase : uint32_t {
    Idle,
    ImportSend,
    ImportWait,
    FetchSend,
    FetchWait,
    Compute,
    Dispatch,
    Results,
    StoreSend,
    StoreWait,
    OutputSend,
    OutputWait,
    DrainOutputs,
    RequestRetire,
    Done
};
enum Kind : uint32_t { Import = 1, Fetch, ExpertInput, ExpertResult, Store, Output, OutputAck, ImportKV };

void require(bool pass, const char *message) {
    if (!pass) throw std::runtime_error(message);
}
template <typename T>
wp::Bytes pack(const T &value, size_t bytes = sizeof(T)) {
    static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
    require(bytes >= sizeof(T), "short packed allocation");
    wp::Bytes result(bytes);
    std::memcpy(result.data(), &value, sizeof(T));
    return result;
}
template <typename T>
T unpack(const wp::Bytes &bytes) {
    require(bytes.size() >= sizeof(T), "short unpacked allocation");
    T result;
    std::memcpy(&result, bytes.data(), sizeof(T));
    return result;
}
struct Frame {
    uint64_t generation = 0, hidden = 0, input = 0;
    uint32_t position = 0, layer = 0, requested = 0, emitted = 0, sent = 0, bitmap = 0;
    Phase phase = Phase::Idle;
    uint32_t fetch_ready = 0, store_ready = 0, import_ready = 0, output_ready = 0;
    std::array<uint32_t, kLanes> routes{};
    uint32_t cancel_after = 0;
};
struct ServiceFrame {
    uint32_t state = 0, lane = 0, layer = 0, position = 0;
    uint64_t generation = 0;
};
static_assert(
    sizeof(Frame) == 104 && sizeof(ServiceFrame) == 24, "CPU protocol frame ABI differs from program manifest"
);
struct ExpertBuffers {
    wp::Handle input = 0, output = 0, control = 0;
    ServiceFrame state;
};
struct Bank {
    std::array<wp::Handle, kLanes> send{}, result{};
    std::array<ExpertBuffers, kExperts + 1> expert{};
    std::array<uint32_t, kLanes> input_busy{};
};
struct Slot {
    Frame state;
    uint32_t home = 0;
    wp::Handle control = 0, seed = 0, host_seed = 0, fetch = 0, store = 0, output = 0, host_output = 0;
    wp::Handle host_ack = 0, ack = 0;
    std::array<wp::Handle, 4> deferred_output{}, deferred_host_output{};
    std::array<Frame, 4> deferred_state{};
    std::array<Bank, kBanks> banks{};
    bool output_held = false;
    uint64_t initial = 0, completed = 0;
};
struct Options {
    std::string scenario;
    bool reverse = false, hotspot = false, independent = false, output_pause = false, reuse = false, cancel = false,
         silent = false;
    uint32_t credits = deployment::inbox_entries_per_core, queue = deployment::queue_entries_per_endpoint, tokens = 3,
             slots = 4;
};
class Probe {
public:
    explicit Probe(const Options &options) :
        options_(options),
        fabric_(memory_, 16, 8, options.credits, options.queue) {
        for (uint32_t core = 0; core < 16; ++core)
            sram_[core] = memory_.add_region(wp::Tier::Sram, core, deployment::sram_per_core);
        for (uint32_t chip = 0; chip < 2; ++chip) {
            bailu_[chip] = memory_.add_region(wp::Tier::Bailu, chip * 8, deployment::bailu_per_chip);
            host_[chip] = memory_.add_region(wp::Tier::Host, chip * 8, deployment::host_per_chip);
        }
        for (uint32_t endpoint = 0; endpoint < fabric_.endpoint_count(); ++endpoint) {
            auto region = memory_.add_region(wp::Tier::Batcher, endpoint, deployment::batcher_per_endpoint);
            allocate(region, 8192);
        }
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            auto &s = slots_[slot];
            s.home = deployment::home_cores[slot];
            s.control = allocate(sram_[s.home], sizeof(Frame));
            s.seed = allocate(sram_[s.home], kPacketBytes);
            s.host_seed = allocate(host_[s.home / 8], kPacketBytes);
            s.fetch = allocate(sram_[s.home], kPacketBytes);
            s.store = allocate(sram_[s.home], kPacketBytes);
            s.output = allocate(sram_[s.home], kPacketBytes);
            s.host_output = allocate(host_[s.home / 8], kPacketBytes);
            s.host_ack = allocate(host_[s.home / 8], kPacketBytes);
            s.ack = allocate(sram_[s.home], kPacketBytes);
            for (uint32_t index = 0; index < 4; ++index) {
                s.deferred_output[index] = allocate(sram_[s.home], kPacketBytes);
                s.deferred_host_output[index] = allocate(host_[s.home / 8], kPacketBytes);
            }
            for (auto &bank : s.banks) {
                for (uint32_t lane = 0; lane < kLanes; ++lane) {
                    bank.send[lane] = allocate(sram_[s.home], kPacketBytes);
                    bank.result[lane] = allocate(sram_[s.home], kPacketBytes);
                }
                for (uint32_t expert = 0; expert <= kExperts; ++expert) {
                    auto &e = bank.expert[expert];
                    const auto owner = expert_owner(s, expert);
                    e.input = allocate(sram_[owner], kPacketBytes);
                    e.output = allocate(sram_[owner], kPacketBytes);
                    e.control = allocate(sram_[owner], sizeof(ServiceFrame));
                }
            }
        }
        quota_ = std::make_unique<kvquota::Controller>(
            memory_, sram_, bailu_, host_, options.scenario, [this](uint32_t region, size_t bytes) {
                return allocate(region, bytes);
            }
        );
        for (uint32_t slot = 0; slot < options.slots; ++slot)
            bind(slot, 11 + 7 * slot, options.tokens);
        snapshot_usage();
    }
    void run(const std::string &scenario, const std::filesystem::path &directory) {
        sealed_ = true;
        fabric_.record("SERVICE_LAUNCH", 0);
        bool rebound = false;
        for (uint32_t tick = 0; tick < 500000; ++tick) {
            fabric_.deliver_due();
            quota_->deliver();
            quota_->receive();
            quota_->device_step();
            for (uint32_t core = 0; core < 16; ++core)
                receive(core);
            for (uint32_t core = 0; core < 16; ++core)
                service(core);
            for (uint32_t slot = 0; slot < options_.slots; ++slot)
                step(slot);
            host_consume_outputs();
            quota_->host_step();
            if (fabric_.idle() && quota_->blocked_without_progress())
                throw std::runtime_error("KV quota wait has no runnable page owner in this closed trace");
            if (options_.reuse && !rebound && slots_[0].state.phase == Phase::Done) {
                bind(0, 97, 2);
                rebound = true;
            }
            for (uint32_t endpoint = 0; endpoint < fabric_.endpoint_count(); ++endpoint)
                fabric_.step_batcher(options_.reverse ? fabric_.endpoint_count() - 1 - endpoint : endpoint);
            quota_->advance(options_.reverse);
            check_usage();
            bool done = fabric_.idle() && quota_->idle();
            for (uint32_t slot = 0; slot < options_.slots; ++slot)
                done &= slots_[slot].state.phase == Phase::Done;
            if (done && (!options_.reuse || rebound)) {
                fabric_.record("SERVICE_DRAINED", 0);
                report(scenario, directory);
                return;
            }
            fabric_.advance_tick();
        }
        throw std::runtime_error("static plan failed to drain within logical step bound");
    }
    void negative(const std::string &scenario) {
        if (scenario == "context_overflow") bind(0, 11, kContext);
        else if (scenario == "slot_busy") bind(0, 11, 2);
        else if (scenario == "overwrite_after_dequeue") {
            auto &s = slots_[0];
            auto &bank = s.banks[0];
            auto &e = bank.expert[0];
            memory_.write_local(s.home, bank.send[0], pack(uint64_t{7}, kPacketBytes));
            require(
                fabric_.send({identity(0, 0), s.home, 1, 0, ExpertInput, bank.send[0], e.input}), "test send rejected"
            );
            for (uint32_t tick = 0; tick < 100 && !fabric_.has_message(1); ++tick) {
                fabric_.deliver_due();
                for (uint32_t endpoint = 0; endpoint < fabric_.endpoint_count(); ++endpoint)
                    fabric_.step_batcher(endpoint);
                fabric_.advance_tick();
            }
            require(fabric_.has_message(1), "test input did not arrive");
            receive(1);
            require(!memory_.in_use(e.input), "test did not consume mailbox notification");
            s.state.phase = Phase::Compute;
            save(0);
            step(0);
        } else if (scenario == "late_allocate") {
            sealed_ = true;
            allocate(sram_[0], 128);
        } else if (scenario == "stale_completion" || scenario == "duplicate_completion") {
            auto &s = slots_[0];
            s.state.phase = Phase::Results;
            auto key = identity(0, 0);
            if (scenario == "stale_completion") --key.generation;
            else s.state.bitmap = 1;
            accept_result(0, {key, 1, s.home, 0, ExpertResult, 0, s.banks[0].result[0]});
        } else if (scenario == "overwrite_inflight" || scenario == "premature_reclaim") {
            auto &s = slots_[0];
            memory_.write_local(s.home, s.store, pack(uint64_t{7}, kPacketBytes));
            require(
                fabric_.send({identity(0, 0), s.home, s.home, 0, Store, s.store, s.banks[0].result[0]}),
                "test send rejected"
            );
            if (scenario == "overwrite_inflight") memory_.write_local(s.home, s.store, pack(uint64_t{9}, kPacketBytes));
            else memory_.release(s.store);
        } else throw std::runtime_error("unknown negative scenario");
        throw std::runtime_error("negative scenario unexpectedly succeeded");
    }

private:
    Options options_;
    wp::Memory memory_;
    wp::Fabric fabric_;
    std::unique_ptr<kvquota::Controller> quota_;
    std::array<uint32_t, 16> sram_{};
    std::array<uint32_t, 2> bailu_{}, host_{};
    std::array<Slot, kSlots> slots_{};
    std::vector<uint64_t> fixed_usage_;
    bool sealed_ = false, independent_gate_ = false, independent_release_ = false, output_gate_ = false;
    uint32_t independent_peers_ = 0;
    uint64_t allocations_ = 0, binds_ = 0, token_count_ = 0, last_use_holds_ = 0, queue_stalls_ = 0;
    std::vector<std::string> outputs_;
    struct AllocationRecord {
        uint64_t handle, region, offset, bytes, charged_bytes;
    };
    std::vector<AllocationRecord> allocation_records_;

    wp::Handle allocate(uint32_t region, size_t bytes) {
        require(!sealed_, "physical allocation after launch");
        const auto offset = memory_.regions().at(region).used;
        const auto handle = memory_.allocate(region, bytes);
        ++allocations_;
        allocation_records_.push_back({handle, region, offset, bytes, (bytes + 127) / 128 * 128});
        return handle;
    }
    uint32_t expert_owner(const Slot &s, uint32_t expert) const {
        return expert == kExperts ? deployment::shared_cores[s.home / 8] : deployment::routed_cores[expert];
    }
    wp::Identity identity(uint32_t slot, uint32_t lane) const {
        const auto &f = slots_[slot].state;
        return {slot, f.generation, f.position, f.layer, 0, lane};
    }
    void save(uint32_t slot) {
        auto &s = slots_[slot];
        memory_.write_local(s.home, s.control, pack(s.state));
    }
    void bind(uint32_t slot, uint64_t initial, uint32_t tokens) {
        require(slot < options_.slots && tokens > 0 && tokens + 1 <= kContext, "context exceeds static profile");
        auto &s = slots_[slot];
        require(s.state.phase == Phase::Idle || s.state.phase == Phase::Done, "request slot still busy");
        require(!fabric_.outstanding(slot, s.state.generation) && !s.output_held, "rebind before drain");
        auto generation = s.state.generation + 1;
        quota_->bind(slot, generation);
        s.state = {};
        s.state.generation = generation;
        s.state.position = 1;
        s.state.requested = tokens;
        s.state.cancel_after = options_.cancel && slot == 1 ? 2 : 0;
        s.state.phase = Phase::ImportSend;
        s.initial = initial;
        memory_.stage_host(s.host_seed, pack(initial, kPacketBytes));
        ++binds_;
        fabric_.record("HOST_BIND", s.home, identity(slot, 0));
        save(slot);
    }
    void snapshot_usage() {
        for (const auto &region : memory_.regions())
            fixed_usage_.push_back(region.used);
    }
    void check_usage() const {
        require(memory_.regions().size() == fixed_usage_.size(), "device layout changed");
        for (size_t i = 0; i < fixed_usage_.size(); ++i)
            require(memory_.regions()[i].used == fixed_usage_[i], "physical device allocation changed during service");
    }
    bool send(wp::Message message) {
        if (fabric_.send(message)) return true;
        ++queue_stalls_;
        return false;
    }
    void accept_result(uint32_t slot, const wp::Message &message) {
        auto &s = slots_[slot];
        auto &f = s.state;
        require(
            message.key.generation == f.generation && message.key.position == f.position &&
                message.key.layer == f.layer,
            "stale result identity"
        );
        require(message.key.lane < kLanes && !(f.bitmap & (1u << message.key.lane)), "duplicate result identity");
        const auto &bank = s.banks[f.layer % kBanks];
        const auto lane = message.key.lane;
        require(message.destination == bank.result[lane], "result destination outside fixed plan");
        f.bitmap |= 1u << lane;
        save(slot);
    }
    void receive(uint32_t core) {
        if (!fabric_.has_message(core)) return;
        auto m = fabric_.receive(core);
        require(m.key.slot < options_.slots, "message slot outside profile");
        auto &s = slots_[m.key.slot];
        auto &f = s.state;
        require(
            m.key.generation == f.generation && m.key.position == f.position && m.key.layer == f.layer,
            "stale completion identity"
        );
        if (m.kind == Import) f.import_ready |= 1;
        else if (m.kind == ImportKV) f.import_ready |= 2;
        else if (m.kind == Fetch) f.fetch_ready = 1;
        else if (m.kind == Store) f.store_ready = 1;
        else if (m.kind == Output) {
            require(!s.output_held, "overwrite unacknowledged output");
            s.output_held = !options_.silent;
            f.output_ready = 1;
            fabric_.record("OUTPUT_REGION_READY", core, m.key);
        } else if (m.kind == OutputAck) {
            require(unpack<uint64_t>(memory_.read_local(s.home, s.ack)) == f.generation, "stale Host ack");
            f.output_ready = 2;
        } else if (m.kind == ExpertResult) accept_result(m.key.slot, m);
        else if (m.kind == ExpertInput) {
            require(m.expert <= kExperts && m.key.lane < kLanes, "invalid expert identity");
            auto &bank = s.banks[f.layer % kBanks];
            auto &e = bank.expert[m.expert];
            require(e.state.state == 0 && !bank.input_busy[m.key.lane], "overwrite input before last use");
            require(m.destination == e.input && expert_owner(s, m.expert) == core, "expert placement mismatch");
            bank.input_busy[m.key.lane] = 1;
            e.state = {1, m.key.lane, m.key.layer, m.key.position, m.key.generation};
            memory_.write_local(core, e.control, pack(e.state));
            fabric_.record("INPUT_LEASE_ACQUIRED", core, m.key);
        } else throw std::runtime_error("unknown message kind");
        save(m.key.slot);
    }
    void service(uint32_t core) {
        for (uint32_t slot = 0; slot < options_.slots; ++slot) {
            auto &s = slots_[slot];
            for (uint32_t bank_index = 0; bank_index < kBanks; ++bank_index) {
                auto &bank = s.banks[bank_index];
                for (uint32_t expert = 0; expert <= kExperts; ++expert) {
                    auto &e = bank.expert[expert];
                    if (!e.state.state || expert_owner(s, expert) != core) continue;
                    auto key = wp::Identity{slot, e.state.generation, e.state.position, e.state.layer, 0, e.state.lane};
                    if (e.state.state == 1) {
                        require(bank.input_busy[e.state.lane], "lost input lease");
                        auto x = unpack<uint64_t>(memory_.read_local(core, e.input));
                        memory_.write_local(
                            core, e.output,
                            pack((x * (expert + 2) + 17 * e.state.layer + expert) % kModulus, kPacketBytes)
                        );
                        e.state.state = 2;
                        ++last_use_holds_;
                        fabric_.record("EXPERT_COMPUTE", core, key);
                    } else if (e.state.state == 2) {
                        // Mailbox dequeue precedes this separately observed last-use
                        // boundary.
                        bank.input_busy[e.state.lane] = 0;
                        e.state.state = 3;
                        fabric_.record("INPUT_LAST_USE", core, key);
                    } else if (e.state.state == 3) {
                        if (send({key, core, s.home, expert, ExpertResult, e.output, bank.result[e.state.lane]}))
                            e.state.state = 4;
                    } else if (!memory_.in_use(e.output) && !memory_.in_use(bank.result[e.state.lane])) {
                        e.state.state = 0;
                        fabric_.record("EXPERT_BUFFER_FREE", core, key);
                    }
                    memory_.write_local(core, e.control, pack(e.state));
                    return;
                }
            }
        }
    }
    bool services_idle(uint32_t slot) const {
        for (const auto &bank : slots_[slot].banks)
            for (const auto &e : bank.expert)
                if (e.state.state) return false;
        return true;
    }
    void step(uint32_t slot) {
        auto &s = slots_[slot];
        s.state = unpack<Frame>(memory_.read_local(s.home, s.control));
        auto &f = s.state;
        auto &bank = s.banks[f.layer % kBanks];
        if (!quota_->allow_data(slot)) return;
        switch (f.phase) {
        case Phase::ImportSend:
            if (!quota_->ready(slot)) break;
            if (send({identity(slot, 0), s.home, s.home, 0, Import, s.host_seed, s.seed})) f.phase = Phase::ImportWait;
            break;
        case Phase::ImportWait:
            if (f.import_ready == 1 && !f.sent) {
                if (send(
                        {identity(slot, 0), s.home, s.home, 0, ImportKV, quota_->initial_source(slot),
                         quota_->page(slot, 0)}
                    ))
                    f.sent = 1;
            }
            if (f.import_ready == 3) {
                f.hidden = unpack<uint64_t>(memory_.read_local(s.home, s.seed));
                f.phase = Phase::FetchSend;
            }
            break;
        case Phase::FetchSend:
            if (!quota_->claim(slot, f.position, f.position < f.requested)) {
                if (quota_->denied(slot)) {
                    quota_->begin_retire(slot, true);
                    f.phase = Phase::RequestRetire;
                    fabric_.record("REQUEST_QUOTA_STOP", s.home, identity(slot, 0));
                }
                break;
            }
            if (send({identity(slot, 0), s.home, s.home, 0, Fetch, quota_->page(slot, f.position - 1), s.fetch}))
                f.phase = Phase::FetchWait;
            break;
        case Phase::FetchWait:
            if (f.fetch_ready) {
                if (options_.independent && slot == 0 && f.position == 1 && !independent_release_) {
                    if (!independent_gate_) fabric_.record("FETCH_GATE_CLOSED", s.home, identity(slot, 0));
                    independent_gate_ = true;
                    break;
                }
                f.hidden = (f.hidden + unpack<uint64_t>(memory_.read_local(s.home, s.fetch))) % kModulus;
                f.fetch_ready = 0;
                f.phase = Phase::Compute;
            }
            break;
        case Phase::Compute:
            f.input = (f.hidden * 3 + f.position + f.layer) % kModulus;
            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                f.routes[lane] = lane == 6 ? kExperts : (options_.hotspot ? lane : (f.input + f.layer + lane) % 8);
                require(!bank.input_busy[lane], "input lease still owned");
                memory_.write_local(s.home, bank.send[lane], pack(f.input, kPacketBytes));
            }
            f.sent = f.bitmap = 0;
            f.phase = Phase::Dispatch;
            break;
        case Phase::Dispatch:
            if (f.sent == kLanes) f.phase = Phase::Results;
            else {
                auto lane = f.sent, expert = f.routes[lane];
                auto &e = bank.expert[expert];
                if (e.state.state == 0 && send(
                                              {identity(slot, lane), s.home, expert_owner(s, expert), expert,
                                               ExpertInput, bank.send[lane], e.input}
                                          ))
                    ++f.sent;
            }
            break;
        case Phase::Results:
            if (f.bitmap == (1u << kLanes) - 1 && services_idle(slot)) {
                uint64_t sum = f.input;
                for (uint32_t lane = 0; lane < kLanes; ++lane)
                    sum += (lane + 1) * unpack<uint64_t>(memory_.read_local(s.home, bank.result[lane]));
                f.hidden = sum % kModulus;
                fabric_.record("LAYER_DONE", s.home, identity(slot, 0), f.hidden);
                quota_->layer_done(slot);
                if (options_.independent && independent_gate_ && !independent_release_) {
                    if (slot == 1 || slot == 2) independent_peers_ |= 1u << slot;
                    if (independent_peers_ == 6) {
                        independent_release_ = true;
                        fabric_.record("FETCH_GATE_RELEASED", s.home, identity(slot, 0));
                    }
                }
                if (++f.layer < kLayers) f.phase = Phase::Compute;
                else {
                    memory_.write_local(s.home, s.store, pack(f.hidden, kPacketBytes));
                    f.phase = Phase::StoreSend;
                }
            }
            break;
        case Phase::StoreSend:
            if (send({identity(slot, 0), s.home, s.home, 0, Store, s.store, quota_->page(slot, f.position)}))
                f.phase = Phase::StoreWait;
            break;
        case Phase::StoreWait:
            if (f.store_ready) {
                f.store_ready = 0;
                const auto output = options_.silent ? s.deferred_output.at(f.emitted) : s.output;
                memory_.write_local(s.home, output, pack(f.hidden % 97, kPacketBytes));
                f.phase = Phase::OutputSend;
            }
            break;
        case Phase::OutputSend:
            require(!s.output_held, "output buffer still owned by Host");
            if (send(
                    {identity(slot, 0), s.home, s.home, 0, Output,
                     options_.silent ? s.deferred_output.at(f.emitted) : s.output,
                     options_.silent ? s.deferred_host_output.at(f.emitted) : s.host_output}
                ))
                f.phase = Phase::OutputWait;
            break;
        case Phase::OutputWait:
            if (options_.silent) {
                if (f.output_ready == 1 && !fabric_.outstanding(slot, f.generation)) {
                    s.deferred_state.at(f.emitted) = f;
                    fabric_.record("TOKEN_PRODUCED_WITHOUT_HOST_ACK", s.home, identity(slot, 0));
                    fabric_.record("LOCAL_TOKEN_FEEDBACK", s.home, identity(slot, 0));
                    ++f.emitted;
                    f.output_ready = 0;
                    if (f.emitted == f.requested) f.phase = Phase::DrainOutputs;
                    else {
                        ++f.position;
                        f.layer = 0;
                        f.hidden %= 97;
                        f.phase = Phase::FetchSend;
                    }
                }
                break;
            }
            if (f.output_ready != 2 || fabric_.outstanding(slot, f.generation) || !services_idle(slot)) break;
            finish_output(slot);
            break;
        case Phase::DrainOutputs:
            if (f.output_ready == 2 && !fabric_.outstanding(slot, f.generation) && services_idle(slot)) {
                s.output_held = false;
                quota_->begin_retire(slot);
                f.phase = Phase::RequestRetire;
            }
            break;
        case Phase::RequestRetire:
            if (!fabric_.outstanding(slot, f.generation) && services_idle(slot) && !s.output_held &&
                quota_->retire(slot)) {
                f.phase = Phase::Done;
                ++s.completed;
                fabric_.record("REQUEST_DRAINED", s.home, identity(slot, 0));
            }
            break;
        case Phase::Idle:
        case Phase::Done:
            break;
        }
        save(slot);
    }
    void finish_output(uint32_t slot) {
        auto &s = slots_[slot];
        auto &f = s.state;
        require(
            !s.output_held && !fabric_.outstanding(slot, f.generation) && services_idle(slot), "output before drain"
        );
        const auto token = f.hidden % 97;
        fabric_.record("LOCAL_TOKEN_FEEDBACK", s.home, identity(slot, 0));
        ++f.emitted;
        s.output_held = false;
        f.output_ready = 0;
        if (f.emitted == f.requested || (f.cancel_after && f.emitted == f.cancel_after)) {
            if (f.cancel_after && f.emitted < f.requested)
                fabric_.record("TOKEN_BOUNDARY_CANCEL", s.home, identity(slot, 0));
            quota_->begin_retire(slot);
            f.phase = Phase::RequestRetire;
        } else {
            ++f.position;
            f.layer = 0;
            f.hidden = token;
            f.phase = Phase::FetchSend;
        }
    }
    void host_consume_outputs() {
        for (uint32_t slot = 0; slot < options_.slots; ++slot) {
            auto &s = slots_[slot];
            auto &f = s.state;
            if (options_.silent) {
                if (f.phase != Phase::DrainOutputs || s.output_held) continue;
                memory_.stage_host(s.host_ack, pack(f.generation, kPacketBytes));
                if (!send({identity(slot, 0), s.home, s.home, 0, OutputAck, s.host_ack, s.ack})) continue;
                for (uint32_t index = 0; index < f.emitted; ++index) {
                    const auto &produced = s.deferred_state[index];
                    const auto token = unpack<uint64_t>(memory_.inspect_after_drain(s.deferred_host_output[index]));
                    outputs_.push_back(
                        "{\"slot\":" + std::to_string(slot) + ",\"generation\":" + std::to_string(produced.generation) +
                        ",\"position\":" + std::to_string(produced.position) +
                        ",\"initial\":" + std::to_string(s.initial) + ",\"token\":" + std::to_string(token) +
                        ",\"hidden\":" + std::to_string(produced.hidden) + "}"
                    );
                    ++token_count_;
                }
                s.output_held = true;
                fabric_.record("OUTPUT_ACK", s.home, identity(slot, 0));
                continue;
            }
            if (!s.output_held || f.output_ready != 1) continue;
            if (options_.output_pause && slot == 0 && slots_[1].state.phase != Phase::Done) {
                if (!output_gate_) fabric_.record("OUTPUT_ACK_WITHHELD", s.home, identity(slot, 0));
                output_gate_ = true;
                continue;
            }
            memory_.stage_host(s.host_ack, pack(f.generation, kPacketBytes));
            if (!send({identity(slot, 0), s.home, s.home, 0, OutputAck, s.host_ack, s.ack})) continue;
            auto token = unpack<uint64_t>(memory_.inspect_after_drain(s.host_output));
            std::string row = "{\"slot\":" + std::to_string(slot) + ",\"generation\":" + std::to_string(f.generation) +
                              ",\"position\":" + std::to_string(f.position) +
                              ",\"initial\":" + std::to_string(s.initial) + ",\"token\":" + std::to_string(token) +
                              ",\"hidden\":" + std::to_string(f.hidden) + "}";
            outputs_.push_back(row);
            ++token_count_;
            s.output_held = false;
            fabric_.record("OUTPUT_ACK", s.home, identity(slot, 0));
            // ACK carries generation only; no token value returns from the Host.
        }
    }
    void report(const std::string &scenario, const std::filesystem::path &directory) const {
        std::filesystem::create_directories(directory);
        quota_->report(directory);
        std::ofstream out(directory / "results.json");
        out << "{\"scenario\":\"" << scenario << "\",\"deployment_digest\":\"" << deployment::digest
            << "\",\"launches\":1,\"binds\":" << binds_ << ",\"physical_allocations_before_launch\":" << allocations_
            << ",\"physical_allocations_during_service\":0,\"physical_releases_"
               "during_service\":0"
            << ",\"per_layer_host_decisions\":0,\"host_token_feedback\":0,"
               "\"tokens\":"
            << token_count_ << ",\"queue_stalls\":" << queue_stalls_ << ",\"credit_stalls\":" << fabric_.credit_stalls()
            << ",\"last_use_holds\":" << last_use_holds_
            << ",\"all_drained\":true,\"hotspot\":" << (options_.hotspot ? "true" : "false") << ",\"outputs\":[";
        for (size_t i = 0; i < outputs_.size(); ++i)
            out << (i ? "," : "") << outputs_[i];
        out << "],\"memory\":[";
        for (size_t i = 0; i < memory_.regions().size(); ++i) {
            const auto &r = memory_.regions()[i];
            out << (i ? "," : "") << "{\"tier\":" << static_cast<unsigned>(r.tier) << ",\"owner\":" << r.owner
                << ",\"used\":" << r.used << ",\"capacity\":" << r.capacity << "}";
        }
        out << "],\"transport\":{\"inbox_entries_per_core\":" << options_.credits
            << ",\"queue_entries_per_endpoint\":" << options_.queue << "}}\n";
        std::ofstream allocated(directory / "allocations.json");
        allocated << "[";
        for (size_t index = 0; index < allocation_records_.size(); ++index) {
            const auto &r = allocation_records_[index];
            allocated << (index ? "," : "") << "{\"handle\":" << r.handle << ",\"region\":" << r.region
                      << ",\"offset\":" << r.offset << ",\"bytes\":" << r.bytes
                      << ",\"charged_bytes\":" << r.charged_bytes << "}";
        }
        allocated << "]\n";
        std::ofstream trace(directory / "events.jsonl");
        for (const auto &e : fabric_.events())
            trace << "{\"tick\":" << e.tick << ",\"event\":\"" << e.event << "\",\"actor\":" << e.actor
                  << ",\"slot\":" << e.key.slot << ",\"generation\":" << e.key.generation
                  << ",\"position\":" << e.key.position << ",\"layer\":" << e.key.layer << ",\"lane\":" << e.key.lane
                  << ",\"transfer_id\":" << e.transfer_id << ",\"value\":" << e.bytes << "}\n";
    }
};
}  // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 3, "usage: static_plan_probe scenario output-directory");
        std::string scenario = argv[1];
        Options options;
        options.scenario = scenario;
        options.reverse = scenario == "reverse";
        options.hotspot = scenario == "hotspot";
        options.independent = scenario == "independent_progress";
        options.output_pause = scenario == "output_backpressure";
        options.reuse = scenario == "slot_reuse" || scenario == "kv_stale_generation";
        options.cancel = scenario == "token_boundary_cancel";
        options.silent = scenario == "host_silent_burst";
        if (scenario == "one_credit") options.credits = 1;
        if (scenario == "small_queue") {
            options.credits = 1;
            options.queue = 2;
        }
        if (scenario == "context_boundary") {
            options.tokens = kContext - 1;
            options.slots = 1;
        }
        if (scenario == "kv_watermark_edge") options.tokens = 6;
        Probe probe(options);
        const std::array<std::string, 8> negatives = {"context_overflow",   "slot_busy",
                                                      "overwrite_inflight", "premature_reclaim",
                                                      "stale_completion",   "duplicate_completion",
                                                      "late_allocate",      "overwrite_after_dequeue"};
        if (std::find(negatives.begin(), negatives.end(), scenario) != negatives.end()) probe.negative(scenario);
        else probe.run(scenario, argv[2]);
        std::cout << "passed: " << scenario << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    } catch (...) {
        return 1;
    }
}
