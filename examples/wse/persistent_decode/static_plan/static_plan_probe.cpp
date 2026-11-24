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
#include <tuple>

namespace wp = wse::persistent;
namespace {
constexpr uint32_t kSlots = 4, kBanks = 2, kExperts = 8, kLanes = 7, kLayers = 40, kContext = 16;
constexpr uint64_t kModulus = 1000003;
constexpr size_t kPacketBytes = 128;
enum class Phase : uint32_t {
    Idle,
    ImportWait,
    FetchSend,
    FetchWait,
    Compute,
    Dispatch,
    Results,
    StoreSend,
    RequestRetire,
    Done
};
enum Kind : uint32_t {
    Import = 1,
    Fetch,
    ExpertInput,
    ExpertResult,
    Store,
    Output,
    OutputAck,
    ImportKV,
    BackgroundImport,
    Retire
};

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
    uint32_t fetch_ready = 0;
    std::array<uint32_t, kLanes> routes{};
    uint32_t cancel_after = 0;
};
struct ServiceFrame {
    uint32_t state = 0, lane = 0, layer = 0, position = 0;
    uint64_t generation = 0;
};
static_assert(
    sizeof(Frame) == 88 && sizeof(ServiceFrame) == 24, "CPU protocol frame ABI differs from program manifest"
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
struct AsyncRecord {
    uint64_t generation = 0, value = 0;
    uint32_t position = 0, layer = 0, flags = 0, reserved = 0;
};
struct AsyncImage {
    uint64_t generation = 0, hot = 0;
    uint32_t hot_position = 0, flags = 0;
    std::array<AsyncRecord, 4> dirty{}, output{};
};
static_assert(sizeof(AsyncImage) <= 512);
struct Binding {
    uint64_t generation, initial;
    uint32_t tokens, cancel_after;
};
struct AsyncEntry {
    Frame frame{};
    bool live = false, sent = false, ready = false;
};
struct DirtyEntry {
    wp::Identity key{};
    bool live = false, sent = false;
};
struct Slot {
    Frame state;
    uint32_t home = 0;
    wp::Handle control = 0, fetch = 0, async_state = 0;
    wp::Handle host_ack = 0, ack = 0;
    std::array<wp::Handle, 4> deferred_output{}, deferred_host_output{};
    std::array<Bank, kBanks> banks{};
    wp::Handle activation_tx = 0, activation_rx = 0, retire_tx = 0, retire_rx = 0;
    std::array<wp::Handle, 4> dirty{};
    std::array<AsyncEntry, 4> output_entries{};
    std::array<DirtyEntry, 4> dirty_entries{};
    wp::Handle hot = 0;
    uint32_t hot_position = 0, host_next = 1;
    uint64_t host_generation = 0;
    bool activation_pending = false, retire_sent = false, retire_seen = false, host_busy = false;
    bool output_blocked = false, dirty_blocked = false;
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
        fabric_(memory_, options.credits, options.queue, options.scenario) {
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
            s.fetch = allocate(sram_[s.home], kPacketBytes);
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
            memory_, sram_, bailu_, host_, options.scenario,
            [this](uint32_t region, size_t bytes) {
                return allocate(region, bytes);
            },
            fabric_
        );
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            auto &a = slots_[slot];
            a.async_state = allocate(sram_[a.home], 512);
            a.activation_tx = allocate(host_[a.home / 8], 128);
            a.activation_rx = allocate(sram_[a.home], 128);
            a.retire_tx = allocate(sram_[a.home], 128);
            a.retire_rx = allocate(host_[a.home / 8], 128);
            for (auto &h : a.dirty)
                h = allocate(sram_[a.home], 128);
        }
        background_host_ = allocate(host_[0], 1024);
        background_bailu_[0] = allocate(bailu_[0], 1024);
        background_bailu_[1] = allocate(bailu_[0], 1024);
        memory_.stage_host(background_host_, pack(uint64_t{731}, 1024));
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
                receive(core, true);
            for (uint32_t core = 0; core < 16; ++core)
                if (!fabric_.engine().core_blocked(core)) receive(core);
            for (uint32_t core = 0; core < 16; ++core)
                if (!fabric_.engine().core_blocked(core)) service(core);
            drive_async();
            step_requests();
            host_consume_outputs();
            quota_->host_step();
            if (fabric_.idle() && quota_->blocked_without_progress())
                throw std::runtime_error("KV quota wait has no runnable page owner in this closed trace");
            if (options_.reuse && !rebound && !slots_[0].host_busy &&
                (slots_[0].state.phase == Phase::Done || options_.scenario == "hw_cancel_import")) {
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
        std::ostringstream pending;
        for (uint32_t slot = 0; slot < options_.slots; ++slot)
            pending << " slot=" << slot << ":phase=" << static_cast<uint32_t>(slots_[slot].state.phase)
                    << ",pos=" << slots_[slot].state.position << ",emitted=" << slots_[slot].state.emitted;
        throw std::runtime_error(
            "static plan failed to drain within logical step bound:" + pending.str() + fabric_.engine().pending()
        );
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
            memory_.write_local(s.home, s.dirty[0], pack(uint64_t{7}, kPacketBytes));
            require(
                fabric_.send({identity(0, 0), s.home, s.home, 0, Store, s.dirty[0], s.banks[0].result[0]}),
                "test send rejected"
            );
            if (scenario == "overwrite_inflight")
                memory_.write_local(s.home, s.dirty[0], pack(uint64_t{9}, kPacketBytes));
            else memory_.release(s.dirty[0]);
        } else throw std::runtime_error("unknown negative scenario");
        throw std::runtime_error("negative scenario unexpectedly succeeded");
    }

private:
    Options options_;
    wp::Memory memory_;
    hw::Fabric fabric_;
    std::unique_ptr<kvquota::Controller> quota_;
    std::array<uint32_t, 16> sram_{};
    std::array<uint32_t, 2> bailu_{}, host_{};
    std::array<Slot, kSlots> slots_{};
    std::vector<uint64_t> fixed_usage_;
    bool sealed_ = false, independent_gate_ = false, independent_release_ = false, output_gate_ = false;
    uint32_t independent_peers_ = 0;
    uint64_t allocations_ = 0, binds_ = 0, token_count_ = 0, last_use_holds_ = 0, queue_stalls_ = 0;
    std::vector<std::string> outputs_;
    wp::Handle background_host_ = 0;
    std::array<wp::Handle, 2> background_bailu_{};
    bool background_sent_ = false;
    bool import_cancelled_ = false;
    std::array<uint32_t, 16> core_cursor_{};
    struct AllocationRecord {
        uint64_t handle, region, offset, bytes, charged_bytes;
    };
    std::vector<AllocationRecord> allocation_records_;

    wp::Handle allocate(uint32_t region, size_t bytes) {
        require(!sealed_, "physical allocation after launch");
        const auto offset = memory_.regions().at(region).used;
        const auto handle = memory_.allocate(region, bytes);
        fabric_.engine().register_buffer(handle);
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
        AsyncImage image;
        image.generation = s.state.generation;
        image.hot = s.hot;
        image.hot_position = s.hot_position;
        image.flags = s.retire_sent;
        for (uint32_t index = 0; index < 4; ++index) {
            const auto &d = s.dirty_entries[index];
            const auto &o = s.output_entries[index];
            image.dirty[index] = {
                d.key.generation,
                0,
                d.key.position,
                d.key.layer,
                static_cast<uint32_t>(d.live) | (static_cast<uint32_t>(d.sent) << 1),
                0
            };
            image.output[index] = {
                o.frame.generation,
                o.frame.hidden,
                o.frame.position,
                o.frame.layer,
                static_cast<uint32_t>(o.live) | (static_cast<uint32_t>(o.sent) << 1),
                0
            };
        }
        memory_.write_local(s.home, s.async_state, pack(image, 512));
    }
    void bind(uint32_t slot, uint64_t initial, uint32_t tokens) {
        require(slot < options_.slots && tokens > 0 && tokens + 1 <= kContext, "context exceeds static profile");
        auto &a = slots_[slot];
        require(!a.host_busy, "request slot still busy");
        require(!fabric_.outstanding(slot, a.host_generation), "rebind before drain");
        ++a.host_generation;
        a.host_busy = true;
        a.activation_pending = true;
        a.retire_seen = a.retire_sent = false;
        a.host_next = 1;
        a.initial = initial;
        quota_->bind(slot, a.host_generation);
        memory_.stage_host(
            a.activation_tx,
            pack(Binding{a.host_generation, initial, tokens, options_.cancel && slot == 1 ? 2u : 0u}, 128)
        );
        ++binds_;
        fabric_.record("HOST_BIND", a.home, {slot, a.host_generation, 0, 0, 0, 0});
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
    void receive(uint32_t core, bool host = false) {
        if (!(host ? fabric_.has_host_message(core) : fabric_.has_message(core))) return;
        auto m = host ? fabric_.receive_host(core) : fabric_.receive(core);
        require(m.key.slot < options_.slots, "message slot outside profile");
        auto &s = slots_[m.key.slot];
        auto &f = s.state;
        if (m.kind == Import) {
            const auto b = unpack<Binding>(memory_.read_local(core, s.activation_rx));
            require(b.generation > f.generation && b.tokens + 1 <= kContext, "activation identity or capacity invalid");
            f = {};
            f.generation = b.generation;
            f.hidden = b.initial;
            f.requested = b.tokens;
            f.cancel_after = b.cancel_after;
            f.position = 1;
            f.phase = Phase::ImportWait;
            s.hot = 0;
            s.hot_position = 0;
            fabric_.record("ACTIVATION_INSTALLED", core, m.key);
            save(m.key.slot);
            return;
        }
        if (m.kind == Retire) {
            require(
                unpack<uint64_t>(memory_.inspect_after_drain(s.retire_rx)) == s.host_generation, "stale retire notice"
            );
            s.retire_seen = true;
            fabric_.record("REQUEST_DRAINED_DELIVERED", core, m.key);
            return;
        }
        if (m.kind == Store || m.kind == Output || m.kind == OutputAck) {
            require(m.key.generation == f.generation && m.key.lane < 4, "stale asynchronous completion");
            if (m.kind == Store) {
                auto &e = s.dirty_entries[m.key.lane];
                require(e.live && e.key.position == m.key.position, "stale KV writeback");
                e.live = false;
                fabric_.record("KV_BAILU_COMMITTED", core, m.key);
            } else if (m.kind == Output) {
                auto &e = s.output_entries[m.key.lane];
                require(e.live && e.frame.position == m.key.position, "stale output completion");
                e.ready = true;
                fabric_.record("OUTPUT_REGION_READY", core, m.key);
            } else {
                auto &e = s.output_entries[m.key.lane];
                require(e.live && e.frame.position == m.key.position, "stale output credit");
                e.live = false;
                fabric_.record("OUTPUT_CREDIT_RETURNED", core, m.key);
            }
            return;
        }
        require(
            m.key.generation == f.generation && m.key.position == f.position && m.key.layer == f.layer,
            "stale completion identity"
        );
        if (m.kind == Fetch) f.fetch_ready = 1;
        else if (m.kind == ExpertResult) accept_result(m.key.slot, m);
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
                        fabric_.engine().release_payload(e.input, key);
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
    void step_requests() {
        for (uint32_t core = 0; core < 16; ++core) {
            if (fabric_.engine().core_blocked(core)) continue;
            for (uint32_t offset = 0; offset < options_.slots; ++offset) {
                const auto slot = (core_cursor_[core] + offset) % options_.slots;
                if (slots_[slot].home != core) continue;
                const auto &f = slots_[slot].state;
                const auto before = std::make_tuple(f.phase, f.position, f.layer, f.emitted, f.sent, f.bitmap);
                step(slot);
                if (before != std::make_tuple(f.phase, f.position, f.layer, f.emitted, f.sent, f.bitmap)) {
                    core_cursor_[core] = (slot + 1) % options_.slots;
                    break;
                }
            }
        }
    }
    void step(uint32_t slot) {
        auto &s = slots_[slot];
        s.state = unpack<Frame>(memory_.read_local(s.home, s.control));
        auto &f = s.state;
        auto &bank = s.banks[f.layer % kBanks];
        if (!quota_->allow_data(slot)) return;
        switch (f.phase) {
        case Phase::ImportWait:
            if (quota_->ready_for(slot, f.generation)) f.phase = Phase::FetchSend;
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
            if (s.hot && s.hot_position == f.position - 1) {
                f.hidden = (f.hidden + unpack<uint64_t>(memory_.read_local(s.home, s.hot))) % kModulus;
                fabric_.record("KV_LOCAL_HIT", s.home, identity(slot, 0));
                f.phase = Phase::Compute;
            } else {
                auto message = wp::Message{
                    identity(slot, 0), s.home, s.home, 0, Fetch, quota_->page(slot, f.position - 1), s.fetch
                };
                bool accepted = options_.scenario == "hw_direct_blocking" && slot == 0 ? fabric_.send_direct(message) :
                                                                                         send(message);
                if (accepted) f.phase = Phase::FetchWait;
            }
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
            fabric_.record("CORE_COMPUTE_TILE", s.home, identity(slot, 0));
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
                if (send(
                        {identity(slot, lane), s.home, expert_owner(s, expert), expert, ExpertInput, bank.send[lane],
                         e.input}
                    ))
                    ++f.sent;
            }
            break;
        case Phase::Results:
            if (f.bitmap == (1u << kLanes) - 1) {
                uint64_t sum = f.input;
                for (uint32_t lane = 0; lane < kLanes; ++lane)
                    sum += (lane + 1) * unpack<uint64_t>(memory_.read_local(s.home, bank.result[lane]));
                f.hidden = sum % kModulus;
                for (uint32_t lane = 0; lane < kLanes; ++lane)
                    fabric_.engine().release_payload(bank.result[lane], identity(slot, lane));
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
                    f.phase = Phase::StoreSend;
                }
            }
            break;
        case Phase::StoreSend:
            publish_token(slot);
            break;
        case Phase::RequestRetire:
            if (async_idle(slot) && !fabric_.outstanding(slot, f.generation) && services_idle(slot)) {
                if (!s.retire_sent) {
                    memory_.write_local(s.home, s.retire_tx, pack(f.generation, 128));
                    if (send({identity(slot, 0), s.home, s.home, 0, Retire, s.retire_tx, s.retire_rx})) {
                        s.retire_sent = true;
                        fabric_.record("DEVICE_DATA_DRAINED_SENT", s.home, identity(slot, 0));
                    }
                } else f.phase = Phase::Done;
            }
            break;
        case Phase::Idle:
        case Phase::Done:
            break;
        }
        save(slot);
    }
    bool async_idle(uint32_t slot) const {
        const auto &s = slots_[slot];
        return std::none_of(
                   s.dirty_entries.begin(), s.dirty_entries.end(),
                   [](const auto &e) {
                       return e.live;
                   }
               ) &&
               std::none_of(s.output_entries.begin(), s.output_entries.end(), [](const auto &e) {
                   return e.live;
               });
    }
    void publish_token(uint32_t slot) {
        auto &s = slots_[slot];
        auto &f = s.state;
        const auto index = (f.position - 1) % 4;
        if (s.dirty_entries[index].live || memory_.in_use(s.dirty[index])) {
            if (!s.dirty_blocked) fabric_.record("DIRTY_RING_BLOCKED", s.home, identity(slot, index));
            s.dirty_blocked = true;
            return;
        }
        if (s.output_entries[index].live) {
            if (!s.output_blocked) fabric_.record("OUTPUT_RING_BLOCKED", s.home, identity(slot, index));
            s.output_blocked = true;
            return;
        }
        memory_.write_local(s.home, s.dirty[index], pack(f.hidden, 128));
        memory_.write_local(s.home, s.deferred_output[index], pack(f.hidden % 97, 128));
        s.dirty_entries[index] = {identity(slot, index), true, false};
        s.output_entries[index] = {f, true, false, false};
        s.hot = s.dirty[index];
        s.hot_position = f.position;
        fabric_.record("KV_RESIDENT_COMMITTED", s.home, identity(slot, index));
        fabric_.record("LOCAL_TOKEN_FEEDBACK", s.home, identity(slot, index));
        if (options_.silent) fabric_.record("TOKEN_PRODUCED_WITHOUT_HOST_ACK", s.home, identity(slot, index));
        ++f.emitted;
        if (f.emitted == f.requested || (f.cancel_after && f.emitted == f.cancel_after)) {
            if (f.emitted < f.requested) fabric_.record("TOKEN_BOUNDARY_CANCEL", s.home, identity(slot, index));
            quota_->begin_retire(slot);
            f.phase = Phase::RequestRetire;
        } else {
            ++f.position;
            f.layer = 0;
            f.hidden %= 97;
            f.phase = Phase::FetchSend;
        }
    }
    void drive_async() {
        const bool cancel = options_.scenario == "hw_cancel_import" || options_.scenario == "hw_rebind_during_import" ||
                            options_.scenario == "hw_reclaim_during_import";
        if (cancel && !import_cancelled_ && quota_->importing(0)) {
            quota_->cancel_import(0);
            slots_[0].activation_pending = false;
            import_cancelled_ = true;
            fabric_.record("HOST_CANCEL_IMPORT", 0, {0, slots_[0].host_generation});
            if (options_.scenario == "hw_rebind_during_import") bind(0, 97, 2);
            if (options_.scenario == "hw_reclaim_during_import") memory_.release(quota_->import_target(0));
        }
        if (cancel && import_cancelled_ && slots_[0].host_busy && slots_[0].host_generation == 1 &&
            quota_->finished(0)) {
            slots_[0].host_busy = false;
            fabric_.record("REQUEST_DRAINED", 0, {0, 1});
        }
        for (uint32_t slot = 0; slot < options_.slots; ++slot) {
            auto &s = slots_[slot];
            if (s.activation_pending && quota_->imported(slot)) {
                if (send(
                        {{slot, s.host_generation, 0, 0, 0, 0},
                         s.home,
                         s.home,
                         0,
                         Import,
                         s.activation_tx,
                         s.activation_rx}
                    ))
                    s.activation_pending = false;
            }
            for (uint32_t index = 0; index < 4; ++index) {
                if (fabric_.engine().core_blocked(s.home)) break;
                auto &d = s.dirty_entries[index];
                if (d.live && !d.sent &&
                    send({d.key, s.home, s.home, 0, Store, s.dirty[index], quota_->page(slot, d.key.position)}))
                    d.sent = true;
                auto &o = s.output_entries[index];
                if (o.live && !o.sent) {
                    wp::Identity key{slot, o.frame.generation, o.frame.position, o.frame.layer, 0, index};
                    if (send({key, s.home, s.home, 0, Output, s.deferred_output[index], s.deferred_host_output[index]}))
                        o.sent = true;
                }
            }
            if (s.retire_seen && quota_->retire(slot)) {
                s.retire_seen = false;
                s.host_busy = false;
                ++s.completed;
                fabric_.record("REQUEST_DRAINED", s.home, {slot, s.host_generation, 0, 0, 0, 0});
            }
        }
        if ((fabric_.engine().read_gate() || (options_.scenario == "hw_background_stream" && quota_->importing(0))) &&
            !background_sent_) {
            if (send(
                    {{4, 1, 0, 0, 0, 0},
                     0,
                     0,
                     0,
                     BackgroundImport,
                     background_host_,
                     background_bailu_[background_bailu_[0] % 2 == quota_->import_target(0) % 2 ? 0 : 1]}
                ))
                background_sent_ = true;
        }
    }
    void host_consume_outputs() {
        for (uint32_t slot = 0; slot < options_.slots; ++slot) {
            auto &s = slots_[slot];
            const auto index = (s.host_next - 1) % 4;
            auto &o = s.output_entries[index];
            if (!o.live || !o.ready || o.frame.position != s.host_next) continue;
            if (options_.silent && s.state.phase != Phase::RequestRetire && s.state.phase != Phase::Done) continue;
            if (options_.scenario == "hw_output_pressure" && slot == 0 && !s.output_blocked) continue;
            if (options_.output_pause && slot == 0 && slots_[1].host_busy) {
                if (!output_gate_)
                    fabric_.record("OUTPUT_ACK_WITHHELD", s.home, {slot, o.frame.generation, o.frame.position});
                output_gate_ = true;
                continue;
            }
            if (memory_.in_use(s.host_ack) || memory_.in_use(s.ack)) continue;
            memory_.stage_host(s.host_ack, pack(o.frame.generation, 128));
            wp::Identity key{slot, o.frame.generation, o.frame.position, o.frame.layer, 0, index};
            if (!send({key, s.home, s.home, 0, OutputAck, s.host_ack, s.ack})) continue;
            const auto token = unpack<uint64_t>(memory_.inspect_after_drain(s.deferred_host_output[index]));
            outputs_.push_back(
                "{\"slot\":" + std::to_string(slot) + ",\"generation\":" + std::to_string(o.frame.generation) +
                ",\"position\":" + std::to_string(o.frame.position) + ",\"initial\":" + std::to_string(s.initial) +
                ",\"token\":" + std::to_string(token) + ",\"hidden\":" + std::to_string(o.frame.hidden) + "}"
            );
            ++token_count_;
            ++s.host_next;
            o.ready = false;
            fabric_.record("OUTPUT_HOST_CONSUMED", s.home, key);
            fabric_.record("OUTPUT_ACK", s.home, key);
        }
    }
    void report(const std::string &scenario, const std::filesystem::path &directory) const {
        std::filesystem::create_directories(directory);
        quota_->report(directory);
        fabric_.engine().report(directory);
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
        options.reuse = scenario == "slot_reuse" || scenario == "kv_stale_generation" || scenario == "hw_cancel_import";
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
        if (scenario == "kv_watermark_edge" || scenario == "hw_dirty_pressure" || scenario == "hw_output_pressure")
            options.tokens = 6;
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
