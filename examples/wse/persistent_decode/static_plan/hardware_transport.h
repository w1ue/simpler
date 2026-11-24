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
#include "device.h"
#include <deployment_config.h>

#include <algorithm>
#include <array>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace hw {
namespace wp = wse::persistent;
enum class Path : uint32_t { DirectImport, CorePush, Batcher, DirectCore };
enum class Stage : uint32_t { Queued, External, Delivered, Published };
struct Event {
    uint64_t tick, transfer;
    std::string event;
    wp::Identity key;
    Path path;
    uint32_t endpoint;
    uint64_t bytes;
    uint32_t bank = UINT32_MAX;
};
struct Job {
    bool live = false, control = false, credit = false, external = false;
    wp::Message message{};
    Path path = Path::Batcher;
    Stage stage = Stage::Queued;
    uint64_t due = 0, remaining = 0, ingress_remaining = 0, bytes = 0;
};
struct Credit {
    uint64_t due;
    wp::Handle handle;
    wp::Message message;
};
inline void require(bool value, const char *text) {
    if (!value) throw std::runtime_error(text);
}

class Engine {
    wp::Memory &memory_;
    std::vector<Job> jobs_;
    std::array<std::array<std::deque<size_t>, 16>, 2> inboxes_;
    std::array<std::array<uint32_t, 48>, 2> credits_{};
    std::array<uint64_t, 8> stepped_{};
    std::array<uint32_t, 8> cursor_{}, turn_{}, contexts_{};
    std::array<std::array<uint32_t, 2>, 2> bank_cursor_{};
    std::array<std::array<uint32_t, 3>, 2> link_cursor_{};
    std::array<uint64_t, 16> blocked_{};
    std::array<uint64_t, 16> noc_port_due_{};
    std::array<bool, 16> sram_written_{};
    std::vector<bool> payload_free_;
    std::vector<Credit> returns_;
    std::array<std::vector<wp::TraceEvent>, 2> events_;
    std::vector<Event> resources_;
    uint32_t data_credit_, data_queue_, control_credit_, control_queue_;
    uint64_t tick_ = 0, next_ = 1, queue_stalls_ = 0, bytes_ = 0;
    std::array<uint64_t, 2> credit_stalls_{};
    std::string scenario_;
    bool read_gate_ = false, read_gate_open_ = false, peer_progress_ = false;
    bool other_progress_ = false, control_progress_ = false, background_visible_ = false;
    bool overlap_released_ = false, dirty_released_ = false;
    bool cancel_gate_ = false, cancel_gate_open_ = false;

    uint32_t endpoint(const Job &j) const { return j.message.source_core / 2; }
    uint32_t credit_index(const Job &j) const {
        const auto destination = tier(j.message.destination);
        const auto domain = destination == wp::Tier::Host ? 2u : destination == wp::Tier::Bailu ? 1u : 0u;
        return domain * 16 + j.message.destination_core;
    }
    wp::Tier tier(wp::Handle h) const { return memory_.regions().at(memory_.region_of(h)).tier; }
    bool bailu(const Job &j) const {
        return tier(j.message.source) == wp::Tier::Bailu || tier(j.message.destination) == wp::Tier::Bailu;
    }
    uint32_t chip(const Job &j) const {
        return tier(j.message.destination) == wp::Tier::Bailu ? j.message.destination_core / 8 :
                                                                j.message.source_core / 8;
    }
    uint32_t bank(const Job &j) const {
        const auto h = tier(j.message.source) == wp::Tier::Bailu ? j.message.source : j.message.destination;
        return h % 2;
    }
    bool eligible(Job &j) {
        if (!j.live || j.path != Path::Batcher || j.stage == Stage::Published) return false;
        if (j.stage == Stage::Delivered) return true;
        const auto limit = deployment::batcher_wait_entries - (j.control ? 0 : 1);
        return j.stage == Stage::Queued && contexts_[endpoint(j)] < limit;
    }
    void trace(const Job &j, const std::string &name) {
        auto key = j.message.key;
        key.phase = j.message.kind;
        resources_.push_back(
            {tick_, j.message.transfer_id, name, key, j.path, endpoint(j), j.bytes, bailu(j) ? bank(j) : UINT32_MAX}
        );
    }
    void basic(const Job &j, const std::string &name, uint32_t actor) {
        events_[j.control].push_back({tick_, name, actor, j.message.key, j.bytes, j.message.transfer_id});
    }
    bool acquire(Job &j) {
        if (j.credit || j.path == Path::DirectImport) return true;
        auto &credit = credits_[j.control][credit_index(j)];
        if (!credit) {
            ++credit_stalls_[j.control];
            return false;
        }
        --credit;
        j.credit = true;
        return true;
    }
    void issue(Job &j) {
        require(j.stage == Stage::Queued, "duplicate transport issue");
        j.stage = Stage::External;
        j.due = tick_ + (j.path == Path::CorePush ? deployment::noc_latency_ticks : deployment::external_latency_ticks);
        basic(j, "COPY_BEGIN", endpoint(j));
        trace(
            j, j.path == Path::Batcher      ? "BATCHER_ISSUED" :
               j.path == Path::DirectImport ? "DIRECT_IMPORT_SUBMITTED" :
               j.path == Path::DirectCore   ? "CORE_DIRECT_READ_SUBMITTED" :
                                              "CORE_PUSH_SUBMITTED"
        );
        if (j.path == Path::Batcher) {
            ++contexts_[endpoint(j)];
            trace(j, "BATCHER_EXT_WAIT");
        }
        if (j.path == Path::DirectCore) blocked_[j.message.destination_core] = j.message.transfer_id;
    }
    bool deliver(Job &j) {
        if (tier(j.message.destination) == wp::Tier::Sram) {
            if (sram_written_[j.message.destination_core]) return false;
            sram_written_[j.message.destination_core] = true;
        }
        memory_.complete_transfer(j.message.source, j.message.destination);
        j.stage = Stage::Delivered;
        bytes_ += j.bytes;
        basic(j, "DELIVER", j.message.destination_core);
        trace(j, "SOURCE_READ_DONE");
        return true;
    }
    void publish(Job &j, size_t index) {
        memory_.publish_transfer(j.message.destination);
        basic(j, "PUBLISH", j.message.destination_core);
        j.stage = Stage::Published;
        if (j.path == Path::Batcher) --contexts_[endpoint(j)];
        trace(j, "RECEIVER_VISIBLE");
        if (j.message.kind == 101 && read_gate_) control_progress_ = true;
        if (j.path == Path::DirectImport) {
            trace(j, "DIRECT_IMPORT_VISIBLE");
            if (j.message.kind == 9) background_visible_ = true;
            consume(index);
        } else {
            inboxes_[j.control][j.message.destination_core].push_back(index);
            if (j.path == Path::DirectCore) blocked_[j.message.destination_core] = 0;
        }
    }
    bool hold_read(Job &j) {
        const bool cancel = scenario_ == "hw_cancel_import" || scenario_ == "hw_rebind_during_import" ||
                            scenario_ == "hw_reclaim_during_import";
        if (cancel && j.path == Path::DirectImport && j.message.key.slot == 0 && j.message.key.generation == 1) {
            if (!cancel_gate_) {
                cancel_gate_ = true;
                trace(j, "IMPORT_CANCEL_GATE_CLOSED");
            }
            if (other_progress_ && !cancel_gate_open_) {
                cancel_gate_open_ = true;
                trace(j, "IMPORT_CANCEL_GATE_OPENED");
            }
            return !cancel_gate_open_;
        }
        if (j.message.key.slot == 0 && scenario_ == "hw_overlap" && !overlap_released_ &&
            ((j.message.kind == 5 || j.message.kind == 6) && j.message.key.position == 1))
            return true;
        if (j.message.key.slot == 0 && scenario_ == "hw_dirty_pressure" && j.message.kind == 5 && !dirty_released_)
            return true;
        const bool compare = scenario_ == "hw_contention" || scenario_ == "hw_direct_blocking";
        if (!compare || j.message.kind != 2 || j.message.key.slot != 0 || j.message.key.position != 1) return false;
        if (!read_gate_) {
            read_gate_ = true;
            trace(j, "BAILU_READ_GATE_CLOSED");
        }
        if (!read_gate_open_ && other_progress_ && control_progress_ && background_visible_) {
            read_gate_open_ = true;
            trace(j, "BAILU_READ_GATE_OPENED");
        }
        return !read_gate_open_;
    }
    template <typename Select>
    void serve(uint32_t &cursor, uint64_t quantum, Select select) {
        for (size_t offset = 0; offset < jobs_.size(); ++offset) {
            auto index = (cursor + offset) % jobs_.size();
            auto &j = jobs_[index];
            if (!j.live || j.stage != Stage::External || j.due > tick_ || !select(j) || hold_read(j)) continue;
            const auto count = std::min(quantum, j.remaining);
            j.remaining -= count;
            trace(j, "EXTERNAL_SERVICE");
            resources_.back().bytes = count;
            cursor = static_cast<uint32_t>((index + 1) % jobs_.size());
            if (!j.remaining) deliver(j);
            return;
        }
    }

    void serve_link(uint32_t chip_id, uint32_t direction) {
        auto &cursor = link_cursor_[chip_id][direction];
        for (size_t off = 0; off < jobs_.size(); ++off) {
            const auto index = (cursor + off) % jobs_.size();
            auto &j = jobs_[index];
            if (!j.live || j.stage != Stage::External || j.due > tick_ || chip(j) != chip_id) continue;
            const auto actual = tier(j.message.source) == wp::Tier::Host      ? 0u :
                                tier(j.message.destination) == wp::Tier::Host ? 1u :
                                                                                2u;
            const bool ingress = j.path == Path::DirectImport && j.ingress_remaining;
            if (actual != direction || (!ingress && (j.path != Path::Batcher || bailu(j))) || hold_read(j)) continue;
            auto &remaining = ingress ? j.ingress_remaining : j.remaining;
            const auto count = std::min<uint64_t>(deployment::external_bytes_per_tick, remaining);
            remaining -= count;
            trace(j, "HOST_LINK_SERVICE");
            resources_.back().bytes = count;
            cursor = static_cast<uint32_t>((index + 1) % jobs_.size());
            if (!ingress && !remaining) deliver(j);
            return;
        }
    }

public:
    Engine(
        wp::Memory &memory, uint32_t data_credit, uint32_t data_queue, uint32_t control_credit, uint32_t control_queue,
        std::string scenario
    ) :
        memory_(memory),
        jobs_(8 * (data_queue + control_queue + deployment::noc_queue_entries) + deployment::direct_queue_entries),
        data_credit_(data_credit),
        data_queue_(data_queue),
        control_credit_(control_credit),
        control_queue_(control_queue),
        scenario_(std::move(scenario)) {
        stepped_.fill(UINT64_MAX);
        credits_[0].fill(data_credit);
        credits_[1].fill(control_credit);
        returns_.reserve(jobs_.size());
        payload_free_.resize(2048, true);
    }
    void register_buffer(wp::Handle handle) {
        require(handle < payload_free_.size(), "compiled buffer handle bound exceeded");
    }
    bool core_blocked(uint32_t core) const { return blocked_[core] != 0; }
    bool read_gate() const { return read_gate_; }
    bool read_gate_open() const { return read_gate_open_; }
    bool peer_progress() const { return peer_progress_; }
    uint64_t tick() const { return tick_; }
    uint64_t credit_stalls(bool control) const { return credit_stalls_[control]; }
    uint64_t queue_stalls() const { return queue_stalls_; }
    uint64_t bytes_transferred() const { return bytes_; }
    const std::vector<wp::TraceEvent> &events(bool control) const { return events_[control]; }
    void observe(bool control, const std::string &name, uint32_t actor, wp::Identity key, uint64_t value) {
        events_[control].push_back({tick_, name, actor, key, value, 0});
        if (name == "LAYER_DONE" && key.slot == 2 && read_gate_ && !read_gate_open_) peer_progress_ = true;
        if (name == "LAYER_DONE" && key.slot == 1) other_progress_ = true;
        if (name == "LOCAL_TOKEN_FEEDBACK" && key.slot == 0 && key.position == 2) overlap_released_ = true;
        if (name == "DIRTY_RING_BLOCKED" && key.slot == 0) dirty_released_ = true;
    }
    bool send(wp::Message message, bool control, bool direct_core = false) {
        require(message.source_core < 16 && message.destination_core < 16, "invalid message endpoint");
        const auto &src = memory_.regions().at(memory_.region_of(message.source));
        const auto &dst = memory_.regions().at(memory_.region_of(message.destination));
        require(
            src.owner == message.source_core && dst.owner == message.destination_core && dst.tier != wp::Tier::Batcher,
            "message source or destination capability mismatch"
        );
        Path path = src.tier == wp::Tier::Host && dst.tier == wp::Tier::Bailu ? Path::DirectImport :
                    direct_core                                               ? Path::DirectCore :
                    src.tier == wp::Tier::Sram && dst.tier == wp::Tier::Sram &&
                            message.source_core / 8 == message.destination_core / 8 ?
                                  Path::CorePush :
                                  Path::Batcher;
        uint32_t count = 0;
        for (const auto &j : jobs_)
            if (j.live && j.path == path && j.control == control &&
                (path == Path::DirectImport || endpoint(j) == message.source_core / 2))
                ++count;
        const auto bound = path == Path::DirectImport ? deployment::direct_queue_entries :
                           path == Path::CorePush     ? deployment::noc_queue_entries :
                           control                    ? control_queue_ :
                                                        data_queue_;
        if (count >= bound) {
            ++queue_stalls_;
            return false;
        }
        const bool tracked = dst.tier == wp::Tier::Sram && (message.kind == 3 || message.kind == 4);
        if (tracked && !payload_free_[message.destination]) {
            ++credit_stalls_[control];
            return false;
        }
        auto found = std::find_if(jobs_.begin(), jobs_.end(), [](const auto &j) {
            return !j.live;
        });
        require(found != jobs_.end(), "compiled transport job capacity exhausted");
        Job candidate;
        candidate.live = true;
        candidate.control = control;
        candidate.message = message;
        candidate.path = path;
        candidate.bytes = candidate.remaining = memory_.size(message.source);
        if (path == Path::DirectImport) candidate.ingress_remaining = candidate.bytes;
        if (path == Path::CorePush && !acquire(candidate)) return false;
        memory_.begin_transfer(message.source, message.destination);
        candidate.message.transfer_id = next_++;
        if (tracked) payload_free_[message.destination] = false;
        *found = candidate;
        basic(*found, "SEND", message.source_core);
        trace(*found, "TRANSPORT_ACCEPTED");
        return true;
    }
    bool completed(wp::Handle destination) const {
        for (const auto &j : jobs_)
            if (j.live && j.message.destination == destination) return false;
        return !memory_.in_use(destination);
    }
    void release_payload(wp::Handle handle, const wp::Identity &key) {
        if (payload_free_[handle]) return;
        require(
            std::none_of(
                returns_.begin(), returns_.end(),
                [&](const auto &r) {
                    return r.handle == handle;
                }
            ),
            "duplicate payload credit return"
        );
        wp::Message message{key, 0, 0, 0, 0, 0, handle};
        returns_.push_back({tick_ + deployment::noc_latency_ticks, handle, message});
        resources_.push_back({tick_, 0, "CONSUMER_LAST_USE", key, Path::CorePush, 0, handle});
    }
    void consume(size_t index) {
        auto &j = jobs_[index];
        require(j.live && j.stage == Stage::Published, "consume before publication");
        memory_.consume_transfer(j.message.destination);
        basic(j, "CONSUMED", j.message.destination_core);
        if (j.credit) {
            auto &credit = credits_[j.control][credit_index(j)];
            ++credit;
            require(credit <= (j.control ? control_credit_ : data_credit_), "notification credit over-return");
        }
        j.live = false;
    }
    bool has_message(uint32_t core, bool control, bool host = false) const {
        const auto &inbox = inboxes_[control][core];
        return std::any_of(inbox.begin(), inbox.end(), [&](auto index) {
            return (tier(jobs_[index].message.destination) == wp::Tier::Host) == host;
        });
    }
    wp::Message receive(uint32_t core, bool control, bool host = false) {
        auto &inbox = inboxes_[control][core];
        auto selected = std::find_if(inbox.begin(), inbox.end(), [&](auto index) {
            return (tier(jobs_[index].message.destination) == wp::Tier::Host) == host;
        });
        require(selected != inbox.end(), "receive on empty mailbox");
        const auto index = *selected;
        inbox.erase(selected);
        const auto message = jobs_[index].message;
        consume(index);
        return message;
    }
    void step_batcher(uint32_t ep) {
        if (stepped_[ep] == tick_) return;
        stepped_[ep] = tick_;
        for (uint32_t pass = 0; pass < 2; ++pass) {
            const auto channel = (turn_[ep] + pass) % 2;
            for (size_t off = 0; off < jobs_.size(); ++off) {
                const auto index = (cursor_[ep] + off) % jobs_.size();
                auto &j = jobs_[index];
                if (endpoint(j) != ep || j.control != static_cast<bool>(channel) || !eligible(j)) continue;
                if (j.stage == Stage::Queued) {
                    if (!acquire(j)) continue;
                    issue(j);
                } else publish(j, index);
                cursor_[ep] = static_cast<uint32_t>((index + 1) % jobs_.size());
                turn_[ep] = 1 - channel;
                trace(j, "BATCHER_BUDGET_USED");
                return;
            }
        }
    }
    void deliver_due() {
        sram_written_.fill(false);
        for (auto it = returns_.begin(); it != returns_.end();) {
            if (it->due > tick_) {
                ++it;
                continue;
            }
            payload_free_[it->handle] = true;
            resources_.push_back({tick_, 0, "CREDIT_RETURNED", it->message.key, Path::CorePush, 0, it->handle});
            it = returns_.erase(it);
        }
        std::array<bool, 8> noc_used{};
        std::array<bool, 2> direct_used{};
        for (size_t i = 0; i < jobs_.size(); ++i) {
            auto &j = jobs_[i];
            if (!j.live || j.path == Path::Batcher) continue;
            if (j.stage == Stage::Queued) {
                if (j.path == Path::CorePush && noc_used[endpoint(j)]) continue;
                if (j.path == Path::CorePush &&
                    noc_port_due_[j.message.destination_core] == tick_ + deployment::noc_latency_ticks)
                    continue;
                if (j.path == Path::DirectImport && direct_used[chip(j)]) continue;
                if (!acquire(j)) continue;
                issue(j);
                if (j.path == Path::CorePush)
                    noc_port_due_[j.message.destination_core] = tick_ + deployment::noc_latency_ticks;
                if (j.path == Path::CorePush) noc_used[endpoint(j)] = true;
                if (j.path == Path::DirectImport) direct_used[chip(j)] = true;
            }
            if (j.path == Path::CorePush && j.stage == Stage::External && j.due <= tick_) deliver(j);
            if (j.stage == Stage::Delivered) publish(j, i);
        }
        for (uint32_t c = 0; c < 2; ++c) {
            for (uint32_t b = 0; b < 2; ++b)
                serve(bank_cursor_[c][b], deployment::bailu_bytes_per_tick, [&](const Job &j) {
                    return bailu(j) && !j.ingress_remaining && chip(j) == c && bank(j) == b;
                });
            for (uint32_t direction = 0; direction < 3; ++direction)
                serve_link(c, direction);
        }
    }
    void advance_tick() { ++tick_; }
    bool outstanding(uint32_t slot, uint64_t generation, bool control) const {
        for (const auto &j : jobs_)
            if (j.live && j.control == control && j.message.key.slot == slot && j.message.key.generation == generation)
                return true;
        return false;
    }
    bool idle() const {
        return std::none_of(
                   jobs_.begin(), jobs_.end(),
                   [](const auto &j) {
                       return j.live;
                   }
               ) &&
               returns_.empty() && std::all_of(payload_free_.begin(), payload_free_.end(), [](bool value) {
                   return value;
               });
    }
    std::string pending() const {
        std::ostringstream out;
        for (const auto &j : jobs_)
            if (j.live)
                out << " job(slot=" << j.message.key.slot << ",pos=" << j.message.key.position
                    << ",kind=" << j.message.kind << ",path=" << static_cast<uint32_t>(j.path)
                    << ",stage=" << static_cast<uint32_t>(j.stage) << ",remaining=" << j.remaining << ")";
        return out.str();
    }
    void report(const std::filesystem::path &directory) const {
        std::ofstream out(directory / "transport_events.jsonl");
        for (const auto &e : resources_)
            out << "{\"tick\":" << e.tick << ",\"event\":\"" << e.event << "\",\"transfer_id\":" << e.transfer
                << ",\"path\":" << static_cast<uint32_t>(e.path) << ",\"endpoint\":" << e.endpoint
                << ",\"slot\":" << e.key.slot << ",\"generation\":" << e.key.generation
                << ",\"position\":" << e.key.position << ",\"layer\":" << e.key.layer << ",\"kind\":" << e.key.phase
                << ",\"bytes\":" << e.bytes << ",\"bank\":" << e.bank << "}\n";
    }
};

class Fabric {
    std::shared_ptr<Engine> engine_;
    bool control_ = false;

public:
    Fabric(wp::Memory &memory, uint32_t data_credit, uint32_t data_queue, std::string scenario) :
        engine_(
            std::make_shared<Engine>(
                memory, data_credit, data_queue, deployment::quota_control_inbox, deployment::quota_control_queue,
                std::move(scenario)
            )
        ) {}
    Fabric(const Fabric &other, bool control) :
        engine_(other.engine_),
        control_(control) {}
    Engine &engine() { return *engine_; }
    const Engine &engine() const { return *engine_; }
    bool send(wp::Message message) { return engine_->send(message, control_); }
    bool send_direct(wp::Message message) { return engine_->send(message, control_, true); }
    bool has_message(uint32_t core) const { return engine_->has_message(core, control_); }
    bool has_host_message(uint32_t core) const { return engine_->has_message(core, control_, true); }
    wp::Message receive(uint32_t core) { return engine_->receive(core, control_); }
    wp::Message receive_host(uint32_t core) { return engine_->receive(core, control_, true); }
    bool outstanding(uint32_t slot, uint64_t generation) const {
        return engine_->outstanding(slot, generation, control_);
    }
    bool idle() const { return engine_->idle(); }
    uint64_t tick() const { return engine_->tick(); }
    uint64_t credit_stalls() const { return engine_->credit_stalls(control_); }
    uint64_t bytes_transferred() const { return engine_->bytes_transferred(); }
    uint32_t endpoint_count() const { return 8; }
    const std::vector<wp::TraceEvent> &events() const { return engine_->events(control_); }
    void record(const std::string &name, uint32_t actor, wp::Identity key = {}, uint64_t value = 0) {
        engine_->observe(control_, name, actor, key, value);
    }
    void deliver_due() { engine_->deliver_due(); }
    void step_batcher(uint32_t ep) { engine_->step_batcher(ep); }
    void advance_tick() { engine_->advance_tick(); }
};
}  // namespace hw
