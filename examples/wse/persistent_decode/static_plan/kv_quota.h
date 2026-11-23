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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace kvquota {
namespace wp = wse::persistent;
constexpr uint32_t Slots = 4, Chips = 2, MaxPages = 32, TablePages = 16, ControlBytes = 256;
constexpr uint64_t Epoch = 1;
enum Kind : uint32_t { Low = 100, Grant = 101, Ack = 102, Denied = 103 };
struct Packet {
    uint64_t epoch = Epoch, generation = 0, sequence = 0, event_sequence = 0;
    uint32_t slot = 0, pool = 0, granted = 0, claimed = 0, kind = 0;
    std::array<uint32_t, TablePages> pages{};
};
struct DeviceState {
    uint64_t generation = 0, sequence = 0, event_sequence = 0;
    uint32_t granted = 0, claimed = 0, paused = 0, notified = 0, denied = 0, ending = 0, capacity_stop = 0;
    std::array<uint32_t, TablePages> pages{};
};
static_assert(std::is_trivially_copyable_v<Packet> && std::is_standard_layout_v<Packet>);
static_assert(std::is_trivially_copyable_v<DeviceState> && std::is_standard_layout_v<DeviceState>);
static_assert(sizeof(Packet) <= ControlBytes && sizeof(DeviceState) <= ControlBytes);

inline void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename T>
wp::Bytes bytes(const T &value, uint32_t size = ControlBytes) {
    require(sizeof(T) <= size, "quota packet exceeds fixed buffer");
    wp::Bytes result(size);
    std::memcpy(result.data(), &value, sizeof(T));
    return result;
}
template <typename T>
T value(const wp::Bytes &data) {
    require(data.size() >= sizeof(T), "short quota packet");
    T result;
    std::memcpy(&result, data.data(), sizeof(T));
    return result;
}

class Controller {
    struct Page {
        wp::Handle handle = 0;
        int32_t slot = -1;
        uint64_t generation = 0;
    };
    struct Buffers {
        wp::Handle state = 0, initial = 0;
        std::array<wp::Handle, 2> low_tx{}, low_rx{}, grant_tx{}, grant_rx{}, ack_tx{}, ack_rx{};
    };
    struct Host {
        uint64_t generation = 0, sequence = 0, acked = 0, last_event = 0;
        uint32_t count = 0;
        std::array<uint32_t, TablePages> pages{};
        bool active = false, initial = false, wants = false, pending_send = false, awaiting_ack = false;
        bool waiting = false, host_gate_reported = false;
        Packet outgoing{}, last_grant{}, first_grant{};
    };
    struct Endpoint {
        std::array<bool, 2> low_ready{}, grant_ready{}, ack_ready{};
        bool low_pending = false, ack_pending = false, install_gate_reported = false;
        Packet low{}, ack{};
    };
    struct Trace {
        uint64_t tick, generation, packet_generation, packet_epoch, sequence;
        std::string event;
        uint32_t slot, granted, claimed, host_owned, free_pages, physical;
    };
    wp::Memory &memory_;
    wp::Fabric control_;
    std::array<std::array<Page, MaxPages>, Chips> pool_{};
    std::array<DeviceState, Slots> device_{};
    std::array<Host, Slots> host_{};
    std::array<Endpoint, Slots> endpoint_{};
    std::array<Buffers, Slots> buffers_{};
    std::array<uint64_t, Slots> layers_{};
    std::array<bool, Slots> finished_{};
    std::vector<Trace> trace_;
    struct Outcome {
        uint32_t slot;
        uint64_t generation;
        bool capacity_stop;
    };
    std::vector<Outcome> outcomes_;
    std::string scenario_;
    bool injected_low_ = false, injected_grant_ = false, stale_low_ = false, stale_grant_ = false;
    bool host_gate_released_ = false, install_gate_released_ = false;
    uint64_t low_sent_ = 0, installed_ = 0, reserved_ = 0, pauses_ = 0, waits_ = 0, denials_ = 0;
    uint64_t duplicate_low_ = 0, duplicate_grant_ = 0, stale_low_count_ = 0, stale_grant_count_ = 0;
    uint64_t claims_ = 0, released_ = 0, control_stalls_ = 0;

    uint32_t home(uint32_t slot) const { return deployment::home_cores[slot]; }
    uint32_t chip(uint32_t slot) const { return home(slot) / 8; }
    uint32_t free_pages(uint32_t c) const {
        uint32_t count = 0;
        for (uint32_t i = 0; i < deployment::quota_pool_pages; ++i)
            count += pool_[c][i].slot < 0;
        return count;
    }
    wp::Identity key(const Packet &p, uint32_t lane) const { return {p.slot, p.generation, 0, 0, p.kind, lane}; }
    void record(const char *event, uint32_t slot, const Packet *p = nullptr, uint32_t physical = UINT32_MAX) {
        const auto &d = device_[slot];
        trace_.push_back(
            {control_.tick(), d.generation, p ? p->generation : d.generation, p ? p->epoch : Epoch,
             p ? p->sequence : d.sequence, event, slot, d.granted, d.claimed, host_[slot].count, free_pages(chip(slot)),
             physical}
        );
    }
    void save(uint32_t slot) { memory_.write_local(home(slot), buffers_[slot].state, bytes(device_[slot])); }
    bool send_packet(Packet p, uint32_t lane) {
        auto &b = buffers_[p.slot];
        wp::Handle source = 0, destination = 0;
        auto &e = endpoint_[p.slot];
        if (p.kind == Low) {
            source = b.low_tx[lane];
            destination = b.low_rx[lane];
            if (e.low_ready[lane]) return false;
        } else if (p.kind == Ack) {
            source = b.ack_tx[lane];
            destination = b.ack_rx[lane];
            if (e.ack_ready[lane]) return false;
        } else {
            source = b.grant_tx[lane];
            destination = b.grant_rx[lane];
            if (e.grant_ready[lane]) return false;
        }
        if (memory_.in_use(source) || memory_.in_use(destination)) return false;
        if (p.kind == Low || p.kind == Ack) memory_.write_local(home(p.slot), source, bytes(p));
        else memory_.stage_host(source, bytes(p));
        if (!control_.send({key(p, lane), home(p.slot), home(p.slot), 0, p.kind, source, destination})) {
            ++control_stalls_;
            return false;
        }
        return true;
    }
    void request_low(uint32_t slot) {
        auto &d = device_[slot];
        auto &e = endpoint_[slot];
        if (d.notified || d.ending || d.denied) return;
        d.notified = 1;
        e.low = {};
        e.low.slot = slot;
        e.low.pool = chip(slot);
        e.low.kind = Low;
        e.low.generation = d.generation;
        e.low.sequence = d.sequence;
        e.low.event_sequence = ++d.event_sequence;
        e.low.granted = d.granted;
        e.low.claimed = d.claimed;
        e.low_pending = true;
    }
    bool reserve(uint32_t slot, uint32_t target) {
        auto &h = host_[slot];
        require(target >= h.count && target <= TablePages, "quota reservation exceeds compiled page table");
        if (target - h.count > free_pages(chip(slot))) return false;
        for (uint32_t page = 0; h.count < target && page < deployment::quota_pool_pages; ++page) {
            auto &entry = pool_[chip(slot)][page];
            if (entry.slot >= 0) continue;
            entry.slot = static_cast<int32_t>(slot);
            entry.generation = h.generation;
            h.pages[h.count++] = page;
        }
        return true;
    }
    void stage_grant(uint32_t slot, bool denied) {
        auto &h = host_[slot];
        h.outgoing = {};
        h.outgoing.kind = denied ? Denied : Grant;
        h.outgoing.slot = slot;
        h.outgoing.pool = chip(slot);
        h.outgoing.generation = h.generation;
        h.outgoing.sequence = ++h.sequence;
        h.outgoing.granted = h.count;
        h.outgoing.pages = h.pages;
        h.pending_send = true;
        h.wants = false;
        h.waiting = false;
        if (denied) {
            ++denials_;
            record("KV_GRANT_DENIED", slot, &h.outgoing);
        } else {
            ++reserved_;
            record("KV_GRANT_RESERVED", slot, &h.outgoing);
        }
    }
    void process_low(uint32_t slot, uint32_t lane) {
        auto &h = host_[slot];
        const auto p = value<Packet>(memory_.inspect_after_drain(buffers_[slot].low_rx[lane]));
        endpoint_[slot].low_ready[lane] = false;
        if (p.epoch != Epoch || p.generation != h.generation || !h.active) {
            ++stale_low_count_;
            record("KV_STALE_EVENT_DROPPED", slot, &p);
            return;
        }
        require(p.slot == slot && p.kind == Low && p.sequence <= h.sequence, "low-water routing or sequence invalid");
        require(p.pool == chip(slot) && p.claimed <= p.granted, "invalid low-water counters");
        if (p.event_sequence <= h.last_event || p.sequence < h.acked) {
            ++duplicate_low_;
            record("KV_LOW_DUPLICATE_DROPPED", slot, &p);
            return;
        }
        h.last_event = p.event_sequence;
        if (!device_[slot].ending) h.wants = true;
        record("KV_LOW_HOST_RECEIVED", slot, &p);
    }
    void process_ack(uint32_t slot, uint32_t lane) {
        auto &h = host_[slot];
        const auto p = value<Packet>(memory_.inspect_after_drain(buffers_[slot].ack_rx[lane]));
        endpoint_[slot].ack_ready[lane] = false;
        if (p.epoch != Epoch || p.generation != h.generation || !h.active) return;
        require(p.slot == slot && p.kind == Ack && p.pool == chip(slot), "quota ACK identity invalid");
        require(p.sequence <= h.sequence, "ack refers to unissued grant");
        if (p.sequence == h.sequence && h.awaiting_ack) {
            h.awaiting_ack = false;
            h.acked = p.sequence;
            record("KV_GRANT_ACKED", slot, &p);
        }
    }
    void install(uint32_t slot, uint32_t lane) {
        auto &d = device_[slot];
        auto &e = endpoint_[slot];
        const auto p = value<Packet>(memory_.read_local(home(slot), buffers_[slot].grant_rx[lane]));
        if (p.epoch != Epoch || p.generation != d.generation) {
            e.grant_ready[lane] = false;
            ++stale_grant_count_;
            record("KV_STALE_GRANT_DROPPED", slot, &p);
            return;
        }
        if (scenario_ == "kv_delayed_install" && slot == 0 && p.sequence > 1 && !install_gate_released_) {
            if (!e.install_gate_reported) {
                e.install_gate_reported = true;
                record("KV_INSTALL_GATE_CLOSED", slot, &p);
            }
            if (!d.paused || layers_[2] == 0) return;
            install_gate_released_ = true;
            record("KV_INSTALL_GATE_RELEASED", slot, &p);
        }
        e.grant_ready[lane] = false;
        require(p.slot == slot && (p.kind == Grant || p.kind == Denied), "quota grant identity invalid");
        require(p.pool == chip(slot) && p.granted <= TablePages, "grant exceeds compiled page table");
        if (p.sequence < d.sequence) {
            ++stale_grant_count_;
            record("KV_OLD_GRANT_DROPPED", slot, &p);
            return;
        }
        if (p.sequence == d.sequence) {
            require(p.granted == d.granted && (p.kind == Denied || p.pages == d.pages), "conflicting duplicate grant");
            ++duplicate_grant_;
            record("KV_GRANT_DUPLICATE_DROPPED", slot, &p);
        } else {
            require(p.sequence == d.sequence + 1 && p.granted >= d.granted, "out-of-order grant sequence");
            if (p.kind == Grant) {
                for (uint32_t i = 0; i < p.granted; ++i) {
                    require(p.pages[i] < deployment::quota_pool_pages, "grant page outside physical pool");
                    if (i < d.granted) require(p.pages[i] == d.pages[i], "grant changes live page prefix");
                    for (uint32_t j = 0; j < i; ++j)
                        require(p.pages[i] != p.pages[j], "duplicate physical page in grant");
                }
                d.pages = p.pages;
                d.granted = p.granted;
                d.notified = 0;
                e.low_pending = false;
                if (d.claimed == 0) d.claimed = 1;
                ++installed_;
                record("KV_GRANT_INSTALLED", slot, &p);
            } else {
                d.denied = 1;
                record("KV_DENIAL_INSTALLED", slot, &p);
            }
            d.sequence = p.sequence;
        }
        e.ack = p;
        e.ack.kind = Ack;
        e.ack_pending = true;
    }
    void inject() {
        auto &e = endpoint_[0];
        auto &h = host_[0];
        if (!injected_grant_ && device_[0].granted &&
            (scenario_ == "kv_invalid_grant_bound" || scenario_ == "kv_conflicting_grant")) {
            Packet bad = h.last_grant;
            bad.kind = Grant;
            if (scenario_ == "kv_invalid_grant_bound") {
                bad.sequence = device_[0].sequence + 1;
                bad.granted = 17;
            } else {
                bad.sequence = device_[0].sequence;
                bad.granted = device_[0].granted + 1;
            }
            if (send_packet(bad, 1)) injected_grant_ = true;
        }
        if (scenario_ == "kv_duplicate_event" && !injected_low_ && e.low.event_sequence && !e.low_pending &&
            send_packet(e.low, 1)) {
            injected_low_ = true;
            record("KV_DUPLICATE_EVENT_INJECTED", 0, &e.low);
        }
        if (scenario_ == "kv_duplicate_grant" && !injected_grant_ && h.last_grant.sequence > 1 &&
            send_packet(h.last_grant, 1)) {
            injected_grant_ = true;
            record("KV_DUPLICATE_GRANT_INJECTED", 0, &h.last_grant);
        }
        if (scenario_ == "kv_old_grant" && !injected_grant_ && device_[0].sequence > 1 &&
            send_packet(h.first_grant, 1)) {
            injected_grant_ = true;
            record("KV_OLD_GRANT_INJECTED", 0, &h.first_grant);
        }
        bool stale_case = scenario_ == "kv_stale_generation" || scenario_ == "kv_stale_epoch";
        const auto expected_generation = scenario_ == "kv_stale_generation" ? 2u : 1u;
        if (stale_case && device_[0].generation == expected_generation && device_[0].granted && h.last_grant.sequence) {
            Packet p = h.last_grant;
            if (scenario_ == "kv_stale_generation") --p.generation;
            else p.epoch = 0;
            if (!stale_low_) {
                p.kind = Low;
                p.event_sequence = 999;
                if (send_packet(p, 1)) {
                    stale_low_ = true;
                    record("KV_STALE_EVENT_INJECTED", 0, &p);
                }
            } else if (!stale_grant_) {
                p.kind = Grant;
                if (send_packet(p, 1)) {
                    stale_grant_ = true;
                    record("KV_STALE_GRANT_INJECTED", 0, &p);
                }
            }
        }
    }

public:
    template <typename Allocate>
    Controller(
        wp::Memory &memory, const std::array<uint32_t, 16> &sram, const std::array<uint32_t, 2> &bailu,
        const std::array<uint32_t, 2> &host, std::string scenario, Allocate allocate
    ) :
        memory_(memory),
        control_(memory, 16, 8, deployment::quota_control_inbox, deployment::quota_control_queue),
        scenario_(std::move(scenario)) {
        for (uint32_t chip = 0; chip < Chips; ++chip)
            for (uint32_t page = 0; page < deployment::quota_pool_pages; ++page)
                pool_[chip][page].handle = allocate(bailu[chip], 128);
        for (uint32_t slot = 0; slot < Slots; ++slot) {
            auto &b = buffers_[slot];
            auto c = home(slot);
            b.state = allocate(sram[c], ControlBytes);
            b.initial = allocate(host[c / 8], 128);
            for (uint32_t lane = 0; lane < 2; ++lane) {
                b.low_tx[lane] = allocate(sram[c], ControlBytes);
                b.low_rx[lane] = allocate(host[c / 8], ControlBytes);
                b.grant_tx[lane] = allocate(host[c / 8], ControlBytes);
                b.grant_rx[lane] = allocate(sram[c], ControlBytes);
                b.ack_tx[lane] = allocate(sram[c], ControlBytes);
                b.ack_rx[lane] = allocate(host[c / 8], ControlBytes);
            }
        }
    }
    void bind(uint32_t slot, uint64_t generation) {
        require(!host_[slot].active, "quota slot rebound before lease return");
        device_[slot] = {};
        device_[slot].generation = generation;
        host_[slot] = {};
        host_[slot].generation = generation;
        host_[slot].active = true;
        host_[slot].initial = true;
        endpoint_[slot] = {};
        finished_[slot] = false;
        layers_[slot] = 0;
        memory_.stage_host(buffers_[slot].initial, bytes(uint64_t{5}, 128));
        save(slot);
        record("KV_BIND", slot);
    }
    wp::Handle initial_source(uint32_t slot) const { return buffers_[slot].initial; }
    wp::Handle page(uint32_t slot, uint32_t logical) const {
        const auto &d = device_[slot];
        require(logical < d.claimed && logical < d.granted, "KV access outside installed grant");
        return pool_[chip(slot)][d.pages[logical]].handle;
    }
    bool ready(uint32_t slot) const { return device_[slot].granted != 0; }
    bool allow_data(uint32_t slot) const {
        bool delayed = scenario_ == "kv_delayed_host" || scenario_ == "kv_delayed_install";
        return !(delayed && slot == 2 && !device_[0].paused && layers_[2] == 0);
    }
    bool claim(uint32_t slot, uint32_t position, bool more_inputs) {
        auto &d = device_[slot];
        require(position < TablePages, "claim exceeds compiled page table");
        if (position < d.claimed) return true;
        require(position == d.claimed, "nonsequential KV page claim");
        if (d.claimed == d.granted) {
            if (!d.paused) {
                d.paused = 1;
                ++pauses_;
                record("KV_HARD_PAUSE", slot);
            }
            request_low(slot);
            save(slot);
            return false;
        }
        require(d.claimed < d.granted, "claim would exceed installed grant");
        if (d.paused) {
            d.paused = 0;
            record("KV_RESUMED", slot);
        }
        const auto physical = d.pages[d.claimed++];
        ++claims_;
        record("KV_PAGE_CLAIMED", slot, nullptr, physical);
        if (more_inputs && d.granted - d.claimed <= deployment::quota_low_watermark) request_low(slot);
        save(slot);
        return true;
    }
    bool denied(uint32_t slot) const { return device_[slot].denied != 0; }
    void layer_done(uint32_t slot) { ++layers_[slot]; }
    void begin_retire(uint32_t slot, bool capacity_stop = false) {
        if (!device_[slot].ending) {
            device_[slot].ending = 1;
            device_[slot].capacity_stop = capacity_stop;
            host_[slot].wants = false;
            host_[slot].initial = false;
            endpoint_[slot].low_pending = false;
            record("KV_RETIRE_BEGIN", slot);
            save(slot);
        }
    }
    bool retire(uint32_t slot) {
        auto &h = host_[slot];
        auto &e = endpoint_[slot];
        if (control_.outstanding(slot, h.generation) || h.pending_send || h.awaiting_ack || e.ack_pending) return false;
        for (uint32_t lane = 0; lane < 2; ++lane)
            if (e.low_ready[lane] || e.grant_ready[lane] || e.ack_ready[lane]) return false;
        for (uint32_t i = 0; i < h.count; ++i) {
            auto &p = pool_[chip(slot)][h.pages[i]];
            require(
                p.slot == static_cast<int32_t>(slot) && p.generation == h.generation,
                "KV lease owner mismatch on retirement"
            );
            require(!memory_.in_use(p.handle), "KV lease returned before transfer drain");
            p.slot = -1;
            p.generation = 0;
            ++released_;
        }
        record("KV_PAGES_RETURNED", slot);
        outcomes_.push_back({slot, h.generation, device_[slot].capacity_stop != 0});
        h.count = 0;
        h.active = false;
        h.waiting = false;
        finished_[slot] = true;
        return true;
    }
    void receive() {
        for (uint32_t core = 0; core < 16; ++core) {
            if (!control_.has_message(core)) continue;
            const auto m = control_.receive(core);
            require(m.key.slot < Slots && m.key.lane < 2, "quota control address invalid");
            auto &e = endpoint_[m.key.slot];
            if (m.kind == Low) e.low_ready[m.key.lane] = true;
            else if (m.kind == Ack) e.ack_ready[m.key.lane] = true;
            else if (m.kind == Grant || m.kind == Denied) e.grant_ready[m.key.lane] = true;
            else throw std::runtime_error("unknown quota control message");
        }
    }
    void device_step() {
        for (uint32_t slot = 0; slot < Slots; ++slot) {
            auto &e = endpoint_[slot];
            for (uint32_t lane = 0; lane < 2; ++lane)
                if (e.grant_ready[lane]) install(slot, lane);
            if (e.low_pending && send_packet(e.low, 0)) {
                e.low_pending = false;
                ++low_sent_;
                record("KV_LOW_SENT", slot, &e.low);
            }
            if (e.ack_pending && send_packet(e.ack, 0)) {
                e.ack_pending = false;
                record("KV_GRANT_ACK_SENT", slot, &e.ack);
            }
            if (host_[slot].active) save(slot);
        }
    }
    void host_step() {
        for (uint32_t slot = 0; slot < Slots; ++slot) {
            auto &h = host_[slot];
            auto &d = device_[slot];
            auto &e = endpoint_[slot];
            for (uint32_t lane = 0; lane < 2; ++lane) {
                if (e.low_ready[lane]) process_low(slot, lane);
                if (e.ack_ready[lane]) process_ack(slot, lane);
            }
            if (!h.active) continue;
            if (!h.pending_send && !h.awaiting_ack && !d.ending && (h.initial || h.wants)) {
                bool held = scenario_ == "kv_delayed_host" && slot == 0 && !h.initial && !host_gate_released_;
                if (held) {
                    if (!h.host_gate_reported) {
                        h.host_gate_reported = true;
                        record("KV_HOST_GATE_CLOSED", slot);
                    }
                    if (d.paused && layers_[2] > 0) {
                        host_gate_released_ = true;
                        record("KV_HOST_GATE_RELEASED", slot);
                        held = false;
                    }
                }
                if (!held) {
                    const auto target = h.initial ?
                                            deployment::quota_initial_pages :
                                            std::min<uint32_t>(TablePages, h.count + deployment::quota_refill_pages);
                    if (reserve(slot, target)) {
                        h.initial = false;
                        stage_grant(slot, false);
                    } else if (deployment::quota_deny_on_exhaustion && !h.initial) stage_grant(slot, true);
                    else if (!h.waiting) {
                        h.waiting = true;
                        ++waits_;
                        record("KV_HOST_WAIT_POOL", slot);
                    }
                }
            }
            if (h.pending_send && send_packet(h.outgoing, 0)) {
                h.pending_send = false;
                h.awaiting_ack = true;
                h.last_grant = h.outgoing;
                if (h.outgoing.sequence == 1) h.first_grant = h.outgoing;
                record("KV_GRANT_SENT", slot, &h.outgoing);
            }
        }
        inject();
        audit();
    }
    void audit() const {
        for (uint32_t slot = 0; slot < Slots; ++slot) {
            const auto &d = device_[slot];
            require(d.claimed <= d.granted && d.granted <= TablePages, "quota counter invariant violated");
            if (!host_[slot].active) continue;
            require(d.granted <= host_[slot].count, "device counted unreserved Host pages");
            for (uint32_t i = 0; i < d.granted; ++i) {
                const auto &p = pool_[chip(slot)][d.pages[i]];
                require(
                    p.slot == static_cast<int32_t>(slot) && p.generation == d.generation,
                    "installed page belongs to another request"
                );
            }
        }
    }
    void deliver() { control_.deliver_due(); }
    void advance(bool reverse) {
        for (uint32_t ep = 0; ep < control_.endpoint_count(); ++ep)
            control_.step_batcher(reverse ? control_.endpoint_count() - 1 - ep : ep);
        control_.advance_tick();
    }
    bool idle() const {
        if (!control_.idle()) return false;
        for (const auto &h : host_)
            if (h.active || h.pending_send || h.awaiting_ack) return false;
        return true;
    }
    bool blocked_without_progress() const {
        if (!control_.idle()) return false;
        bool active = false;
        for (uint32_t slot = 0; slot < Slots; ++slot) {
            const auto &h = host_[slot];
            if (!h.active) continue;
            active = true;
            if (!device_[slot].paused || !h.waiting || h.pending_send || h.awaiting_ack || endpoint_[slot].low_pending)
                return false;
            for (uint32_t lane = 0; lane < 2; ++lane)
                if (endpoint_[slot].low_ready[lane] || endpoint_[slot].grant_ready[lane] ||
                    endpoint_[slot].ack_ready[lane])
                    return false;
        }
        return active;
    }
    void report(const std::filesystem::path &dir) const {
        std::ofstream out(dir / "kv_quota.json");
        out << "{\"low_sent\":" << low_sent_ << ",\"grants_reserved\":" << reserved_
            << ",\"grants_installed\":" << installed_ << ",\"hard_pauses\":" << pauses_ << ",\"pool_waits\":" << waits_
            << ",\"denials\":" << denials_ << ",\"duplicate_events_dropped\":" << duplicate_low_
            << ",\"duplicate_grants_dropped\":" << duplicate_grant_ << ",\"stale_events_dropped\":" << stale_low_count_
            << ",\"stale_grants_dropped\":" << stale_grant_count_ << ",\"claims\":" << claims_
            << ",\"pages_returned\":" << released_ << ",\"control_queue_stalls\":" << control_stalls_
            << ",\"control_credit_stalls\":" << control_.credit_stalls() << ",\"free_pages\":[" << free_pages(0) << ","
            << free_pages(1) << "],\"outcomes\":[";
        for (size_t i = 0; i < outcomes_.size(); ++i) {
            const auto &r = outcomes_[i];
            out << (i ? "," : "") << "{\"slot\":" << r.slot << ",\"generation\":" << r.generation << ",\"reason\":\""
                << (r.capacity_stop ? "capacity_denied" : "released") << "\"}";
        }
        out << "]}\n";
        std::ofstream events(dir / "kv_events.jsonl");
        for (const auto &t : trace_)
            events << "{\"tick\":" << t.tick << ",\"event\":\"" << t.event << "\",\"slot\":" << t.slot
                   << ",\"generation\":" << t.generation << ",\"packet_generation\":" << t.packet_generation
                   << ",\"packet_epoch\":" << t.packet_epoch << ",\"sequence\":" << t.sequence
                   << ",\"granted\":" << t.granted << ",\"claimed\":" << t.claimed << ",\"host_owned\":" << t.host_owned
                   << ",\"free_pages\":" << t.free_pages << ",\"physical\":" << t.physical << "}\n";
        std::ofstream transport(dir / "control_events.jsonl");
        for (const auto &e : control_.events())
            transport << "{\"tick\":" << e.tick << ",\"event\":\"" << e.event << "\",\"actor\":" << e.actor
                      << ",\"slot\":" << e.key.slot << ",\"generation\":" << e.key.generation
                      << ",\"kind\":" << e.key.phase << ",\"transfer_id\":" << e.transfer_id << "}\n";
    }
};
}  // namespace kvquota
