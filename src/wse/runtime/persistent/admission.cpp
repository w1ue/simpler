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
#include "decode.h"

#include <stdexcept>

namespace wse::persistent {
std::vector<uint64_t> DecodeSimulation::request_budget(const RequestSpec &spec) const {
    const auto aligned = [](uint64_t bytes) {
        return (bytes + wse::SRAM_ALIGNMENT - 1) / wse::SRAM_ALIGNMENT * wse::SRAM_ALIGNMENT;
    };
    const auto packed = [](uint64_t width) {
        return width / 2 + width / 4 * 2;
    };
    const auto pages = [&](uint64_t length) {
        const auto count = [&](uint64_t rows) {
            return (rows + config_.page_rows - 1) / config_.page_rows;
        };
        return 3 * count(length / 2) + count(length);
    };
    const auto limit = spec.context_limit ? spec.context_limit : config_.max_context;
    const auto max_pages = pages(limit);
    const auto kv_page = aligned(config_.page_rows * packed(config_.latent));
    const auto key_page = aligned(config_.page_rows * packed(config_.index_dim));
    const auto hidden = aligned(uint64_t(config_.hidden) * 8);
    const auto lookup = aligned(24 * 8);
    const auto windows = config_.layers * aligned(uint64_t(config_.window) * config_.latent * 8);
    const auto states = 3 * aligned(uint64_t(8) * 2 * config_.latent * 8);
    std::vector<uint64_t> budget(memory_.regions().size());
    const auto core = home(spec.home_chip);
    // All potential tails can coexist with Bailu copies; keys stay in SRAM.
    budget[sram_[core]] = aligned(4096 + uint64_t(256) * config_.hidden) + windows + states +
                          max_pages * (kv_page + key_page) + 2 * kv_page + 14 * hidden +
                          config_.chips * (lookup + hidden);
    budget[bailu_[spec.home_chip]] = max_pages * kv_page;
    budget[host_[spec.home_chip]] = windows + states + pages(spec.prefill.tokens.size()) * (kv_page + key_page);
    for (uint32_t expert = 0; expert < config_.experts; ++expert)
        budget[sram_[expert_owner(expert)]] += 2 * hidden;
    budget[sram_[core + 5]] += 2 * hidden;
    for (uint32_t chip = 0; chip < config_.chips; ++chip)
        budget[sram_[home(chip) + 6]] += lookup + hidden;
    return budget;
}

bool DecodeSimulation::submit_request(RequestSpec request) {
    if (input_closed_ || drained_) throw std::runtime_error("request input is closed");
    if (request.name.empty() || !request.decode_inputs || request.home_chip >= config_.chips ||
        (request.slot != automatic_slot && request.slot >= config_.max_slots) ||
        uint64_t(request.prefill.tokens.size()) + request.decode_inputs > config_.max_context)
        throw std::invalid_argument("invalid request or maximum context exceeded");
    const auto limit = request.context_limit ? request.context_limit : config_.max_context;
    if (limit > config_.max_context || uint64_t(request.prefill.tokens.size()) + request.decode_inputs > limit)
        throw std::invalid_argument("request exceeds promised context limit");
    request.prefill.validate(config_);
    for (const auto &existing : specifications_)
        if (existing.name == request.name) throw std::invalid_argument("duplicate request name");
    const auto budget = request_budget(request);
    for (size_t i = 0; i < budget.size(); ++i)
        if (budget[i] > memory_.regions()[i].capacity - baseline_used_[i])
            throw std::invalid_argument("request maximum context cannot fit device capacity");
    if (specifications_.size() - requests_.size() >= config_.max_pending) return false;
    specifications_.push_back(std::move(request));
    return true;
}

void DecodeSimulation::add_request(RequestSpec request) {
    if (running_) throw std::runtime_error("use submit_request between service steps");
    if (!submit_request(std::move(request))) throw std::runtime_error("request ingress queue is full");
}
}  // namespace wse::persistent
