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

#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace wse::persistent {

struct DecodeConfig {
    uint32_t layers = 40;
    uint32_t hidden = 16;
    uint32_t latent = 8;
    uint32_t index_dim = 4;
    uint32_t heads = 2;
    uint32_t intermediate = 24;
    uint32_t experts = 8;
    uint32_t top_k = 6;
    uint32_t vocabulary = 32;
    uint32_t window = 8;
    uint32_t page_rows = 4;
    uint32_t index_top_k = 3;
    uint32_t max_context = 128;
    uint32_t chips = 2;
    uint32_t cores_per_chip = 8;
    uint32_t inbox_capacity = 2;
    bool reverse_batcher = false;
    bool cancel_long_request = false;
};

struct RequestSpec {
    std::string name;
    uint32_t slot;
    uint32_t home_chip;
    std::vector<uint32_t> prompt;
    uint32_t decode_inputs;
    std::string admit_after;
    uint32_t admit_after_outputs = 0;
};

struct Matrix {
    uint32_t rows;
    uint32_t cols;
    Handle handle;
    uint32_t owner;
};

struct LayerWeights {
    Matrix hc_attention, hc_ffn, q_a, q_b, kv, compressor, gate, index_key, index_query, output, router;
    Matrix shared_a, shared_b, shared_c;
    std::vector<std::array<Matrix, 3>> experts;
};

enum class DecodeOperation : uint32_t {
    Embed = 10,
    AttentionPrepare = 11,
    Index = 12,
    Gather = 13,
    Attention = 14,
    Router = 15,
    Dispatch = 16,
    Combine = 17,
    AdvanceLayer = 18,
    Sample = 19
};

class DecodeSimulation {
public:
    explicit DecodeSimulation(DecodeConfig config = {});
    ~DecodeSimulation();
    void set_core_program(Program program);
    void set_batcher_program(Program program);
    void set_service_program(Program program);
    void register_batcher_operation(uint32_t id, std::function<StepResult(const Message &, uint32_t)> operation);
    void register_core_operation(uint32_t id, std::function<void(Values &)> operation);
    void add_request(RequestSpec request);
    void run();
    void write_report(const std::filesystem::path &output) const;
    const Fabric &fabric() const { return fabric_; }
    const Memory &memory() const { return memory_; }
    uint32_t completed_requests() const;

private:
    struct Page;
    struct Request;
    struct ExpertWork;
    DecodeConfig config_;
    Memory memory_;
    Fabric fabric_;
    std::vector<uint32_t> sram_;
    std::vector<uint32_t> bailu_;
    std::vector<uint32_t> batcher_regions_;
    std::vector<std::vector<LayerWeights>> weights_;
    std::vector<Matrix> embedding_;
    std::vector<Matrix> lm_head_;
    std::vector<Matrix> engram_;
    std::vector<RequestSpec> specifications_;
    std::vector<std::unique_ptr<Request>> requests_;
    std::vector<std::deque<ExpertWork>> work_;
    std::map<uint32_t, uint64_t> generations_;
    std::map<uint32_t, std::function<void(Values &)>> custom_core_operations_;
    Program core_program_;
    Program service_program_;
    bool running_ = false;
    uint64_t launch_count_ = 0;
    uint64_t core_steps_ = 0;
    uint64_t history_fetches_ = 0;
    uint64_t kv_rows_ = 0;
    uint64_t request_reuses_ = 0;

    Matrix make_weight(uint32_t owner, uint32_t rows, uint32_t cols, uint32_t seed);
    Values multiply(uint32_t core, const Values &x, const Matrix &matrix) const;
    uint32_t home(uint32_t chip) const;
    uint32_t expert_owner(uint32_t expert) const;
    void deploy_weights();
    bool admit_requests();
    bool receive(uint32_t core);
    bool step_expert(uint32_t core);
    bool step_request(Request &request);
    bool collect_pages(Request &request);
    bool service_work_pending(uint32_t slot, uint64_t generation) const;
    void prepare_attention(Request &request);
    void select_history(Request &request);
    StepResult gather_history(Request &request);
    void compute_attention(Request &request);
    void route(Request &request);
    StepResult dispatch(Request &request);
    StepResult combine(Request &request);
    void embed(Request &request);
    void sample(Request &request);
    void append_compressed(Request &request, uint32_t source, const Values &latent);
    StepResult flush_pages(Request &request);
    void retire(Request &request);
    Request &find_request(const Identity &key);
};

Program default_core_program();
Bytes pack_fp4(const Values &values);
Values unpack_fp4(const Bytes &bytes, uint32_t dimensions);

}  // namespace wse::persistent
