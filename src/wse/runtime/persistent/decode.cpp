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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace wse::persistent {
namespace {
constexpr uint32_t kExpert = 1, kResult = 2, kLookup = 3, kLookupResult = 4, kShared = 5;
constexpr uint32_t kFetch = 6, kStore = 7, kImport = 8;
constexpr std::array<uint32_t, 4> kSources{2, 8, 14, 20};
constexpr std::array<double, 8> kFp4{0, 0.5, 1, 1.5, 2, 3, 4, 6};

Values normalize(Values x) {
    double sum = 0;
    for (const auto value : x)
        sum += value * value;
    const auto inv = 1.0 / std::sqrt(sum / x.size() + 1e-6);
    for (auto &value : x)
        value *= inv;
    return x;
}
Values softmax(Values x) {
    const double maximum = *std::max_element(x.begin(), x.end());
    double sum = 0;
    for (auto &value : x) {
        value = std::exp(value - maximum);
        sum += value;
    }
    for (auto &value : x)
        value /= sum;
    return x;
}
double sigmoid(double value) { return 1.0 / (1.0 + std::exp(-value)); }
void rope(Values &x, uint32_t position, double theta = 10000.0) {
    const auto width = std::min<size_t>(4, x.size());
    for (size_t i = 0; i < width / 2; ++i) {
        const auto angle = position / std::pow(theta, 2.0 * i / width);
        const auto a = x[i], b = x[i + width / 2];
        x[i] = a * std::cos(angle) - b * std::sin(angle);
        x[i + width / 2] = a * std::sin(angle) + b * std::cos(angle);
    }
}
struct Mix {
    Values pre, post, residual;
};
Mix mixes(const Values &projected, const Values &streams) {
    double square = 0;
    for (auto value : streams)
        square += value * value;
    const auto inv = 1.0 / std::sqrt(square / streams.size() + 1e-6);
    Mix mix{Values(4), Values(4), Values(16)};
    for (uint32_t i = 0; i < 4; ++i) {
        mix.pre[i] = sigmoid(projected[i] * inv) + 1e-6;
        mix.post[i] = 2 * sigmoid(projected[4 + i] * inv);
        Values row(4);
        for (uint32_t j = 0; j < 4; ++j)
            row[j] = projected[8 + 4 * i + j] * inv;
        row = softmax(std::move(row));
        for (uint32_t j = 0; j < 4; ++j)
            mix.residual[4 * i + j] = row[j] + 1e-6;
    }
    for (uint32_t iteration = 0; iteration < 20; ++iteration) {
        if (iteration)
            for (uint32_t i = 0; i < 4; ++i) {
                double sum = 1e-6;
                for (uint32_t j = 0; j < 4; ++j)
                    sum += mix.residual[i * 4 + j];
                for (uint32_t j = 0; j < 4; ++j)
                    mix.residual[i * 4 + j] /= sum;
            }
        for (uint32_t j = 0; j < 4; ++j) {
            double sum = 1e-6;
            for (uint32_t i = 0; i < 4; ++i)
                sum += mix.residual[i * 4 + j];
            for (uint32_t i = 0; i < 4; ++i)
                mix.residual[i * 4 + j] /= sum;
        }
    }
    return mix;
}
Values collapse(const Values &streams, const Mix &mix, uint32_t hidden) {
    Values x(hidden);
    for (uint32_t i = 0; i < hidden; ++i)
        x[i] = (streams[i] * mix.pre[0] + streams[hidden + i] * mix.pre[1]) +
               (streams[2 * hidden + i] * mix.pre[2] + streams[3 * hidden + i] * mix.pre[3]);
    return x;
}
Values expand(const Values &update, const Values &streams, const Mix &mix, uint32_t hidden) {
    Values result(4 * hidden);
    for (uint32_t j = 0; j < 4; ++j)
        for (uint32_t k = 0; k < hidden; ++k) {
            double value = update[k] * mix.post[j];
            for (uint32_t i = 0; i < 4; ++i)
                value += mix.residual[i * 4 + j] * streams[i * hidden + k];
            result[j * hidden + k] = value;
        }
    return result;
}
uint32_t source_for_layer(uint32_t layer) {
    if (layer < 2) return 4;
    if (layer >= 20) return 3;
    if (layer >= 14) return 2;
    return layer >= 8 ? 1 : 0;
}
bool is_full(uint32_t layer) { return std::find(kSources.begin(), kSources.end(), layer) != kSources.end(); }
bool is_reindex(uint32_t layer) { return layer >= 24 && layer % 4 == 0; }
size_t packed_size(uint32_t dimensions) { return dimensions / 2 + dimensions / 4 * 2; }
Bytes slice(const Bytes &bytes, size_t offset, size_t count) {
    if (offset > bytes.size() || count > bytes.size() - offset)
        throw std::runtime_error("payload slice outside allocation");
    return Bytes(bytes.begin() + offset, bytes.begin() + offset + count);
}
void json_values(std::ostream &out, const Values &values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}
std::string json_string(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char character : value) {
        if (character == '"' || character == '\\') out << '\\' << character;
        else if (character < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(character);
        else out << character;
    }
    out << '"';
    return out.str();
}
template <typename T>
void json_integers(std::ostream &out, const std::vector<T> &values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}
}  // namespace

Bytes pack_fp4(const Values &values) {
    if (values.empty() || values.size() % 4) throw std::invalid_argument("FP4 fixture groups contain four elements");
    Bytes result(packed_size(values.size()));
    for (size_t group = 0; group < values.size() / 4; ++group) {
        double maximum = 0;
        for (size_t i = 0; i < 4; ++i) {
            if (!std::isfinite(values[group * 4 + i])) throw std::invalid_argument("nonfinite FP4 fixture input");
            maximum = std::max(maximum, std::abs(values[group * 4 + i]));
        }
        const int16_t exponent = maximum ? static_cast<int16_t>(std::ceil(std::log2(maximum / 6.0))) : 0;
        const auto scale = std::ldexp(1.0, exponent);
        std::memcpy(result.data() + values.size() / 2 + group * 2, &exponent, 2);
        for (size_t i = 0; i < 4; ++i) {
            const double value = std::abs(values[group * 4 + i]) / scale;
            uint8_t best = 0;
            for (uint8_t candidate = 1; candidate < 8; ++candidate) {
                const auto error = std::abs(value - kFp4[candidate]);
                const auto old = std::abs(value - kFp4[best]);
                if (error < old || (error == old && !(candidate & 1))) best = candidate;
            }
            if (std::signbit(values[group * 4 + i])) best |= 8;
            const auto index = group * 4 + i;
            result[index / 2] |= best << (4 * (index % 2));
        }
    }
    return result;
}
Values unpack_fp4(const Bytes &bytes, uint32_t dimensions) {
    if (!dimensions || dimensions % 4 || bytes.size() != packed_size(dimensions))
        throw std::runtime_error("invalid packed FP4 row");
    Values result(dimensions);
    for (uint32_t i = 0; i < dimensions; ++i) {
        int16_t exponent;
        std::memcpy(&exponent, bytes.data() + dimensions / 2 + (i / 4) * 2, 2);
        const uint8_t value = (bytes[i / 2] >> (4 * (i % 2))) & 15;
        result[i] = std::ldexp(kFp4[value & 7] * (value & 8 ? -1 : 1), exponent);
    }
    return result;
}

Program default_core_program() {
    Program program{"decode_40_layer", {{InstructionKind::Invoke, 20, 0}}};
    for (uint32_t op = 10; op <= 19; ++op)
        program.code.push_back({InstructionKind::Invoke, op, op == 18 ? 2u : 0u});
    program.code.push_back({InstructionKind::Halt, 0, 0});
    return program;
}

struct DecodeSimulation::Page {
    Handle tail = 0, external = 0, keys = 0;
    uint32_t valid = 0;
    bool flush_sent = false, flush_done = false;
};
struct DecodeSimulation::Request {
    RequestSpec spec;
    uint64_t generation;
    uint32_t home_core;
    uint32_t position = 0, layer = 0, input = 0;
    bool imported = false, finishing = false, retired = false, cancelled = false;
    struct ImportChunk {
        Handle source, destination;
        bool consumed = false;
        uint32_t rows = 0;
    };
    std::vector<ImportChunk> imports;
    uint32_t import_next = 0, import_done = 0;
    Continuation continuation;
    Handle frame = 0;
    std::vector<Handle> windows;
    std::array<Handle, 3> compressor_states{};
    std::array<std::vector<Page>, 4> pages;
    std::array<uint32_t, 4> rows{};
    Values streams, residual, x, q;
    Mix attention_mix, ffn_mix;
    std::vector<uint32_t> selected, candidate_pool, tokens;
    std::map<uint32_t, Values> gathered;
    std::vector<uint32_t> fetch_pages;
    uint32_t fetch_next = 0, fetch_live = 0;
    std::vector<Values> results;
    uint32_t result_bitmap = 0, dispatch_next = 0;
    std::vector<uint32_t> routes;
    Values route_weights;
    uint32_t lookup_next = 0, lookup_bitmap = 0;
    std::vector<Values> lookup_results;
    bool lookup_applied = false;
    std::vector<uint32_t> generated;
    std::vector<Values> logits;
    std::vector<double> layer_checksums;
    std::array<std::vector<Values>, 4> final_rows;
    std::array<std::vector<Values>, 4> final_keys;
    std::vector<std::vector<Values>> final_windows;
    std::vector<Values> final_states;
    uint32_t prefill_next_token = 0;

    Identity identity(uint32_t phase = 0, uint32_t lane = 0) const {
        return {spec.slot, generation, position, layer, phase, lane};
    }
};
struct DecodeSimulation::ExpertWork {
    Message incoming;
    Handle output = 0, destination = 0;
    bool computed = false;
    bool sent = false;
    Continuation continuation;
};

DecodeSimulation::DecodeSimulation(DecodeConfig config) :
    config_(config),
    fabric_(memory_, config.chips * config.cores_per_chip, config.cores_per_chip, config.inbox_capacity),
    work_(config.chips * config.cores_per_chip),
    core_program_(default_core_program()),
    service_program_(
        {"expert_lookup_service",
         {{InstructionKind::Invoke, 30, 0}, {InstructionKind::Invoke, 31, 0}, {InstructionKind::Halt, 0, 0}}}
    ) {
    if (config.layers != 40 || config.chips != 2 || config.cores_per_chip != 8 || config.experts != 8 ||
        config.top_k != 6 || config.hidden < 4 || !config.latent || config.latent % 4 || !config.index_dim ||
        config.index_dim % 4 || !config.heads || !config.intermediate || !config.index_top_k || !config.page_rows ||
        config.vocabulary < 2 || !config.window || config.max_context < 1)
        throw std::invalid_argument("unsupported scaled fixture contract");
    for (uint32_t core = 0; core < config.chips * config.cores_per_chip; ++core)
        sram_.push_back(memory_.add_region(Tier::Sram, core, 16 * 1024 * 1024));
    for (uint32_t chip = 0; chip < config.chips; ++chip)
        bailu_.push_back(memory_.add_region(Tier::Bailu, home(chip), 64 * 1024 * 1024));
    for (uint32_t chip = 0; chip < config.chips; ++chip)
        host_.push_back(memory_.add_region(Tier::Host, home(chip), 128 * 1024 * 1024));
    for (uint32_t endpoint = 0; endpoint < fabric_.endpoint_count(); ++endpoint) {
        batcher_regions_.push_back(memory_.add_region(Tier::Batcher, endpoint, 64 * 1024));
        memory_.allocate(batcher_regions_.back(), 32 * 256);
    }
    deploy_weights();
}
DecodeSimulation::~DecodeSimulation() = default;
uint32_t DecodeSimulation::home(uint32_t chip) const { return chip * config_.cores_per_chip; }
uint32_t DecodeSimulation::expert_owner(uint32_t expert) const { return home(expert / 4) + 1 + expert % 4; }
Matrix DecodeSimulation::make_weight(uint32_t owner, uint32_t rows, uint32_t cols, uint32_t seed) {
    Values values(rows * cols);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = std::sin((seed + 1) * 0.137 + (i + 1) * 1.731) * 0.35 / std::sqrt(rows);
    auto handle = memory_.allocate(sram_.at(owner), values.size() * sizeof(double));
    memory_.load_before_launch(handle, encode_values(values));
    return {rows, cols, handle, owner};
}
Values DecodeSimulation::multiply(uint32_t core, const Values &x, const Matrix &matrix) const {
    if (x.size() != matrix.rows) throw std::runtime_error("matrix input width mismatch");
    const auto weight = decode_values(memory_.read_local(core, matrix.handle));
    Values result(matrix.cols);
    for (uint32_t k = 0; k < matrix.rows; ++k)
        for (uint32_t j = 0; j < matrix.cols; ++j)
            result[j] += x[k] * weight[k * matrix.cols + j];
    return result;
}
void DecodeSimulation::deploy_weights() {
    const auto d = config_.hidden, latent = config_.latent, index = config_.index_dim;
    for (uint32_t chip = 0; chip < config_.chips; ++chip) {
        const auto core = home(chip);
        embedding_.push_back(make_weight(core, config_.vocabulary, d, 1));
        lm_head_.push_back(make_weight(core, d, config_.vocabulary, 2));
        engram_.push_back(make_weight(core + 6, 64, d, 3 + chip));
        weights_.push_back({});
        for (uint32_t layer = 0; layer < config_.layers; ++layer) {
            LayerWeights weights;
            auto make = [&](uint32_t rows, uint32_t cols, uint32_t offset) {
                return make_weight(core, rows, cols, 100 + layer * 40 + offset);
            };
            weights.hc_attention = make(4 * d, 24, 0);
            weights.hc_ffn = make(4 * d, 24, 1);
            weights.q_a = make(d, latent, 2);
            weights.q_b = make(latent, config_.heads * latent, 3);
            weights.kv = make(d, latent, 4);
            weights.compressor = make(d, latent, 5);
            weights.gate = make(d, latent, 6);
            weights.index_key = make(latent, index, 7);
            weights.index_query = make(d, index, 8);
            weights.output = make(config_.heads * latent, d, 9);
            weights.router = make(d, config_.experts, 10);
            const auto shared = core + 5;
            weights.shared_a = make_weight(shared, d, config_.intermediate, 100 + layer * 40 + 11);
            weights.shared_b = make_weight(shared, d, config_.intermediate, 100 + layer * 40 + 12);
            weights.shared_c = make_weight(shared, config_.intermediate, d, 100 + layer * 40 + 13);
            for (uint32_t expert = 0; expert < config_.experts; ++expert) {
                std::array<Matrix, 3> matrices{};
                if (expert / 4 == chip) {
                    const auto owner = expert_owner(expert);
                    matrices = {
                        make_weight(owner, d, config_.intermediate, 10000 + layer * 100 + expert * 3),
                        make_weight(owner, d, config_.intermediate, 10001 + layer * 100 + expert * 3),
                        make_weight(owner, config_.intermediate, d, 10002 + layer * 100 + expert * 3)
                    };
                }
                weights.experts.push_back(matrices);
            }
            weights_.back().push_back(std::move(weights));
        }
    }
}
void DecodeSimulation::set_core_program(Program program) {
    if (running_) throw std::runtime_error("cannot replace live AICore program");
    program.validate();
    core_program_ = std::move(program);
}
void DecodeSimulation::set_batcher_program(Program program) {
    if (running_) throw std::runtime_error("cannot replace live batcher program");
    fabric_.set_program(std::move(program));
}
void DecodeSimulation::set_service_program(Program program) {
    if (running_) throw std::runtime_error("cannot replace live service program");
    program.validate();
    service_program_ = std::move(program);
}
void DecodeSimulation::register_batcher_operation(
    uint32_t id, std::function<StepResult(const Message &, uint32_t)> operation
) {
    if (running_) throw std::runtime_error("cannot replace live batcher operations");
    fabric_.register_operation(id, std::move(operation));
}
void DecodeSimulation::register_core_operation(uint32_t id, std::function<void(Values &)> operation) {
    if (running_ || id < 100 || !operation || !custom_core_operations_.emplace(id, std::move(operation)).second)
        throw std::invalid_argument("invalid custom AICore operation");
}
void DecodeSimulation::add_request(RequestSpec request) {
    if (running_ || request.name.empty() || !request.decode_inputs || request.home_chip >= config_.chips ||
        request.prefill.tokens.size() + request.decode_inputs > config_.max_context)
        throw std::invalid_argument("invalid request or maximum context exceeded");
    request.prefill.validate(config_);
    for (const auto &existing : specifications_)
        if (existing.name == request.name) throw std::invalid_argument("duplicate request name");
    auto images = prepare_import(request);
    specifications_.push_back(std::move(request));
    import_images_.push_back(std::move(images));
}
std::vector<DecodeSimulation::ImportImage> DecodeSimulation::prepare_import(const RequestSpec &spec) {
    std::vector<ImportImage> images;
    const auto &state = spec.prefill;
    auto add = [&](uint32_t type, uint32_t family, uint32_t index, uint32_t valid, const Bytes &bytes) {
        auto source = memory_.allocate(host_[spec.home_chip], bytes.size());
        images.push_back({source, type, family, index, valid});
        memory_.load_before_launch(source, bytes);
    };
    try {
        for (uint32_t family = 0; family < 4; ++family) {
            for (uint32_t first = 0; first < state.kv_rows[family].size(); first += config_.page_rows) {
                const auto valid = std::min<size_t>(config_.page_rows, state.kv_rows[family].size() - first);
                Bytes kv(config_.page_rows * packed_size(config_.latent));
                Bytes keys(config_.page_rows * packed_size(config_.index_dim));
                for (uint32_t row = 0; row < valid; ++row) {
                    const auto &value = state.kv_rows[family][first + row];
                    const auto &key = state.key_rows[family][first + row];
                    std::copy(value.begin(), value.end(), kv.begin() + row * value.size());
                    std::copy(key.begin(), key.end(), keys.begin() + row * key.size());
                }
                add(0, family, first / config_.page_rows, valid, kv);
                add(1, family, first / config_.page_rows, valid, keys);
            }
        }
        for (uint32_t layer = 0; layer < config_.layers; ++layer) {
            Values ring(config_.window * config_.latent);
            const auto rows = state.windows[layer].size() / config_.latent;
            for (size_t row = 0; row < rows; ++row) {
                const auto position = state.tokens.size() - rows + row;
                std::copy_n(
                    state.windows[layer].begin() + row * config_.latent, config_.latent,
                    ring.begin() + (position % config_.window) * config_.latent
                );
            }
            add(2, layer, 0, 0, encode_values(ring));
        }
        for (uint32_t family = 0; family < 3; ++family)
            add(3, family, 0, 0, encode_values(state.compressor_states[family]));
    } catch (...) {
        for (const auto &image : images)
            memory_.release(image.source);
        throw;
    }
    return images;
}
void DecodeSimulation::bind_import(Request &request, size_t specification_index) {
    const auto &state = request.spec.prefill;
    request.position = state.tokens.size();
    request.tokens = state.tokens;
    request.input = request.prefill_next_token = state.next_token;
    for (uint32_t family = 0; family < 4; ++family)
        request.rows[family] = state.kv_rows[family].size();
    for (const auto &image : import_images_.at(specification_index)) {
        Handle destination = 0;
        if (image.type == 0) {
            Page page;
            page.valid = image.valid;
            page.keys = memory_.allocate(sram_[request.home_core], config_.page_rows * packed_size(config_.index_dim));
            if (image.valid == config_.page_rows) {
                page.external = memory_.allocate(bailu_[request.spec.home_chip], memory_.size(image.source));
                destination = page.external;
                page.flush_sent = page.flush_done = true;
            } else {
                page.tail = memory_.allocate(sram_[request.home_core], memory_.size(image.source));
                destination = page.tail;
            }
            request.pages[image.family].push_back(page);
        } else if (image.type == 1) destination = request.pages[image.family].at(image.index).keys;
        else if (image.type == 2) destination = request.windows.at(image.family);
        else destination = request.compressor_states.at(image.family);
        request.imports.push_back({image.source, destination, false, image.type == 0 ? image.valid : 0});
    }
}
StepResult DecodeSimulation::import_prefill(Request &request) {
    if (request.imported) throw std::runtime_error("duplicate prefill import instruction");
    if (request.import_next < request.imports.size()) {
        const auto lane = request.import_next;
        const auto &chunk = request.imports[lane];
        if (!fabric_.send(
                {request.identity(kImport, lane), request.home_core, request.home_core, 0, kImport, chunk.source,
                 chunk.destination}
            ))
            return StepResult::Waiting;
        import_bytes_ += memory_.size(chunk.source);
        ++request.import_next;
        return StepResult::Waiting;
    }
    if (request.import_done != request.imports.size()) return StepResult::Waiting;
    request.imported = true;
    fabric_.record("PREFILL_IMPORTED", request.home_core, request.identity());
    return StepResult::Progress;
}
bool DecodeSimulation::admit_requests() {
    bool progress = false;
    for (uint32_t i = 0; i < specifications_.size(); ++i) {
        const auto &spec = specifications_[i];
        if (std::any_of(requests_.begin(), requests_.end(), [&](const auto &r) {
                return r->spec.name == spec.name;
            }))
            continue;
        bool gate = spec.admit_after.empty();
        bool slot_busy = false;
        for (const auto &previous : requests_) {
            if (previous->spec.name == spec.admit_after)
                gate = spec.admit_after_outputs ? previous->generated.size() >= spec.admit_after_outputs :
                                                  previous->retired;
            if (!previous->retired && previous->spec.slot == spec.slot) slot_busy = true;
        }
        if (!gate || slot_busy) continue;
        auto request = std::make_unique<Request>();
        request->spec = spec;
        request->home_core = home(spec.home_chip);
        request->generation = ++generations_[spec.slot];
        if (request->generation > 1) ++request_reuses_;
        request->streams.resize(4 * config_.hidden);
        request->frame = memory_.allocate(sram_[request->home_core], 4096 + 256 * config_.hidden);
        for (uint32_t layer = 0; layer < config_.layers; ++layer)
            request->windows.push_back(
                memory_.allocate(sram_[request->home_core], config_.window * config_.latent * 8)
            );
        for (auto &state : request->compressor_states)
            state = memory_.allocate(sram_[request->home_core], 8 * 2 * config_.latent * 8);
        request->results.resize(7);
        request->lookup_results.resize(config_.chips);
        bind_import(*request, i);
        fabric_.record("ADMIT", request->home_core, request->identity());
        requests_.push_back(std::move(request));
        progress = true;
    }
    return progress;
}
DecodeSimulation::Request &DecodeSimulation::find_request(const Identity &key) {
    for (auto &request : requests_)
        if (request->spec.slot == key.slot && request->generation == key.generation && !request->retired)
            return *request;
    throw std::runtime_error("message has stale request generation");
}
void DecodeSimulation::embed(Request &request) {
    if (!request.imported) throw std::runtime_error("decode before prefill import completed");
    if (request.position >= config_.max_context) throw std::runtime_error("maximum context write rejected");
    const auto &embedding = embedding_[request.spec.home_chip];
    const auto values = decode_values(memory_.read_local(request.home_core, embedding.handle));
    for (uint32_t stream = 0; stream < 4; ++stream)
        for (uint32_t j = 0; j < config_.hidden; ++j)
            request.streams[stream * config_.hidden + j] = values[request.input * config_.hidden + j];
    request.tokens.push_back(request.input);
    request.layer = 0;
    request.lookup_next = request.lookup_bitmap = 0;
    request.lookup_applied = false;
    fabric_.record("DECODE_INPUT", request.home_core, request.identity());
}
void DecodeSimulation::prepare_attention(Request &request) {
    const auto core = request.home_core;
    const auto &weights = weights_[request.spec.home_chip][request.layer];
    if ((request.layer == 1 || request.layer == 14) && !request.lookup_applied) {
        for (uint32_t chip = 0; chip < config_.chips; ++chip)
            for (uint32_t stream = 0; stream < 4; ++stream)
                for (uint32_t j = 0; j < config_.hidden; ++j)
                    request.streams[stream * config_.hidden + j] += request.lookup_results[chip][j];
        request.lookup_applied = true;
    }
    request.residual = request.streams;
    request.attention_mix = mixes(multiply(core, request.streams, weights.hc_attention), request.streams);
    request.x = normalize(collapse(request.streams, request.attention_mix, config_.hidden));
    request.q = multiply(core, normalize(multiply(core, request.x, weights.q_a)), weights.q_b);
    auto window = decode_values(memory_.read_local(core, request.windows[request.layer]));
    auto kv = normalize(multiply(core, request.x, weights.kv));
    rope(kv, request.position);
    std::copy(kv.begin(), kv.end(), window.begin() + (request.position % config_.window) * config_.latent);
    memory_.write_local(core, request.windows[request.layer], encode_values(window));
    if (is_full(request.layer)) {
        const auto source = source_for_layer(request.layer);
        auto projected = multiply(core, request.x, weights.compressor);
        bool publish = source == 3;
        if (source != 3) {
            auto state = decode_values(memory_.read_local(core, request.compressor_states[source]));
            const auto gate = multiply(core, request.x, weights.gate);
            if (request.position % 2) {
                const auto prior = ((request.position - 1) % 8) * 2 * config_.latent;
                for (uint32_t j = 0; j < config_.latent; ++j) {
                    const auto probability = sigmoid(state[prior + config_.latent + j] - gate[j]);
                    kv[j] = state[prior + j] * probability + projected[j] * (1 - probability);
                }
                publish = true;
            }
            const auto offset = (request.position % 8) * 2 * config_.latent;
            std::copy(projected.begin(), projected.end(), state.begin() + offset);
            std::copy(gate.begin(), gate.end(), state.begin() + offset + config_.latent);
            memory_.write_local(core, request.compressor_states[source], encode_values(state));
            projected = kv;
        }
        if (publish) append_compressed(request, source, normalize(std::move(projected)));
    }
    request.gathered.clear();
    request.fetch_pages.clear();
    request.fetch_next = request.fetch_live = 0;
}
void DecodeSimulation::append_compressed(Request &request, uint32_t source, const Values &latent) {
    const auto row = request.rows[source];
    const auto page_index = row / config_.page_rows;
    auto &pages = request.pages[source];
    if (page_index == pages.size()) {
        Page page;
        page.tail = memory_.allocate(sram_[request.home_core], config_.page_rows * packed_size(config_.latent));
        page.keys = memory_.allocate(sram_[request.home_core], config_.page_rows * packed_size(config_.index_dim));
        pages.push_back(page);
    }
    auto &page = pages.at(page_index);
    if (page.flush_sent) throw std::runtime_error("append to sealed compressed page");
    auto kv = latent;
    rope(kv, row, 160000.0);
    const auto packed = pack_fp4(kv);
    auto bytes = memory_.read_local(request.home_core, page.tail);
    std::copy(packed.begin(), packed.end(), bytes.begin() + (row % config_.page_rows) * packed.size());
    memory_.write_local(request.home_core, page.tail, bytes);
    const auto &weight = weights_[request.spec.home_chip][kSources[source]].index_key;
    const auto key = pack_fp4(normalize(multiply(request.home_core, latent, weight)));
    bytes = memory_.read_local(request.home_core, page.keys);
    std::copy(key.begin(), key.end(), bytes.begin() + (row % config_.page_rows) * key.size());
    memory_.write_local(request.home_core, page.keys, bytes);
    ++page.valid;
    ++request.rows[source];
    ++kv_rows_;
    fabric_.record("ROW_COMMITTED", request.home_core, request.identity(100 + source, row));
}
void DecodeSimulation::select_history(Request &request) {
    if (request.layer < 2) {
        request.selected.clear();
        return;
    }
    const auto source = source_for_layer(request.layer);
    if (is_full(request.layer) || is_reindex(request.layer)) {
        const auto query = normalize(
            multiply(request.home_core, request.x, weights_[request.spec.home_chip][request.layer].index_query)
        );
        std::vector<std::pair<double, uint32_t>> scores;
        for (uint32_t row = 0; row < request.rows[source]; ++row) {
            if (is_reindex(request.layer) &&
                std::find(request.candidate_pool.begin(), request.candidate_pool.end(), row) ==
                    request.candidate_pool.end())
                continue;
            const auto &page = request.pages[source].at(row / config_.page_rows);
            const auto key = unpack_fp4(
                slice(
                    memory_.read_local(request.home_core, page.keys),
                    (row % config_.page_rows) * packed_size(config_.index_dim), packed_size(config_.index_dim)
                ),
                config_.index_dim
            );
            const auto score = std::inner_product(query.begin(), query.end(), key.begin(), 0.0);
            scores.emplace_back(-score, row);
        }
        std::sort(scores.begin(), scores.end());
        request.selected.clear();
        for (size_t i = 0; i < std::min<size_t>(config_.index_top_k, scores.size()); ++i)
            request.selected.push_back(scores[i].second);
        if (request.layer == 20) {
            request.candidate_pool.clear();
            std::set<uint32_t> blocks;
            for (const auto &score : scores) {
                const auto block = score.second / 2;
                if (blocks.size() < 4 || blocks.count(block)) blocks.insert(block);
            }
            for (uint32_t row = 0; row < request.rows[source]; ++row)
                if (blocks.count(row / 2)) request.candidate_pool.push_back(row);
        }
    }
    std::set<uint32_t> page_set;
    for (auto row : request.selected)
        page_set.insert(row / config_.page_rows);
    request.fetch_pages.assign(page_set.begin(), page_set.end());
}
StepResult DecodeSimulation::gather_history(Request &request) {
    if (request.layer < 2) return StepResult::Progress;
    const auto source = source_for_layer(request.layer);
    if (request.fetch_next < request.fetch_pages.size() && request.fetch_live < 2) {
        const auto page_index = request.fetch_pages[request.fetch_next];
        auto &page = request.pages[source].at(page_index);
        const auto source_handle = page.flush_done ? page.external : page.tail;
        if (!source_handle) throw std::runtime_error("history has no authoritative copy");
        auto destination = memory_.allocate(sram_[request.home_core], memory_.size(source_handle));
        Message message{request.identity(kFetch, page_index),
                        request.home_core,
                        request.home_core,
                        source,
                        kFetch,
                        source_handle,
                        destination};
        if (!fabric_.send(message)) {
            memory_.release(destination);
            return StepResult::Waiting;
        }
        ++request.fetch_next;
        ++request.fetch_live;
        if (page.flush_done) ++history_fetches_;
        return StepResult::Waiting;
    }
    return request.fetch_next == request.fetch_pages.size() && !request.fetch_live ? StepResult::Progress :
                                                                                     StepResult::Waiting;
}
void DecodeSimulation::compute_attention(Request &request) {
    const auto window = decode_values(memory_.read_local(request.home_core, request.windows[request.layer]));
    Values output(config_.heads * config_.latent);
    for (uint32_t head = 0; head < config_.heads; ++head) {
        Values q(request.q.begin() + head * config_.latent, request.q.begin() + (head + 1) * config_.latent);
        auto compressed_q = q;
        rope(q, request.position);
        rope(compressed_q, source_for_layer(request.layer) == 3 ? request.position : request.position / 2, 160000.0);
        std::vector<Values> rows;
        Values scores;
        const auto first = request.position + 1 > config_.window ? request.position + 1 - config_.window : 0;
        for (uint32_t position = first; position <= request.position; ++position) {
            const auto start = (position % config_.window) * config_.latent;
            rows.emplace_back(window.begin() + start, window.begin() + start + config_.latent);
            scores.push_back(
                std::inner_product(q.begin(), q.end(), rows.back().begin(), 0.0) / std::sqrt(config_.latent)
            );
        }
        for (const auto row : request.selected) {
            rows.push_back(request.gathered.at(row));
            scores.push_back(
                std::inner_product(compressed_q.begin(), compressed_q.end(), rows.back().begin(), 0.0) /
                std::sqrt(config_.latent)
            );
        }
        scores = softmax(std::move(scores));
        for (size_t i = 0; i < rows.size(); ++i)
            for (uint32_t j = 0; j < config_.latent; ++j)
                output[head * config_.latent + j] += scores[i] * rows[i][j];
    }
    const auto &weights = weights_[request.spec.home_chip][request.layer];
    request.streams = expand(
        multiply(request.home_core, output, weights.output), request.residual, request.attention_mix, config_.hidden
    );
    request.residual = request.streams;
    request.ffn_mix = mixes(multiply(request.home_core, request.streams, weights.hc_ffn), request.streams);
    request.x = normalize(collapse(request.streams, request.ffn_mix, config_.hidden));
}
void DecodeSimulation::route(Request &request) {
    auto scores = multiply(request.home_core, request.x, weights_[request.spec.home_chip][request.layer].router);
    std::vector<std::pair<double, uint32_t>> ordered;
    for (uint32_t i = 0; i < scores.size(); ++i)
        ordered.emplace_back(-std::sqrt(std::log1p(std::exp(scores[i]))), i);
    std::sort(ordered.begin(), ordered.end());
    request.routes.clear();
    request.route_weights.clear();
    double sum = 0;
    for (uint32_t i = 0; i < config_.top_k; ++i) {
        request.routes.push_back(ordered[i].second);
        request.route_weights.push_back(-ordered[i].first);
        sum -= ordered[i].first;
    }
    for (auto &weight : request.route_weights)
        weight *= 1.5 / sum;
    request.result_bitmap = request.dispatch_next = 0;
    for (auto &result : request.results)
        result.clear();
}
StepResult DecodeSimulation::dispatch(Request &request) {
    if (request.dispatch_next == 7) return StepResult::Progress;
    const auto lane = request.dispatch_next;
    const bool shared = lane == 6;
    const auto owner = shared ? request.home_core + 5 : expert_owner(request.routes[lane]);
    auto source = memory_.allocate(sram_[request.home_core], request.x.size() * 8);
    auto destination = memory_.allocate(sram_[owner], request.x.size() * 8);
    memory_.write_local(request.home_core, source, encode_values(request.x));
    Message message{
        request.identity(shared ? kShared : kExpert, lane),
        request.home_core,
        owner,
        shared ? 8u : request.routes[lane],
        shared ? kShared : kExpert,
        source,
        destination
    };
    if (!fabric_.send(message)) {
        memory_.release(source);
        memory_.release(destination);
        return StepResult::Waiting;
    }
    ++request.dispatch_next;
    return StepResult::Waiting;
}
StepResult DecodeSimulation::combine(Request &request) {
    if (request.result_bitmap != 127) return StepResult::Waiting;
    Values result = request.results[6];
    for (uint32_t lane = 0; lane < 6; ++lane)
        for (uint32_t j = 0; j < config_.hidden; ++j)
            result[j] += request.route_weights[lane] * request.results[lane][j];
    request.streams = expand(result, request.residual, request.ffn_mix, config_.hidden);
    double checksum = 0;
    for (size_t i = 0; i < request.streams.size(); ++i)
        checksum += request.streams[i] * (i + 1);
    request.layer_checksums.push_back(checksum);
    fabric_.record("LAYER_DONE", request.home_core, request.identity());
    return StepResult::Progress;
}

bool DecodeSimulation::receive(uint32_t core) {
    if (!fabric_.has_message(core)) return false;
    const auto message = fabric_.receive(core);
    auto &request = find_request(message.key);
    if (message.kind == kExpert || message.kind == kShared || message.kind == kLookup) {
        ExpertWork work{};
        work.incoming = message;
        work_.at(core).push_back(std::move(work));
        return true;
    }
    if (core != request.home_core) throw std::runtime_error("response arrived at incorrect request home");
    if (message.kind == kImport) {
        auto &chunk = request.imports.at(message.key.lane);
        if (request.imported || chunk.consumed || message.destination != chunk.destination ||
            message.source != chunk.source || message.key.position != request.position || message.key.layer != 0)
            throw std::runtime_error("invalid prefill import completion");
        chunk.consumed = true;
        ++request.import_done;
        imported_rows_ += chunk.rows;
        memory_.release(message.source);
        fabric_.record("PREFILL_CHUNK_READY", core, message.key);
        return true;
    }
    if (message.kind == kStore) {
        auto &page = request.pages.at(message.expert).at(message.key.lane);
        if (!page.flush_sent || page.flush_done || message.destination != page.external)
            throw std::runtime_error("invalid KV directory store completion");
        page.flush_done = true;
        fabric_.record("DIRECTORY_BAILU", core, message.key);
        return true;
    }
    if (message.key.position != request.position || message.key.layer != request.layer)
        throw std::runtime_error("response epoch does not match live request continuation");
    if (message.kind == kFetch) {
        const auto bytes = memory_.read_local(core, message.destination);
        for (auto row : request.selected)
            if (row / config_.page_rows == message.key.lane)
                request.gathered.emplace(
                    row,
                    unpack_fp4(
                        slice(
                            bytes, (row % config_.page_rows) * packed_size(config_.latent), packed_size(config_.latent)
                        ),
                        config_.latent
                    )
                );
        if (!request.fetch_live) throw std::runtime_error("unexpected gather completion");
        --request.fetch_live;
        memory_.release(message.destination);
        return true;
    }
    const auto values = decode_values(memory_.read_local(core, message.destination));
    if (message.kind == kResult) {
        const auto lane = message.key.lane;
        if (lane > 6 || (request.result_bitmap & (1u << lane)))
            throw std::runtime_error("duplicate or invalid expert lane");
        if (lane < 6 && request.routes.at(lane) != message.expert)
            throw std::runtime_error("expert returned on wrong route");
        request.result_bitmap |= 1u << lane;
        request.results[lane] = values;
    } else if (message.kind == kLookupResult) {
        const auto lane = message.key.lane;
        if (lane >= config_.chips || (request.lookup_bitmap & (1u << lane)))
            throw std::runtime_error("duplicate lookup response");
        request.lookup_bitmap |= 1u << lane;
        request.lookup_results[lane] = values;
    } else {
        throw std::runtime_error("unknown message protocol");
    }
    memory_.release(message.source);
    memory_.release(message.destination);
    return true;
}
bool DecodeSimulation::step_expert(uint32_t core) {
    auto &queue = work_.at(core);
    if (queue.empty()) return false;
    auto &work = queue.front();
    OperationRegistry operations;
    operations.add(30, [&](uint32_t) {
        if (work.computed) throw std::runtime_error("duplicate service compute instruction");
        const auto x = decode_values(memory_.read_local(core, work.incoming.destination));
        Values result(config_.hidden);
        const auto chip = core / config_.cores_per_chip;
        if (work.incoming.kind == kLookup) {
            if (core != home(chip) + 6) throw std::runtime_error("lookup on non-table owner");
            const auto table = decode_values(memory_.read_local(core, engram_[chip].handle));
            for (auto row : x) {
                if (row < 0) continue;
                const auto index = static_cast<uint32_t>(row);
                if (index >= 64) throw std::runtime_error("Engram index outside shard");
                for (uint32_t j = 0; j < config_.hidden; ++j)
                    result[j] += table[index * config_.hidden + j] / 24;
            }
        } else {
            const auto &weights = weights_[chip].at(work.incoming.key.layer);
            std::array<Matrix, 3> matrices;
            if (work.incoming.kind == kShared) {
                if (core != home(chip) + 5) throw std::runtime_error("shared expert on wrong owner");
                matrices = {weights.shared_a, weights.shared_b, weights.shared_c};
            } else {
                if (core != expert_owner(work.incoming.expert))
                    throw std::runtime_error("expert weight owner mismatch");
                matrices = weights.experts.at(work.incoming.expert);
            }
            auto a = multiply(core, x, matrices[0]);
            const auto b = multiply(core, x, matrices[1]);
            for (size_t i = 0; i < a.size(); ++i)
                a[i] = std::clamp(a[i], -10.0, 10.0) * sigmoid(a[i]) * b[i];
            result = multiply(core, a, matrices[2]);
        }
        memory_.release(work.incoming.source);
        memory_.release(work.incoming.destination);
        work.output = memory_.allocate(sram_[core], result.size() * 8);
        memory_.write_local(core, work.output, encode_values(result));
        const auto destination_core = find_request(work.incoming.key).home_core;
        work.destination = memory_.allocate(sram_[destination_core], result.size() * 8);
        work.computed = true;
        fabric_.record(work.incoming.kind == kLookup ? "LOOKUP_COMPUTE" : "EXPERT_COMPUTE", core, work.incoming.key);
        return StepResult::Progress;
    });
    operations.add(31, [&](uint32_t) {
        if (!work.computed || work.sent) throw std::runtime_error("service send outside computed result state");
        const auto kind = work.incoming.kind == kLookup ? kLookupResult : kResult;
        const auto destination_core = find_request(work.incoming.key).home_core;
        Message message{work.incoming.key, core, destination_core, work.incoming.expert, kind, work.output,
                        work.destination};
        if (!fabric_.send(message)) return StepResult::Waiting;
        work.sent = true;
        return StepResult::Progress;
    });
    const auto result = work.continuation.step(service_program_, operations);
    if (result == StepResult::Halted) {
        if (!work.sent) throw std::runtime_error("service halted before returning result");
        queue.pop_front();
    }
    return result != StepResult::Waiting;
}
StepResult DecodeSimulation::flush_pages(Request &request) {
    for (uint32_t source = 0; source < 4; ++source)
        for (uint32_t i = 0; i < request.pages[source].size(); ++i) {
            auto &page = request.pages[source][i];
            if (!page.flush_sent && (page.valid == config_.page_rows || request.finishing)) {
                if (!page.external)
                    page.external = memory_.allocate(bailu_[request.spec.home_chip], memory_.size(page.tail));
                Message message{request.identity(kStore, i),
                                request.home_core,
                                request.home_core,
                                source,
                                kStore,
                                page.tail,
                                page.external};
                if (!fabric_.send(message)) return StepResult::Waiting;
                page.flush_sent = true;
                fabric_.record("TAIL_FLUSH", request.home_core, message.key);
                return StepResult::Progress;
            }
        }
    return StepResult::Waiting;
}
bool DecodeSimulation::collect_pages(Request &request) {
    bool progress = false;
    for (auto &pages : request.pages)
        for (auto &page : pages)
            if (page.flush_done && page.tail && !memory_.in_use(page.tail)) {
                memory_.release(page.tail);
                page.tail = 0;
                progress = true;
            }
    return progress;
}
bool DecodeSimulation::service_work_pending(uint32_t slot, uint64_t generation) const {
    for (const auto &queue : work_)
        for (const auto &work : queue)
            if (work.incoming.key.slot == slot && work.incoming.key.generation == generation) return true;
    return false;
}
void DecodeSimulation::sample(Request &request) {
    Values head(config_.hidden);
    for (uint32_t j = 0; j < config_.hidden; ++j)
        for (uint32_t stream = 0; stream < 4; ++stream)
            head[j] += request.streams[stream * config_.hidden + j] / 4;
    auto logits = multiply(request.home_core, normalize(std::move(head)), lm_head_[request.spec.home_chip]);
    const auto token = static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    ++request.position;
    request.generated.push_back(token);
    request.logits.push_back(std::move(logits));
    fabric_.record("OUTPUT_TOKEN", request.home_core, request.identity());
    request.input = token;
    if (config_.cancel_long_request && request.spec.name == "B" && request.generated.size() == 2) {
        request.cancelled = true;
        fabric_.record("CANCEL", request.home_core, request.identity());
    }
    request.finishing = request.cancelled || request.generated.size() == request.spec.decode_inputs;
    if (!request.finishing) request.continuation.redirect(1);
}
bool DecodeSimulation::step_request(Request &request) {
    if (request.retired) return false;
    bool progress = collect_pages(request);
    progress |= flush_pages(request) == StepResult::Progress;
    if (request.finishing) {
        bool flushed = true;
        for (const auto &pages : request.pages)
            for (const auto &page : pages)
                flushed &= page.flush_done && !page.tail;
        if (flushed && !fabric_.outstanding(request.spec.slot, request.generation) &&
            !service_work_pending(request.spec.slot, request.generation)) {
            retire(request);
            return true;
        }
        return progress;
    }
    if (!request.imported && request.continuation.pc < core_program_.code.size()) {
        const auto &instruction = core_program_.code[request.continuation.pc];
        if (instruction.kind == InstructionKind::Invoke && instruction.operation != 20)
            throw std::runtime_error("decode before prefill import completed");
    }
    OperationRegistry operations;
    operations.add(20, [&](uint32_t) {
        return import_prefill(request);
    });
    operations.add(10, [&](uint32_t) {
        embed(request);
        return StepResult::Progress;
    });
    operations.add(11, [&](uint32_t) {
        if (request.layer == 1 || request.layer == 14) {
            if (request.lookup_next < config_.chips) {
                const auto chip = request.lookup_next;
                Values rows;
                for (uint32_t ngram = 2; ngram <= 4; ++ngram)
                    for (uint32_t head = 0; head < 8; ++head) {
                        uint32_t hash = 2166136261u ^ (head + ngram * 17);
                        for (uint32_t n = 0; n < ngram; ++n) {
                            const auto token =
                                request.tokens.size() > n ? request.tokens[request.tokens.size() - 1 - n] : 2;
                            hash = (hash ^ token) * 16777619u;
                        }
                        const auto row = hash % (64 * config_.chips);
                        if (row % config_.chips == chip) rows.push_back(row / config_.chips);
                    }
                if (rows.empty()) rows.push_back(-1);
                const auto owner = home(chip) + 6;
                auto source = memory_.allocate(sram_[request.home_core], rows.size() * 8);
                auto destination = memory_.allocate(sram_[owner], rows.size() * 8);
                memory_.write_local(request.home_core, source, encode_values(rows));
                Message message{
                    request.identity(kLookup, chip), request.home_core, owner, chip, kLookup, source, destination
                };
                if (!fabric_.send(message)) {
                    memory_.release(source);
                    memory_.release(destination);
                    return StepResult::Waiting;
                }
                ++request.lookup_next;
                return StepResult::Waiting;
            }
            if (request.lookup_bitmap != (1u << config_.chips) - 1) return StepResult::Waiting;
        }
        prepare_attention(request);
        return StepResult::Progress;
    });
    operations.add(12, [&](uint32_t) {
        select_history(request);
        return StepResult::Progress;
    });
    operations.add(13, [&](uint32_t) {
        return gather_history(request);
    });
    operations.add(14, [&](uint32_t) {
        compute_attention(request);
        return StepResult::Progress;
    });
    operations.add(15, [&](uint32_t) {
        route(request);
        return StepResult::Progress;
    });
    operations.add(16, [&](uint32_t) {
        return dispatch(request);
    });
    operations.add(17, [&](uint32_t) {
        return combine(request);
    });
    operations.add(18, [&](uint32_t layer_entry) {
        ++request.layer;
        request.lookup_next = request.lookup_bitmap = 0;
        request.lookup_applied = false;
        if (request.layer < config_.layers) request.continuation.redirect(layer_entry);
        return StepResult::Progress;
    });
    operations.add(19, [&](uint32_t) {
        sample(request);
        return request.finishing ? StepResult::Halted : StepResult::Progress;
    });
    for (const auto &operation : custom_core_operations_) {
        const auto callback = operation.second;
        operations.add(operation.first, [&, callback](uint32_t) {
            callback(request.streams);
            return StepResult::Progress;
        });
    }
    const auto result = request.continuation.step(core_program_, operations);
    ++core_steps_;
    if (result == StepResult::Halted && !request.finishing)
        throw std::runtime_error("AICore halted before request completed");
    return progress || result != StepResult::Waiting;
}
void DecodeSimulation::retire(Request &request) {
    for (uint32_t source = 0; source < 4; ++source)
        for (auto &page : request.pages[source]) {
            const auto bytes = memory_.inspect_after_drain(page.external);
            const auto keys = memory_.read_local(request.home_core, page.keys);
            for (uint32_t row = 0; row < page.valid; ++row) {
                request.final_rows[source].push_back(unpack_fp4(
                    slice(bytes, row * packed_size(config_.latent), packed_size(config_.latent)), config_.latent
                ));
                request.final_keys[source].push_back(unpack_fp4(
                    slice(keys, row * packed_size(config_.index_dim), packed_size(config_.index_dim)), config_.index_dim
                ));
            }
            memory_.release(page.external);
            memory_.release(page.keys);
        }
    for (auto window : request.windows) {
        const auto values = decode_values(memory_.read_local(request.home_core, window));
        std::vector<Values> rows;
        const auto start = request.position > config_.window ? request.position - config_.window : 0;
        for (uint32_t position = start; position < request.position; ++position) {
            const auto offset = (position % config_.window) * config_.latent;
            rows.emplace_back(values.begin() + offset, values.begin() + offset + config_.latent);
        }
        request.final_windows.push_back(std::move(rows));
        memory_.release(window);
    }
    for (auto state : request.compressor_states) {
        request.final_states.push_back(decode_values(memory_.read_local(request.home_core, state)));
        memory_.release(state);
    }
    memory_.release(request.frame);
    request.retired = true;
    fabric_.record("REQUEST_DRAINED", request.home_core, request.identity());
}
uint32_t DecodeSimulation::completed_requests() const {
    return std::count_if(requests_.begin(), requests_.end(), [](const auto &request) {
        return request->retired;
    });
}
void DecodeSimulation::run() {
    if (running_ || specifications_.empty()) throw std::runtime_error("service already launched or empty");
    running_ = true;
    ++launch_count_;
    fabric_.record("SERVICE_LAUNCH", 0);
    uint64_t idle_ticks = 0;
    while (completed_requests() != specifications_.size() || !fabric_.idle()) {
        bool progress = fabric_.deliver_due();
        progress |= admit_requests();
        for (uint32_t core = 0; core < work_.size(); ++core) {
            progress |= receive(core);
            progress |= step_expert(core);
        }
        for (auto &request : requests_)
            progress |= step_request(*request);
        for (uint32_t endpoint = 0; endpoint < fabric_.endpoint_count(); ++endpoint) {
            const auto index = config_.reverse_batcher ? fabric_.endpoint_count() - 1 - endpoint : endpoint;
            progress |= fabric_.step_batcher(index);
        }
        fabric_.advance_tick();
        idle_ticks = progress ? 0 : idle_ticks + 1;
        if (idle_ticks > 10000 || fabric_.tick() > 10000000)
            throw std::runtime_error("service progress failure: inspect pending request/program/credit state");
    }
    fabric_.record("SERVICE_DRAINED", 0);
}

void DecodeSimulation::write_report(const std::filesystem::path &output) const {
    std::filesystem::create_directories(output);
    std::ofstream report(output / "results.json");
    report << std::setprecision(17);
    report << "{\"executor\":\"wse_persistent_logical\",\"payload\":\"dense_numeric\","
              "\"model\":\"scaled_structural_fixture\",\"arithmetic\":\"fp64_fixture_fp4_history\","
              "\"time\":\"untimed_logical_ticks\",\"prefill\":\"external_snapshot_import\","
              "\"wse_prefill_compute_positions\":0,"
              "\"full_checkpoint_validated\":false,\"launches\":"
           << launch_count_ << ",\"ticks\":" << fabric_.tick() << ",\"core_steps\":" << core_steps_
           << ",\"bytes_transferred\":" << fabric_.bytes_transferred() << ",\"prefill_import_bytes\":" << import_bytes_
           << ",\"decode_bytes_transferred\":" << fabric_.bytes_transferred() - import_bytes_
           << ",\"prefill_kv_rows_imported\":" << imported_rows_
           << ",\"custom_batcher_steps\":" << fabric_.custom_program_steps()
           << ",\"credit_stalls\":" << fabric_.credit_stalls() << ",\"bailu_fetches\":" << history_fetches_
           << ",\"kv_rows_committed\":" << kv_rows_ << ",\"slot_reuses\":" << request_reuses_ << ",\"all_drained\":"
           << (fabric_.idle() && completed_requests() == specifications_.size() ? "true" : "false")
           << ",\"config\":{\"layers\":" << config_.layers << ",\"hidden\":" << config_.hidden
           << ",\"latent\":" << config_.latent << ",\"index_dim\":" << config_.index_dim
           << ",\"heads\":" << config_.heads << ",\"intermediate\":" << config_.intermediate
           << ",\"vocabulary\":" << config_.vocabulary << ",\"window\":" << config_.window
           << ",\"page_rows\":" << config_.page_rows << ",\"index_top_k\":" << config_.index_top_k
           << ",\"max_context\":" << config_.max_context << "},\"requests\":[";
    for (size_t i = 0; i < requests_.size(); ++i) {
        const auto &request = *requests_[i];
        if (i) report << ',';
        report << "{\"name\":" << json_string(request.spec.name) << ",\"slot\":" << request.spec.slot
               << ",\"generation\":" << request.generation << ",\"home_chip\":" << request.spec.home_chip
               << ",\"prompt\":";
        json_integers(report, request.spec.prefill.tokens);
        report << ",\"processed_length\":" << request.position
               << ",\"prefill_next_token\":" << request.prefill_next_token
               << ",\"cancelled\":" << (request.cancelled ? "true" : "false") << ",\"generated\":";
        json_integers(report, request.generated);
        report << ",\"logits\":[";
        for (size_t step = 0; step < request.logits.size(); ++step) {
            if (step) report << ',';
            json_values(report, request.logits[step]);
        }
        report << "],\"layer_checksums\":";
        json_values(report, request.layer_checksums);
        report << ",\"cache_rows\":[";
        for (uint32_t source = 0; source < 4; ++source) {
            if (source) report << ',';
            report << '[';
            for (size_t row = 0; row < request.final_rows[source].size(); ++row) {
                if (row) report << ',';
                json_values(report, request.final_rows[source][row]);
            }
            report << ']';
        }
        report << "],\"key_rows\":[";
        for (uint32_t source = 0; source < 4; ++source) {
            if (source) report << ',';
            report << '[';
            for (size_t row = 0; row < request.final_keys[source].size(); ++row) {
                if (row) report << ',';
                json_values(report, request.final_keys[source][row]);
            }
            report << ']';
        }
        report << "],\"windows\":[";
        for (size_t layer = 0; layer < request.final_windows.size(); ++layer) {
            if (layer) report << ',';
            report << '[';
            for (size_t row = 0; row < request.final_windows[layer].size(); ++row) {
                if (row) report << ',';
                json_values(report, request.final_windows[layer][row]);
            }
            report << ']';
        }
        report << "],\"compressor_states\":[";
        for (size_t source = 0; source < request.final_states.size(); ++source) {
            if (source) report << ',';
            json_values(report, request.final_states[source]);
        }
        report << "]}";
    }
    report << "],\"memory\":[";
    for (size_t i = 0; i < memory_.regions().size(); ++i) {
        if (i) report << ',';
        const auto &region = memory_.regions()[i];
        report << "{\"tier\":" << static_cast<int>(region.tier) << ",\"owner\":" << region.owner
               << ",\"capacity\":" << region.capacity << ",\"used\":" << region.used << ",\"peak\":" << region.peak
               << '}';
    }
    report << "]}\n";
    std::ofstream events(output / "events.jsonl");
    for (const auto &event : fabric_.events())
        events << "{\"tick\":" << event.tick << ",\"event\":\"" << event.event << "\",\"actor\":" << event.actor
               << ",\"slot\":" << event.key.slot << ",\"generation\":" << event.key.generation
               << ",\"position\":" << event.key.position << ",\"layer\":" << event.key.layer
               << ",\"phase\":" << event.key.phase << ",\"lane\":" << event.key.lane << ",\"bytes\":" << event.bytes
               << ",\"transfer_id\":" << event.transfer_id << "}\n";
    std::ofstream programs(output / "programs.json");
    programs << "{\"aicore\":{\"name\":" << json_string(core_program_.name) << ",\"instructions\":[";
    for (size_t i = 0; i < core_program_.code.size(); ++i) {
        if (i) programs << ',';
        const auto &op = core_program_.code[i];
        programs << '[' << static_cast<int>(op.kind) << ',' << op.operation << ',' << op.argument << ']';
    }
    programs << "]},\"batcher\":{\"name\":" << json_string(fabric_.program().name) << ",\"instructions\":[";
    for (size_t i = 0; i < fabric_.program().code.size(); ++i) {
        if (i) programs << ',';
        const auto &op = fabric_.program().code[i];
        programs << '[' << static_cast<int>(op.kind) << ',' << op.operation << ',' << op.argument << ']';
    }
    programs << "]},\"service\":{\"name\":" << json_string(service_program_.name) << ",\"instructions\":[";
    for (size_t i = 0; i < service_program_.code.size(); ++i) {
        if (i) programs << ',';
        const auto &op = service_program_.code[i];
        programs << '[' << static_cast<int>(op.kind) << ',' << op.operation << ',' << op.argument << ']';
    }
    programs << "]}}\n";
}

}  // namespace wse::persistent
