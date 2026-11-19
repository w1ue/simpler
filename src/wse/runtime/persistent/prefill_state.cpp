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
#include "prefill_state.h"

#include "decode.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

namespace wse::persistent {
namespace {
constexpr uint64_t kMaximumSnapshotBytes = 64 * 1024 * 1024;
std::vector<uint32_t> model_contract(const DecodeConfig &c) {
    return {c.layers,  c.hidden, c.latent,     c.index_dim, c.heads,       c.intermediate,
            c.experts, c.top_k,  c.vocabulary, c.window,    c.index_top_k, 1};
}
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(std::string("prefill snapshot: ") + message);
}
class Reader {
public:
    explicit Reader(Bytes bytes) :
        bytes_(std::move(bytes)) {}
    Bytes take(size_t size) {
        require(size <= bytes_.size() - offset_, "truncated payload");
        Bytes result(bytes_.begin() + offset_, bytes_.begin() + offset_ + size);
        offset_ += size;
        return result;
    }
    uint64_t integer(size_t width = 4) {
        const auto bytes = take(width);
        uint64_t value = 0;
        for (size_t i = 0; i < width; ++i)
            value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
        return value;
    }
    Values values(size_t count) {
        require(count <= (bytes_.size() - offset_) / 8, "truncated dense state");
        Values result(count);
        for (auto &value : result) {
            const auto bits = integer(8);
            std::memcpy(&value, &bits, 8);
            require(std::isfinite(value), "nonfinite dense state");
        }
        return result;
    }
    bool done() const { return offset_ == bytes_.size(); }

private:
    Bytes bytes_;
    size_t offset_ = 0;
};
void check_finite(const Values &values) {
    require(
        std::all_of(
            values.begin(), values.end(),
            [](double value) {
                return std::isfinite(value);
            }
        ),
        "nonfinite state"
    );
}
}  // namespace

void PrefillState::validate(const DecodeConfig &config) const {
    require(contract == model_contract(config), "model contract mismatch");
    require(!tokens.empty() && tokens.size() <= config.max_context, "invalid processed length");
    require(next_token < config.vocabulary, "next token outside vocabulary");
    for (auto token : tokens)
        require(token < config.vocabulary, "history token outside vocabulary");
    for (uint32_t source = 0; source < 4; ++source) {
        const auto count = source == 3 ? tokens.size() : tokens.size() / 2;
        require(kv_rows[source].size() == count && key_rows[source].size() == count, "row count mismatch");
        for (const auto &row : kv_rows[source])
            check_finite(unpack_fp4(row, config.latent));
        for (const auto &row : key_rows[source])
            check_finite(unpack_fp4(row, config.index_dim));
    }
    require(windows.size() == config.layers, "window layer count mismatch");
    for (const auto &window : windows) {
        require(
            window.size() == std::min<size_t>(tokens.size(), config.window) * config.latent, "window row count mismatch"
        );
        check_finite(window);
    }
    for (const auto &state : compressor_states) {
        require(state.size() == 8 * 2 * config.latent, "compressor state size mismatch");
        check_finite(state);
    }
}

PrefillState PrefillState::load(const std::filesystem::path &path, const DecodeConfig &config) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(stream.good(), "cannot open file");
    const auto size = stream.tellg();
    require(size >= 24 && static_cast<uint64_t>(size) <= kMaximumSnapshotBytes, "invalid file size");
    Bytes bytes(static_cast<size_t>(size));
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    require(stream.good(), "read failed");
    Reader envelope(std::move(bytes));
    const auto magic = envelope.take(8);
    require(magic == Bytes({'W', 'S', 'E', 'P', 'F', '0', '0', '1'}), "invalid magic");
    require(envelope.integer() == 1, "unsupported version");
    const auto length = envelope.integer();
    const auto expected_checksum = envelope.integer(8);
    const auto payload = envelope.take(length);
    require(envelope.done(), "trailing bytes");
    uint64_t checksum = 14695981039346656037ull;
    for (auto byte : payload)
        checksum = (checksum ^ byte) * 1099511628211ull;
    require(checksum == expected_checksum, "checksum mismatch");
    Reader reader(payload);
    PrefillState result;
    for (size_t i = 0; i < 12; ++i)
        result.contract.push_back(reader.integer());
    require(result.contract == model_contract(config), "model contract mismatch");
    const auto position = reader.integer();
    require(position && position <= config.max_context, "invalid processed length");
    result.next_token = reader.integer();
    for (size_t i = 0; i < position; ++i)
        result.tokens.push_back(reader.integer());
    for (uint32_t source = 0; source < 4; ++source) {
        const auto rows = reader.integer();
        require(rows == (source == 3 ? position : position / 2), "row count mismatch");
        for (size_t i = 0; i < rows; ++i) {
            result.kv_rows[source].push_back(reader.take(config.latent));
            result.key_rows[source].push_back(reader.take(config.index_dim));
        }
    }
    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto rows = reader.integer();
        require(rows == std::min<uint64_t>(position, config.window), "window row count mismatch");
        result.windows.push_back(reader.values(rows * config.latent));
    }
    for (auto &state : result.compressor_states)
        state = reader.values(8 * 2 * config.latent);
    require(reader.done(), "trailing state fields");
    result.validate(config);
    return result;
}
}  // namespace wse::persistent
