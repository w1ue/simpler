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

#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace wse::persistent {

enum class StepResult { Progress, Waiting, Halted };
enum class InstructionKind : uint32_t { Invoke, Jump, Halt };

struct Instruction {
    InstructionKind kind;
    uint32_t operation;
    uint32_t argument;
};
static_assert(std::is_trivially_copyable_v<Instruction> && std::is_standard_layout_v<Instruction>);

struct Program {
    std::string name;
    std::vector<Instruction> code;

    void validate() const {
        if (name.empty() || code.empty()) throw std::invalid_argument("empty device program");
        for (const auto &instruction : code) {
            if (instruction.kind == InstructionKind::Jump && instruction.argument >= code.size())
                throw std::invalid_argument("program jump outside instruction image");
            if (instruction.kind != InstructionKind::Invoke && instruction.kind != InstructionKind::Jump &&
                instruction.kind != InstructionKind::Halt)
                throw std::invalid_argument("unknown device instruction");
        }
    }
};

class OperationRegistry {
public:
    using Operation = std::function<StepResult(uint32_t)>;

    void add(uint32_t id, Operation operation) {
        if (!operation || !operations_.emplace(id, std::move(operation)).second)
            throw std::invalid_argument("duplicate or empty programmable operation");
    }

    StepResult invoke(uint32_t id, uint32_t argument) const {
        const auto found = operations_.find(id);
        if (found == operations_.end()) throw std::invalid_argument("unregistered device operation");
        return found->second(argument);
    }

private:
    std::map<uint32_t, Operation> operations_;
};

struct Continuation {
    uint32_t pc = 0;
    bool halted = false;
    bool redirected = false;

    void redirect(uint32_t destination) {
        pc = destination;
        redirected = true;
    }

    StepResult step(const Program &program, const OperationRegistry &operations) {
        if (halted) return StepResult::Halted;
        if (pc >= program.code.size()) throw std::runtime_error("program fell off instruction image");
        const auto &instruction = program.code[pc];
        switch (instruction.kind) {
        case InstructionKind::Invoke: {
            redirected = false;
            const auto result = operations.invoke(instruction.operation, instruction.argument);
            if (result == StepResult::Progress && !redirected) ++pc;
            if (result == StepResult::Halted) halted = true;
            return result;
        }
        case InstructionKind::Jump:
            pc = instruction.argument;
            return StepResult::Progress;
        case InstructionKind::Halt:
            halted = true;
            return StepResult::Halted;
        }
        throw std::runtime_error("invalid instruction kind");
    }
};

struct Identity {
    uint32_t slot = 0;
    uint64_t generation = 0;
    uint32_t position = 0;
    uint32_t layer = 0;
    uint32_t phase = 0;
    uint32_t lane = 0;

    bool operator==(const Identity &other) const {
        return slot == other.slot && generation == other.generation && position == other.position &&
               layer == other.layer && phase == other.phase && lane == other.lane;
    }
};

}  // namespace wse::persistent
