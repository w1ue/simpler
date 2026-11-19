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

#include <iostream>
#include <stdexcept>

using namespace wse::persistent;

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename Function>
void rejects(Function function, const char *message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, message);
}
}  // namespace

int main() {
    try {
        Memory memory;
        const auto a = memory.add_region(Tier::Sram, 0, 512);
        const auto b = memory.add_region(Tier::Sram, 1, 512);
        const auto external = memory.add_region(Tier::Bailu, 0, 512);
        const auto source = memory.allocate(a, 8);
        const auto destination = memory.allocate(b, 8);
        memory.write_local(0, source, encode_values({7.25}));
        rejects(
            [&] {
                memory.read_local(1, source);
            },
            "remote SRAM read accepted"
        );
        rejects(
            [&] {
                memory.allocate(a, 1000);
            },
            "capacity overrun accepted"
        );
        const auto cold = memory.allocate(external, 8);
        const auto host_region = memory.add_region(Tier::Host, 0, 512);
        const auto host_payload = memory.allocate(host_region, 8);
        memory.load_before_launch(host_payload, encode_values({4.0}));
        rejects(
            [&] {
                memory.read_local(0, host_payload);
            },
            "direct Host state compute accepted"
        );
        rejects(
            [&] {
                memory.read_local(0, cold);
            },
            "direct Bailu compute accepted"
        );
        Fabric fabric(memory, 2, 2, 1);
        require(fabric.send({{}, 0, 1, 0, 0, source, destination}), "send failed");
        rejects(
            [&] {
                memory.release(source);
            },
            "queued source reclaimed"
        );
        rejects(
            [&] {
                memory.read_local(1, destination);
            },
            "destination read before delivery accepted"
        );
        for (uint32_t i = 0; i < 100 && !fabric.has_message(1); ++i) {
            fabric.deliver_due();
            fabric.step_batcher(0);
            fabric.advance_tick();
        }
        require(fabric.has_message(1), "message never published");
        const auto message = fabric.receive(1);
        require(decode_values(memory.read_local(1, message.destination))[0] == 7.25, "payload changed");
        for (uint32_t i = 0; i < 10 && !fabric.idle(); ++i)
            fabric.step_batcher(0);
        require(fabric.idle(), "fabric did not drain");
        memory.release(source);
        rejects(
            [&] {
                memory.read_local(0, source);
            },
            "stale handle accepted"
        );

        // A source lease ends at DMA completion; destination ownership lasts until consume.
        Memory delayed;
        const auto region = delayed.add_region(Tier::Sram, 0, 512);
        const auto from = delayed.allocate(region, 8);
        const auto to = delayed.allocate(region, 8);
        delayed.write_local(0, from, encode_values({3.5}));
        Fabric delayed_fabric(delayed, 1, 1, 1);
        require(delayed_fabric.send({{}, 0, 0, 0, 0, from, to}), "delayed send failed");
        delayed_fabric.step_batcher(0);
        delayed_fabric.step_batcher(0);
        for (uint32_t i = 0; i < 10; ++i) {
            delayed_fabric.advance_tick();
            delayed_fabric.deliver_due();
        }
        delayed.release(from);
        rejects(
            [&] {
                delayed.release(to);
            },
            "unpublished destination reclaimed"
        );
        rejects(
            [&] {
                delayed.read_local(0, to);
            },
            "unpublished destination visible"
        );
        for (uint32_t i = 0; i < 10 && !delayed_fabric.has_message(0); ++i)
            delayed_fabric.step_batcher(0);
        require(delayed_fabric.has_message(0), "publication dereferenced released source");
        rejects(
            [&] {
                delayed.release(to);
            },
            "unconsumed destination reclaimed"
        );
        delayed_fabric.receive(0);
        require(decode_values(delayed.read_local(0, to))[0] == 3.5, "delayed payload changed");
        delayed.release(to);

        Program program{"custom_arithmetic", {{InstructionKind::Invoke, 100, 9}, {InstructionKind::Halt, 0, 0}}};
        OperationRegistry operations;
        int result = 0;
        operations.add(100, [&](uint32_t value) {
            result = value * 3;
            return StepResult::Progress;
        });
        Continuation continuation;
        continuation.step(program, operations);
        require(result == 27, "custom operation did not run");
        require(continuation.step(program, operations) == StepResult::Halted, "custom program did not halt");
        require(pack_fp4({0, 1, 2, -6}) == Bytes({0x20, 0xf4, 0, 0}), "FP4 literal encoding differs");
        require(unpack_fp4({0x20, 0xf4, 0, 0}, 4) == Values({0, 1, 2, -6}), "FP4 literal decoding differs");
        rejects(
            [&] {
                unpack_fp4({0}, 2);
            },
            "partial FP4 group accepted"
        );
        std::cout << "Protocol, ownership, capacity, queued leases and programmable entry checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
