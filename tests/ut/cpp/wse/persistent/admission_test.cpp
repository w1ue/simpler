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
#include <iostream>
#include <stdexcept>

using namespace wse::persistent;
namespace {
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
template <typename Function>
void rejects(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "invalid lifecycle or request accepted");
}
void drain(DecodeSimulation &simulation) {
    for (uint32_t i = 0; i < 1000000; ++i)
        if (simulation.step()) return;
    throw std::runtime_error("service failed to drain");
}
}  // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::invalid_argument("expected prefill and output directories");
        DecodeConfig config;
        config.max_slots = 3;
        config.max_pending = 2;
        config.bailu_bytes = 10240;
        const auto spec = [&](const std::string &name, uint32_t slot, uint32_t home, uint32_t count,
                              uint32_t limit = 0) {
            return RequestSpec{name,  slot,
                               home,  PrefillState::load(std::filesystem::path(argv[1]) / (name + ".wsepf"), config),
                               count, "",
                               0,     limit};
        };
        DecodeSimulation simulation(config);
        rejects([&] {
            simulation.step();
        });
        simulation.start();
        require(simulation.waiting_for_input(), "empty service did not wait for live input");
        require(simulation.submit_request(spec("A", automatic_slot, 0, 4)), "live A not accepted");
        bool output = false;
        for (uint32_t i = 0; i < 1000000 && !output; ++i) {
            require(!simulation.step(), "open input drained early");
            for (const auto &event : simulation.fabric().events()) {
                if (event.event == "OUTPUT_TOKEN") {
                    require(simulation.output_token(event.key) == 21, "live output token changed");
                    output = true;
                    break;
                }
            }
        }
        require(output, "first decode output missing");
        require(simulation.submit_request(spec("C1", 2, 0, 2)), "live C1 not accepted");
        require(simulation.submit_request(spec("B", automatic_slot, 1, 4)), "live B not accepted");
        require(!simulation.submit_request(spec("C2", 0, 0, 2)), "bounded ingress did not backpressure");
        simulation.step();
        require(simulation.submit_request(spec("C2", 0, 0, 2)), "ingress retry after dequeue failed");
        rejects([&] {
            simulation.submit_request(spec("C1", 2, 0, 2));
        });
        rejects([&] {
            simulation.submit_request(spec("A", 3, 0, 4));
        });
        simulation.close_input();
        rejects([&] {
            simulation.submit_request(spec("A", 0, 0, 4));
        });
        drain(simulation);
        require(simulation.completed_requests() == 4, "accepted request lost");
        bool capacity_wait = false, slot_wait = false;
        for (const auto &event : simulation.fabric().events()) {
            capacity_wait |= event.event == "WAIT_CAPACITY";
            slot_wait |= event.event == "WAIT_SLOT";
        }
        require(capacity_wait && slot_wait, "capacity or slot admission did not wait");
        for (const auto &region : simulation.memory().regions()) {
            require(region.reserved == 0, "reservation credit leaked");
            require(region.committed_peak <= region.capacity, "capacity promised twice");
        }
        simulation.write_report(std::filesystem::path(argv[2]) / "api");

        config.bailu_bytes = 1024;
        DecodeSimulation constrained(config);
        rejects([&] {
            constrained.submit_request(spec("A", 0, 0, 4));
        });
        require(constrained.submit_request(spec("C1", automatic_slot, 0, 2, 5)), "short-context contract rejected");
        constrained.run();
        constrained.write_report(std::filesystem::path(argv[2]) / "short_contract");
        require(constrained.completed_requests() == 1, "rejected request poisoned service");
        std::cout << "Live submission, bounded ingress, capacity wait, context guarantees and slot reuse passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
