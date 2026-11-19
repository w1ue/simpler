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
#include <string>

using wse::persistent::DecodeConfig;
using wse::persistent::DecodeSimulation;
using wse::persistent::InstructionKind;

int main(int argc, char **argv) {
    try {
        DecodeConfig config;
        bool multi = false, custom = false, custom_batcher = false, invalid = false, invalid_service = false;
        std::string output = "outputs/wse_decode";
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--multi") multi = true;
            else if (argument == "--single") multi = false;
            else if (argument == "--reverse-batcher") config.reverse_batcher = true;
            else if (argument == "--custom-core") custom = true;
            else if (argument == "--custom-batcher") custom_batcher = true;
            else if (argument == "--invalid-batcher") invalid = true;
            else if (argument == "--invalid-service") invalid_service = true;
            else if (argument == "--max-context" && i + 1 < argc) config.max_context = std::stoul(argv[++i]);
            else if (argument == "--inbox-capacity" && i + 1 < argc) config.inbox_capacity = std::stoul(argv[++i]);
            else if (argument == "--cancel") config.cancel_long_request = true;
            else if (argument == "--output" && i + 1 < argc) output = argv[++i];
            else throw std::invalid_argument("unknown or incomplete argument: " + argument);
        }
        DecodeSimulation simulation(config);
        if (custom_batcher) {
            simulation.register_batcher_operation(100, [](const auto &message, uint32_t) {
                if (!message.key.generation) throw std::runtime_error("unversioned batcher packet");
                return wse::persistent::StepResult::Progress;
            });
            simulation.set_batcher_program(
                {"version_checked_push",
                 {{InstructionKind::Invoke, 100, 0},
                  {InstructionKind::Invoke, 1, 0},
                  {InstructionKind::Invoke, 2, 0},
                  {InstructionKind::Invoke, 3, 0},
                  {InstructionKind::Invoke, 4, 0},
                  {InstructionKind::Halt, 0, 0}}}
            );
        }
        if (custom) {
            auto program = wse::persistent::default_core_program();
            program.name = "decode_with_programmed_head_transform";
            program.code.insert(program.code.begin() + 9, {InstructionKind::Invoke, 100, 0});
            simulation.register_core_operation(100, [](auto &values) {
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] += (static_cast<int>(i % 5) - 2) * 0.25;
            });
            simulation.set_core_program(std::move(program));
        }
        if (invalid) {
            simulation.set_batcher_program(
                {"invalid_publish_before_copy",
                 {{InstructionKind::Invoke, 1, 0}, {InstructionKind::Invoke, 4, 0}, {InstructionKind::Halt, 0, 0}}}
            );
        }
        if (invalid_service)
            simulation.set_service_program(
                {"invalid_send_before_compute", {{InstructionKind::Invoke, 31, 0}, {InstructionKind::Halt, 0, 0}}}
            );
        simulation.add_request({"A", 0, 0, {1, 4, 7, 3, 2, 8, 5}, 4, "", 0});
        if (multi) {
            simulation.add_request({"B", 1, 1, {2, 6, 3, 9, 1, 5, 7, 4, 8, 2, 11, 3, 12}, 4, "", 0});
            simulation.add_request({"C1", 2, 0, {3, 8, 2}, 2, "A", 2});
            simulation.add_request({"C2", 0, 0, {7, 2, 1, 9, 5}, 2, "A", 0});
        }
        simulation.run();
        simulation.write_report(output);
        std::cout << "Completed " << simulation.completed_requests() << " requests; report: " << output << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "WSE decode simulation failed: " << error.what() << '\n';
        return 1;
    }
}
