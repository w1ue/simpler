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
#include "live_input.h"

#include <cerrno>
#include <iomanip>
#include <iostream>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

using namespace wse::persistent;
namespace {
std::string quote(const std::string &value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') output << '\\' << ch;
        else if (ch < 32) output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(ch);
        else output << ch;
    }
    output << '"';
    return output.str();
}
uint32_t number(const std::string &text) {
    size_t end = 0;
    if (text.empty() || text[0] == '-') throw std::invalid_argument("invalid unsigned field");
    const auto value = std::stoull(text, &end);
    if (end != text.size() || value > UINT32_MAX) throw std::invalid_argument("unsigned field outside uint32");
    return value;
}
void command(DecodeSimulation &simulation, const DecodeConfig &config, const std::string &line, bool &closed) {
    std::istringstream input(line);
    std::string operation, name;
    input >> operation;
    if (operation.empty()) return;
    try {
        if (closed) throw std::invalid_argument("request input is closed");
        if (operation == "CLOSE") {
            std::string extra;
            if (input >> extra) throw std::invalid_argument("unexpected CLOSE fields");
            simulation.close_input();
            closed = true;
            std::cout << "{\"event\":\"INPUT_CLOSED\"}\n" << std::flush;
            return;
        }
        std::string slot, home, limit, count, path, extra;
        if (operation != "SUBMIT" ||
            !(input >> std::quoted(name) >> slot >> home >> limit >> count >> std::quoted(path)) || (input >> extra))
            throw std::invalid_argument(
                "expected SUBMIT name slot|auto home context_limit decode_inputs quoted_snapshot_path"
            );
        RequestSpec request{
            name,
            slot == "auto" ? automatic_slot : number(slot),
            number(home),
            PrefillState::load(path, config),
            number(count),
            "",
            0,
            number(limit)
        };
        const auto accepted = simulation.submit_request(std::move(request));
        std::cout << "{\"event\":" << quote(accepted ? "INPUT_ACCEPTED" : "INPUT_BACKPRESSURE")
                  << ",\"name\":" << quote(name) << "}\n"
                  << std::flush;
    } catch (const std::exception &error) {
        std::cout << "{\"event\":\"INPUT_REJECTED\",\"name\":" << quote(name) << ",\"reason\":" << quote(error.what())
                  << "}\n"
                  << std::flush;
    }
}
void publish(DecodeSimulation &simulation, size_t &cursor) {
    const auto &events = simulation.fabric().events();
    while (cursor < events.size()) {
        const auto &event = events[cursor++];
        if (event.event != "SERVICE_LAUNCH" && event.event != "ADMIT" && event.event != "WAIT_SLOT" &&
            event.event != "WAIT_CAPACITY" && event.event != "PREFILL_IMPORTED" && event.event != "OUTPUT_TOKEN" &&
            event.event != "REQUEST_DRAINED" && event.event != "SERVICE_DRAINED")
            continue;
        std::cout << "{\"event\":" << quote(event.event) << ",\"tick\":" << event.tick << ",\"slot\":" << event.key.slot
                  << ",\"generation\":" << event.key.generation << ",\"position\":" << event.key.position;
        if (event.event == "OUTPUT_TOKEN") std::cout << ",\"token\":" << simulation.output_token(event.key);
        std::cout << "}\n" << std::flush;
    }
}
}  // namespace

void serve_input(DecodeSimulation &simulation, const DecodeConfig &config) {
    simulation.start();
    size_t cursor = 0;
    bool closed = false, eof = false;
    std::string buffered;
    publish(simulation, cursor);
    for (;;) {
        if (!eof) {
            pollfd descriptor{STDIN_FILENO, POLLIN, 0};
            const auto status = poll(&descriptor, 1, simulation.waiting_for_input() ? -1 : 0);
            if (status < 0 && errno != EINTR) throw std::runtime_error("request input poll failed");
            if (status > 0) {
                char bytes[4096];
                const auto count = read(STDIN_FILENO, bytes, sizeof(bytes));
                if (count < 0 && errno != EINTR) throw std::runtime_error("request input read failed");
                if (count > 0) buffered.append(bytes, size_t(count));
                if (count == 0) eof = true;
            }
        }
        size_t newline;
        while ((newline = buffered.find('\n')) != std::string::npos) {
            if (newline > 65536) throw std::runtime_error("request input line exceeds 64 KiB");
            const auto line = buffered.substr(0, newline);
            buffered.erase(0, newline + 1);
            command(simulation, config, line, closed);
        }
        if (buffered.size() > 65536) throw std::runtime_error("request input line exceeds 64 KiB");
        if (eof && !buffered.empty()) {
            command(simulation, config, buffered, closed);
            buffered.clear();
        }
        if (eof && !closed) {
            simulation.close_input();
            closed = true;
        }
        const auto done = simulation.step();
        publish(simulation, cursor);
        if (done) break;
    }
}
