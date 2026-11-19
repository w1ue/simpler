# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Independent dense NumPy oracle for the scaled structural fixture."""

import argparse
import json
from functools import cache
from pathlib import Path

import numpy as np

LEVELS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
SOURCES = (2, 8, 14, 20)


def norm(x):
    return x / np.sqrt(np.mean(x * x) + 1e-6)


def softmax(x, axis=-1):
    result = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return result / np.sum(result, axis=axis, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rope(x, position, theta=10000.0):
    result = x.copy()
    width = min(4, len(x))
    angle = position / theta ** (np.arange(width // 2) * 2 / width)
    a, b = x[: width // 2], x[width // 2 : width]
    result[: width // 2] = a * np.cos(angle) - b * np.sin(angle)
    result[width // 2 : width] = a * np.sin(angle) + b * np.cos(angle)
    return result


def quantize(x):
    result = np.empty_like(x)
    for offset in range(0, len(x), 4):
        group = x[offset : offset + 4]
        maximum = np.max(np.abs(group))
        scale = 2.0 ** np.ceil(np.log2(maximum / 6)) if maximum else 1.0
        for index, value in enumerate(group):
            error = np.abs(LEVELS - abs(value) / scale)
            candidates = np.flatnonzero(error == error.min())
            even = candidates[candidates % 2 == 0]
            chosen = even[-1] if len(even) else candidates[0]
            result[offset + index] = np.copysign(LEVELS[chosen] * scale, value)
    return result


@cache
def weight(rows, cols, seed):
    index = np.arange(rows * cols, dtype=np.float64) + 1
    return (np.sin((seed + 1) * 0.137 + index * 1.731) * 0.35 / np.sqrt(rows)).reshape(rows, cols)


def hc_mix(streams, matrix):
    projection = streams.reshape(-1) @ matrix / np.sqrt(np.mean(streams * streams) + 1e-6)
    pre = sigmoid(projection[:4]) + 1e-6
    post = 2 * sigmoid(projection[4:8])
    residual = softmax(projection[8:].reshape(4, 4)) + 1e-6
    residual /= residual.sum(axis=0, keepdims=True) + 1e-6
    for _ in range(19):
        residual /= residual.sum(axis=1, keepdims=True) + 1e-6
        residual /= residual.sum(axis=0, keepdims=True) + 1e-6
    return pre, post, residual


def collapse(streams, pre):
    return (streams[0] * pre[0] + streams[1] * pre[1]) + (streams[2] * pre[2] + streams[3] * pre[3])


def expand(update, streams, mix):
    result = np.empty_like(streams)
    for target in range(4):
        value = update * mix[1][target]
        for source in range(4):
            value = value + mix[2][source, target] * streams[source]
        result[target] = value
    return result


class Reference:
    def __init__(self, config, custom_head=False):
        self.c = config
        self.custom_head = custom_head
        self.windows = [[] for _ in range(40)]
        self.cache = [[] for _ in SOURCES]
        self.keys = [[] for _ in SOURCES]
        self.previous = [None] * 3
        self.states = [np.zeros((8, 2 * config["latent"])) for _ in range(3)]
        self.tokens = []
        self.checksums = []

    def engram(self):
        result = np.zeros(self.c["hidden"])
        partials = [np.zeros_like(result), np.zeros_like(result)]
        for ngram in range(2, 5):
            for head in range(8):
                key = 2166136261 ^ (head + ngram * 17)
                for n in range(ngram):
                    token = self.tokens[-1 - n] if len(self.tokens) > n else 2
                    key = ((key ^ token) * 16777619) & 0xFFFFFFFF
                row = key % 128
                partials[row % 2] += weight(64, self.c["hidden"], 3 + row % 2)[row // 2] / 24
        return partials[0] + partials[1]

    def append_history(self, x, base, source, position):
        d, latent, index = self.c["hidden"], self.c["latent"], self.c["index_dim"]
        projected = x @ weight(d, latent, base + 5)
        publish = source == 3
        if source != 3:
            gate = x @ weight(d, latent, base + 6)
            if position % 2:
                previous, previous_gate = self.previous[source]
                probability = sigmoid(previous_gate - gate)
                pooled = previous * probability + projected * (1 - probability)
                publish = True
            else:
                pooled = projected
            self.previous[source] = (projected, gate)
            self.states[source][position % 8] = np.concatenate((projected, gate))
            projected = pooled
        if publish:
            value = norm(projected)
            self.cache[source].append(quantize(rope(value, len(self.cache[source]), 160000.0)))
            self.keys[source].append(quantize(norm(value @ weight(latent, index, base + 7))))

    def feed(self, token):
        c = self.c
        d, latent, index = c["hidden"], c["latent"], c["index_dim"]
        position = len(self.tokens)
        self.tokens.append(token)
        streams = np.tile(weight(c["vocabulary"], d, 1)[token], (4, 1))
        selected, candidates = [], []
        for layer in range(40):
            base = 100 + layer * 40
            if layer in (1, 14):
                streams += self.engram()[None, :]
            residual = streams.copy()
            mix = hc_mix(streams, weight(4 * d, 24, base))
            x = norm(collapse(streams, mix[0]))
            q = norm(x @ weight(d, latent, base + 2)) @ weight(latent, c["heads"] * latent, base + 3)
            window = rope(norm(x @ weight(d, latent, base + 4)), position)
            self.windows[layer].append(window)
            if len(self.windows[layer]) > c["window"]:
                self.windows[layer].pop(0)
            source = 4 if layer < 2 else 3 if layer >= 20 else 2 if layer >= 14 else 1 if layer >= 8 else 0
            if layer in SOURCES:
                self.append_history(x, base, source, position)
            if layer < 2:
                selected = []
            elif layer in SOURCES or (layer >= 24 and layer % 4 == 0):
                query = norm(x @ weight(d, index, base + 8))
                rows = list(range(len(self.keys[source]))) if layer in SOURCES else candidates
                scores = sorted((-float(query @ self.keys[source][row]), row) for row in rows)
                selected = [row for _, row in scores[: c["index_top_k"]]]
                if layer == 20:
                    blocks = []
                    for _, row in scores:
                        if row // 2 not in blocks and len(blocks) < 4:
                            blocks.append(row // 2)
                    candidates = [row for row in range(len(self.keys[source])) if row // 2 in blocks]
            outputs = []
            for head in range(c["heads"]):
                query = q[head * latent : (head + 1) * latent]
                window_query = rope(query, position)
                compressed_query = rope(query, position if source == 3 else position // 2, 160000.0)
                rows = list(self.windows[layer]) + [self.cache[source][row] for row in selected]
                scores = [window_query @ row / np.sqrt(latent) for row in self.windows[layer]]
                scores += [compressed_query @ self.cache[source][row] / np.sqrt(latent) for row in selected]
                probabilities = softmax(np.array(scores))
                result = np.zeros(latent)
                for probability, row in zip(probabilities, rows):
                    result += probability * row
                outputs.extend(result)
            update = np.array(outputs) @ weight(c["heads"] * latent, d, base + 9)
            streams = expand(update, residual, mix)
            residual = streams.copy()
            mix = hc_mix(streams, weight(4 * d, 24, base + 1))
            x = norm(collapse(streams, mix[0]))
            scores = np.sqrt(np.log1p(np.exp(x @ weight(d, 8, base + 10))))
            routes = sorted(range(8), key=lambda expert: (-scores[expert], expert))[:6]
            route_weights = scores[routes] * (1.5 / sum(scores[routes]))

            def expert(a_seed, b_seed, c_seed):
                a = x @ weight(d, c["intermediate"], a_seed)
                b = x @ weight(d, c["intermediate"], b_seed)
                return (np.clip(a, -10, 10) * sigmoid(a) * b) @ weight(c["intermediate"], d, c_seed)

            update = expert(base + 11, base + 12, base + 13)
            for lane, routed_expert in enumerate(routes):
                seed = 10000 + layer * 100 + routed_expert * 3
                update += route_weights[lane] * expert(seed, seed + 1, seed + 2)
            streams = expand(update, residual, mix)
            self.checksums.append(float(np.sum(streams.reshape(-1) * (np.arange(4 * d) + 1))))
        if self.custom_head:
            streams += ((np.arange(4 * d) % 5 - 2) * 0.25).reshape(4, d)
        head = norm(np.sum(streams / 4, axis=0))
        logits = head @ weight(d, c["vocabulary"], 2)
        return int(np.argmax(logits)), logits


def verify(directory):
    path = Path(directory)
    result = json.loads((path / "results.json").read_text())
    program = json.loads((path / "programs.json").read_text())
    custom = any(instruction[1] == 100 for instruction in program["aicore"]["instructions"])
    errors = dict(logits=0.0, layer_checksums=0.0, cache_rows=0.0, key_rows=0.0, windows=0.0, compressor_states=0.0)
    tokens_checked = 0
    for request in result["requests"]:
        reference = Reference(result["config"])
        for token in request["prompt"]:
            next_token, _ = reference.feed(token)
        assert next_token == request["prefill_next_token"], (request["name"], "prefill token mismatch")
        reference.custom_head = custom
        reference.checksums.clear()
        for token, logits in zip(request["generated"], request["logits"]):
            next_token, golden_logits = reference.feed(next_token)
            assert next_token == token, (request["name"], "decode token mismatch")
            errors["logits"] = max(errors["logits"], float(np.max(np.abs(golden_logits - logits))))
            tokens_checked += 1
        checksum_error = float(np.max(np.abs(np.array(reference.checksums) - request["layer_checksums"])))
        errors["layer_checksums"] = max(errors["layer_checksums"], checksum_error)
        for source in range(4):
            golden = np.array(reference.cache[source])
            actual = np.array(request["cache_rows"][source])
            assert golden.shape == actual.shape, (request["name"], source, "KV row count mismatch")
            if golden.size:
                errors["cache_rows"] = max(errors["cache_rows"], float(np.max(np.abs(golden - actual))))
            golden_keys = np.array(reference.keys[source])
            actual_keys = np.array(request["key_rows"][source])
            assert golden_keys.shape == actual_keys.shape
            if golden_keys.size:
                errors["key_rows"] = max(errors["key_rows"], float(np.max(np.abs(golden_keys - actual_keys))))
        window_error = float(np.max(np.abs(np.array(reference.windows) - request["windows"])))
        state_error = float(np.max(np.abs(np.array(reference.states).reshape(3, -1) - request["compressor_states"])))
        errors["windows"] = max(errors["windows"], window_error)
        errors["compressor_states"] = max(errors["compressor_states"], state_error)
        assert request["processed_length"] == len(request["prompt"]) + len(request["generated"])
    assert max(errors.values()) < 1e-6 and errors["cache_rows"] == 0 and errors["key_rows"] == 0, errors
    assert result["launches"] == 1 and result["all_drained"]
    checks = {
        "independent_numpy_oracle": "passed",
        "tokens_checked": tokens_checked,
        "max_absolute_error": errors,
        "full_checkpoint_validated": False,
        "hardware_timing_validated": False,
    }
    (path / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    verify(parser.parse_args().directory)
