# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generate portable external-prefill fixtures with the independent NumPy model."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from reference import LEVELS, Reference

CONFIG = dict(
    layers=40,
    hidden=16,
    latent=8,
    index_dim=4,
    heads=2,
    intermediate=24,
    experts=8,
    top_k=6,
    vocabulary=32,
    window=8,
    index_top_k=3,
)
CONTRACT_FIELDS = tuple(CONFIG)
PROMPTS = {
    "A": [1, 4, 7, 3, 2, 8, 5],
    "B": [2, 6, 3, 9, 1, 5, 7, 4, 8, 2, 11, 3, 12],
    "C1": [3, 8, 2],
    "C2": [7, 2, 1, 9, 5],
}


def checksum(payload):
    value = 14695981039346656037
    for byte in payload:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def envelope(payload):
    return b"WSEPF001" + struct.pack("<IIQ", 1, len(payload), checksum(payload)) + payload


def packed_row(values):
    """Canonical E2M1 nibbles and little-endian int16 group exponents."""
    nibbles, scales = [], []
    for first in range(0, len(values), 4):
        group = values[first : first + 4]
        maximum = float(np.max(np.abs(group)))
        exponent = int(np.ceil(np.log2(maximum / 6))) if maximum else 0
        scales.append(exponent)
        scale = 2.0**exponent
        for value in group:
            distances = abs(LEVELS - abs(value) / scale)
            options = np.flatnonzero(distances == distances.min())
            even = options[options % 2 == 0]
            choice = int(even[-1] if len(even) else options[0])
            nibbles.append(choice | (8 if np.signbit(value) else 0))
    return bytes(nibbles[i] | nibbles[i + 1] << 4 for i in range(0, len(nibbles), 2)) + struct.pack(
        "<" + "h" * len(scales), *scales
    )


def write_snapshot(path, model, next_token):
    fields = [model.c[field] for field in CONTRACT_FIELDS] + [1, len(model.tokens), next_token]
    payload = bytearray(struct.pack("<" + "I" * len(fields), *fields))
    payload.extend(struct.pack("<" + "I" * len(model.tokens), *model.tokens))
    for rows, keys in zip(model.cache, model.keys):
        payload.extend(struct.pack("<I", len(rows)))
        for row, key in zip(rows, keys):
            payload.extend(packed_row(row))
            payload.extend(packed_row(key))
    for window in model.windows:
        payload.extend(struct.pack("<I", len(window)))
        payload.extend(np.asarray(window, dtype="<f8").tobytes())
    for state in model.states:
        payload.extend(np.asarray(state, dtype="<f8").tobytes())
    path.write_bytes(envelope(payload))


def generate(directory, prompts=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"producer": "external_numpy_fixture", "config": CONFIG, "requests": {}}
    for name, tokens in (PROMPTS if prompts is None else prompts).items():
        if (
            not name
            or Path(name).name != name
            or not tokens
            or any(not 0 <= token < CONFIG["vocabulary"] for token in tokens)
        ):
            raise ValueError("invalid snapshot name or prompt")
        model = Reference(CONFIG)
        for token in tokens:
            next_token, _ = model.feed(token)
        path = directory / (name + ".wsepf")
        write_snapshot(path, model, next_token)
        manifest["requests"][name] = {
            "prompt": tokens,
            "next_token": next_token,
            "compressed_rows": [len(rows) for rows in model.cache],
            "file_bytes": path.stat().st_size,
        }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, help="JSON object mapping request names to token lists")
    arguments = parser.parse_args()
    generate(arguments.output, json.loads(arguments.prompts.read_text()) if arguments.prompts else None)
