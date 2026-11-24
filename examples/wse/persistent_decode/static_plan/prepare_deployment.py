# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate a minimal deployment input and prepare its fixed CPU-simulation layout."""

import argparse
import hashlib
import json
from pathlib import Path

FORMAT = "simpler.wse.deployment.v3"
BOUNDS = dict(
    chips=2,
    cores_per_chip=8,
    cores_per_endpoint=2,
    request_slots=4,
    layers=40,
    routed_experts=8,
    routed_top_k=6,
    layer_banks=2,
    kv_page_table_entries_per_slot=16,
    kv_pool_cells_per_chip=32,
    deferred_output_cells_per_slot=4,
)
STORAGE = dict(
    alignment=128,
    packet_bytes=128,
    request_frame_bytes=88,
    async_frame_bytes=512,
    background_bytes=1024,
    service_frame_bytes=24,
    batcher_control_allowance_bytes=8192,
    quota_frame_bytes=256,
    quota_packet_bytes=256,
)
LAYOUT = {
    "per_slot_home": ["request_frame", "fetch", "ack"],
    "per_slot_async": ["state", "activation_tx", "activation_rx", "retire_tx", "retire_rx", "dirty", "output_ring"],
    "per_background": ["host_import", "bailu_target"],
    "per_slot_host": ["ack"],
    "per_slot_output_cell": ["home_payload", "host_payload"],
    "per_chip_kv_pool_cell": ["bailu_payload"],
    "per_slot_quota": ["device_state", "host_initial_kv"],
    "per_slot_control_lane": ["low_tx", "low_rx", "grant_tx", "grant_rx", "ack_tx", "ack_rx"],
    "per_slot_bank_lane": ["home_send", "home_result"],
    "per_slot_bank_expert": ["expert_input", "expert_output", "service_frame"],
}
PROTOCOL = dict(
    id="static_protocol_memory_hardware_v3",
    kv_initial_u64=5,
    ready="delivery_then_publication",
    input_reuse="last_use_then_related_transfer_retirement",
    request_reuse="outputs_and_transfers_and_services_drained",
    output_ack="bounded_ring_background_consumption",
    kv_grant="install_then_ack",
    kv_low_watermark="claimed_pages_includes_inflight",
    kv_control="shared_batcher_with_progress_reservation",
)


HARDWARE = dict(
    batcher_wait_entries=8,
    noc_queue_entries=32,
    direct_queue_entries=8,
    noc_latency_ticks=2,
    external_latency_ticks=3,
    bailu_bytes_per_tick=64,
    external_bytes_per_tick=64,
)


HARDWARE_MODEL = dict(
    bailu_banks_per_chip=2,
    batcher_actions_per_endpoint_tick=1,
    core_payload_ports=1,
    deterministic_scope="within_chip",
    max_jobs=4096,
    payload_handle_entries=2048,
    dirty_entries_per_slot=4,
)


class DeploymentError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DeploymentError(message)


def keys(value, expected, where):
    require(isinstance(value, dict), where + " must be an object")
    require(set(value) == set(expected), where + ": missing or unknown fields")


def number(value, where, maximum=2**32 - 1):
    require(type(value) is int and 0 < value <= maximum, where + " must be a positive bounded integer")
    return value


def read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON field: " + key)
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError("cannot read JSON: " + str(path)) from error


def reference(base, relative, digest, kind):
    require(isinstance(relative, str) and relative, kind + " path is empty")
    path = (base / relative).resolve()
    require(path.is_file(), kind + " file missing")
    require(
        isinstance(digest, str) and len(digest) == 64 and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
        kind + " digest mismatch",
    )
    return path


def validate(input_path):
    data = read_json(input_path)
    keys(data, ("format", "program", "placement", "capacity_bytes", "transport", "kv_quota", "hardware"), "deployment")
    require(data["format"] == FORMAT, "unsupported deployment format")
    keys(data["program"], ("manifest", "sha256"), "program")
    manifest_path = reference(input_path.parent, data["program"]["manifest"], data["program"]["sha256"], "manifest")
    program = read_json(manifest_path)
    keys(program, ("format", "contract", "artifact", "bounds", "storage", "layout", "protocol", "hardware"), "manifest")
    require(
        program["format"] == "simpler.wse.program.v3" and program["contract"] == "static_protocol_u64_hardware_v3",
        "unsupported program contract",
    )
    require(
        program["bounds"] == BOUNDS
        and program["storage"] == STORAGE
        and program["layout"] == LAYOUT
        and program["protocol"] == PROTOCOL
        and program["hardware"] == HARDWARE_MODEL,
        "unsupported compiled program contract",
    )
    keys(program["artifact"], ("kind", "path", "sha256", "includes"), "artifact")
    require(program["artifact"]["kind"] == "cpu_cpp_source", "unsupported artifact kind")
    artifact = reference(manifest_path.parent, program["artifact"]["path"], program["artifact"]["sha256"], "artifact")
    keys(program["artifact"]["includes"], ("kv_quota.h", "hardware_transport.h"), "artifact.includes")
    for name, digest in program["artifact"]["includes"].items():
        reference(artifact.parent, name, digest, "program header")
    placement = data["placement"]
    keys(placement, ("home_core_by_slot", "routed_expert_core", "shared_expert_core_by_chip"), "placement")
    for key, length in (("home_core_by_slot", 4), ("routed_expert_core", 8), ("shared_expert_core_by_chip", 2)):
        values = placement[key]
        require(
            isinstance(values, list) and len(values) == length and all(type(v) is int and 0 <= v < 16 for v in values),
            key + " shape or Core range invalid",
        )
    homes, routed, shared = (
        placement[name] for name in ("home_core_by_slot", "routed_expert_core", "shared_expert_core_by_chip")
    )
    # Bailu and Host regions in this backend are owned by each chip's home Core.
    require(all(core in (0, 8) for core in homes), "home must be Core 0 or 8 in this backend")
    require(all(shared[chip] // 8 == chip for chip in range(2)), "shared expert must remain on its home chip")
    require(len(set(routed)) == 8 and len(set(shared)) == 2, "expert owner collision")
    require(not set(routed + shared) & {0, 8} and not set(routed) & set(shared), "role owner collision")
    capacities = data["capacity_bytes"]
    keys(capacities, ("sram_per_core", "bailu_per_chip", "host_per_chip", "batcher_per_endpoint"), "capacity_bytes")
    for key, value in capacities.items():
        number(value, key, 2**53 - 1)
        require(value % 128 == 0, key + " must be 128-byte aligned")
    keys(data["transport"], ("inbox_entries_per_core", "queue_entries_per_endpoint"), "transport")
    for key, value in data["transport"].items():
        number(value, key, 65536)
    quota = data["kv_quota"]
    keys(
        quota,
        (
            "pool_pages_per_chip",
            "initial_pages_per_request",
            "refill_pages",
            "low_watermark_pages",
            "exhaustion_policy",
            "control_inbox_entries_per_core",
            "control_queue_entries_per_endpoint",
        ),
        "kv_quota",
    )
    number(quota["pool_pages_per_chip"], "pool_pages_per_chip", 32)
    number(quota["initial_pages_per_request"], "initial_pages_per_request", 16)
    number(quota["refill_pages"], "refill_pages", 16)
    require(
        type(quota["low_watermark_pages"]) is int
        and 0 <= quota["low_watermark_pages"] < quota["initial_pages_per_request"],
        "low_watermark_pages must be below initial grant",
    )
    require(quota["exhaustion_policy"] in ("wait", "deny"), "unsupported exhaustion_policy")
    for key in ("control_inbox_entries_per_core", "control_queue_entries_per_endpoint"):
        number(quota[key], key, 65536)
    require(
        all(
            homes.count(chip * 8) * quota["initial_pages_per_request"] <= quota["pool_pages_per_chip"]
            for chip in range(2)
        ),
        "initial KV grants exceed physical pool",
    )
    keys(data["hardware"], tuple(HARDWARE), "hardware")
    for name, value in data["hardware"].items():
        number(value, name, 65536)
    require(data["hardware"]["batcher_wait_entries"] >= 2, "batcher requires a reserved control context")
    jobs = (
        8
        * (
            data["transport"]["queue_entries_per_endpoint"]
            + quota["control_queue_entries_per_endpoint"]
            + data["hardware"]["noc_queue_entries"]
        )
        + data["hardware"]["direct_queue_entries"]
    )
    require(jobs <= HARDWARE_MODEL["max_jobs"], "transport jobs exceed compiled bound")
    return data, program, artifact


def prepare(input_path, output):
    input_path = Path(input_path).resolve()
    data, program, artifact = validate(input_path)
    p, caps = data["placement"], data["capacity_bytes"]
    regions, cells = [], []

    def region(tier, owner, capacity):
        index = len(regions)
        regions.append(dict(index=index, tier=tier, owner=owner, capacity=capacity, used=0))
        return index

    sram = [region("sram", core, caps["sram_per_core"]) for core in range(16)]
    bailu, host = [], []
    for chip in range(2):
        bailu.append(region("bailu", chip * 8, caps["bailu_per_chip"]))
        host.append(region("host", chip * 8, caps["host_per_chip"]))

    def cell(name, region_index, size):
        r = regions[region_index]
        charged = (size + 127) // 128 * 128
        require(charged <= r["capacity"] - r["used"], f"capacity exceeded: {r['tier']} owner {r['owner']}")
        cells.append(
            dict(
                handle=len(cells) + 1,
                name=name,
                region=region_index,
                offset=r["used"],
                bytes=size,
                charged_bytes=charged,
            )
        )
        r["used"] += charged

    for endpoint in range(8):
        r = region("batcher", endpoint, caps["batcher_per_endpoint"])
        cell(f"batcher.{endpoint}.control_allowance", r, 8192)
    for slot, home in enumerate(p["home_core_by_slot"]):
        chip = home // 8
        prefix = f"slot.{slot}"
        for name, r, size in (
            ("control", sram[home], 88),
            ("fetch", sram[home], 128),
            ("host_ack", host[chip], 128),
            ("ack", sram[home], 128),
        ):
            cell(prefix + "." + name, r, size)
        for index in range(4):
            cell(f"{prefix}.deferred_output.{index}", sram[home], 128)
            cell(f"{prefix}.deferred_host_output.{index}", host[chip], 128)
        for bank in range(2):
            for lane in range(7):
                cell(f"{prefix}.bank.{bank}.send.{lane}", sram[home], 128)
                cell(f"{prefix}.bank.{bank}.result.{lane}", sram[home], 128)
            owners = p["routed_expert_core"] + [p["shared_expert_core_by_chip"][chip]]
            for expert, owner in enumerate(owners):
                for name, size in (("input", 128), ("output", 128), ("control", 24)):
                    cell(f"{prefix}.bank.{bank}.expert.{expert}.{name}", sram[owner], size)

    def quota_cells():
        for chip in range(2):
            for page in range(data["kv_quota"]["pool_pages_per_chip"]):
                cell(f"kv_pool.{chip}.{page}", bailu[chip], 128)
        for slot, home in enumerate(p["home_core_by_slot"]):
            chip = home // 8
            cell(f"slot.{slot}.quota.state", sram[home], 256)
            cell(f"slot.{slot}.quota.initial", host[chip], 128)
            for lane in range(2):
                for name, r in (
                    ("low_tx", sram[home]),
                    ("low_rx", host[chip]),
                    ("grant_tx", host[chip]),
                    ("grant_rx", sram[home]),
                    ("ack_tx", sram[home]),
                    ("ack_rx", host[chip]),
                ):
                    cell(f"slot.{slot}.control.{lane}.{name}", r, 256)

    quota_cells()

    def async_cells():
        for slot, home in enumerate(p["home_core_by_slot"]):
            chip = home // 8
            cell(f"slot.{slot}.async.state", sram[home], 512)
            for name, r in (
                ("activation_tx", host[chip]),
                ("activation_rx", sram[home]),
                ("retire_tx", sram[home]),
                ("retire_rx", host[chip]),
            ):
                cell(f"slot.{slot}.async.{name}", r, 128)
            for index in range(4):
                cell(f"slot.{slot}.dirty.{index}", sram[home], 128)
        cell("background.host", host[0], 1024)
        cell("background.bailu0", bailu[0], 1024)
        cell("background.bailu1", bailu[0], 1024)

    async_cells()
    assert len(cells) == 479 + 2 * data["kv_quota"]["pool_pages_per_chip"]
    identity = dict(data)
    identity["program"] = dict(sha256=data["program"]["sha256"])
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    plan = dict(
        format="simpler.wse.prepared.v3",
        deployment_digest=digest,
        program_contract=program["contract"],
        input=data,
        regions=regions,
        allocations=cells,
        allocation_count=len(cells),
        initial_kv_u64=program["protocol"]["kv_initial_u64"],
        address_semantics="Simulated handles and logical region-relative offsets; not physical WSE addresses",
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "prepared_deployment.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    arrays = {
        "home_cores": p["home_core_by_slot"],
        "routed_cores": p["routed_expert_core"],
        "shared_cores": p["shared_expert_core_by_chip"],
    }
    header = "#pragma once\n#include <array>\n#include <cstdint>\nnamespace deployment {\n"
    header += f'inline constexpr const char *digest = "{digest}";\n'
    for name, values in arrays.items():
        header += f"inline constexpr std::array<uint32_t, {len(values)}> {name} = {{{', '.join(map(str, values))}}};\n"
    for name, value in caps.items():
        header += f"inline constexpr uint64_t {name} = {value}ULL;\n"
    for name, value in data["transport"].items():
        header += f"inline constexpr uint32_t {name} = {value};\n"
    quota_names = dict(
        pool_pages_per_chip="quota_pool_pages",
        initial_pages_per_request="quota_initial_pages",
        refill_pages="quota_refill_pages",
        low_watermark_pages="quota_low_watermark",
        control_inbox_entries_per_core="quota_control_inbox",
        control_queue_entries_per_endpoint="quota_control_queue",
    )
    for name, generated in quota_names.items():
        header += f"inline constexpr uint32_t {generated} = {data['kv_quota'][name]};\n"
    deny = "true" if data["kv_quota"]["exhaustion_policy"] == "deny" else "false"
    header += f"inline constexpr bool quota_deny_on_exhaustion = {deny};\n"
    for name, value in data["hardware"].items():
        header += f"inline constexpr uint32_t {name} = {value};\n"
    header += "}\n"
    (output / "deployment_config.h").write_text(header, encoding="utf-8")
    return plan, artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan, _ = prepare(args.input, args.output)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "allocation_count": plan["allocation_count"],
                    "deployment_digest": plan["deployment_digest"],
                }
            )
        )
    except DeploymentError as error:
        parser.exit(1, "deployment rejected: " + str(error) + "\n")
