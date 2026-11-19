# External prefill snapshot contract

The WSE logical simulator accepts completed prefill state. It performs no
prompt-layer computation. `prefill.py` is the external numerical fixture
producer; `PrefillState::load()` is the C++ file boundary, and operation 20 is
the device-side import state machine. There is no fallback to prompt replay.

## Semantic state and placement

For processed length N, the state contains the N prompt token IDs, the first
sampled successor, and these payloads:

| State | Logical shape | Imported destination |
| --- | --- | --- |
| Compressed KV sources 2/8/14 | floor(N/2) rows each | Full pages in home Bailu, partial tail in home SRAM |
| Compressed KV source 20 | N rows | Full pages in home Bailu, partial tail in home SRAM |
| Index keys for those four sources | One key per compressed row | Owner SRAM |
| All layer windows | Last min(N, window) rows, chronological order | Owner SRAM ring, indexed by absolute position modulo window |
| Three C2A state rings | Eight physical slots, each projected latent followed by gate | Owner SRAM |
| Prompt token history | N uint32 IDs | Request metadata, used by Engram lookup |
| Next token and N | Scalars | Request input and next absolute position |

The odd-length C2A prefix state is necessary: the first decode input may complete
a pair begun by the final prompt position. KV/key alone cannot resume that case.
Candidate lists and per-layer hidden activations are recomputed for each decode
input, so they are not transferred. The next token has not been processed yet
and has no KV row in the snapshot.

Snapshots contain no physical addresses, page-pool leases, slot IDs or request
generations. The receiver validates the model contract, reserves allocations,
repackages rows into its configured page size and assigns the generation on
admission. Page size and maximum context are receiver policy; N plus requested
decode input count must fit its maximum context.

All numerical payload buffers are staged in owner-associated simulated Host
regions before the logical service launch. A later-admitted request retains its
Host payload until admission. Operation 20 issues bounded batcher jobs, waits
for every completion, and then records `PREFILL_IMPORTED`. Every chunk obeys
credit -> DMA -> delivery -> publication -> consume. Decode embedding rejects
execution before that completion point. Import buffers are reclaimed after
consumption; full imported KV pages remain read-only in Bailu, and the imported
partial tail can accept new decode rows. Request retirement drains and reclaims
both imported and newly generated storage.

## Binary format

Version 1 is little-endian. It is specific to the scaled synthetic-weight FP64 /
FP4 fixture; it is not a DeepSeek checkpoint interoperability format. The loader
caps a snapshot file at 64 MiB. The envelope is:

| Offset | Type | Field |
| --- | --- | --- |
| 0 | 8 bytes | ASCII `WSEPF001` |
| 8 | uint32 | Version, currently 1 |
| 12 | uint32 | Payload byte length |
| 16 | uint64 | FNV-1a-64 of the payload |
| 24 | bytes | Payload |

The checksum starts at 14695981039346656037; each byte updates
`hash = (hash XOR byte) * 1099511628211 modulo 2^64`.

Payload fields, in order:

1. Twelve uint32 model-contract words: layers, hidden, latent, index dimension,
   heads, intermediate, experts, top-k, vocabulary, window, index top-k, and
   synthetic weight/formula revision (1).
2. Processed length N and next token, both uint32, followed by N uint32 prompt IDs.
3. Four source families in order 2/8/14/20. Each starts with uint32 row count.
   Each row contains packed KV bytes followed by packed key bytes.
4. One window per layer. Each starts with uint32 valid-row count, then
   `count * latent` float64 values in chronological row-major order.
5. Three C2A rings, each `8 * 2 * latent` float64 values in physical ring order.

A packed row of width D contains D/2 nibble bytes, low nibble first, followed
by D/4 little-endian int16 exponents. Each nibble has a sign bit and an E2M1
magnitude code from `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`; its four-element group uses
scale `2^exponent`. Thus each row consumes D bytes in this fixture format.
Dense windows and C2A state use IEEE float64. Missing or trailing fields,
incompatible contracts, invalid counts/tokens, nonfinite state and damaged
envelopes are errors. Failed loading occurs before service launch.

## Generate and consume

```bash
.venv/bin/python examples/wse/persistent_decode/prefill.py --output build/prefill
build/wse-persistent/wse_decode_sim --prefill-dir build/prefill --multi --output build/decode
```

The default producer writes A/B/C1/C2 snapshots. `--prompts prompts.json` accepts
a JSON object mapping names to token lists. The example executable binds A to
slot 0, B to slot 1, C1 to slot 2 and C2 to reused slot 0; custom applications
can use `PrefillState::load()` and `RequestSpec` directly.

The producer and numerical oracle share the independent NumPy model. The C++
decoder runs its own operators and transport. Verification compares continuous
NumPy prefill+decode with snapshot-imported C++ decode, including tokens,
decode-layer checksums, final KV/key, windows and compressor states. The custom
AICore head transform applies only after import, during decode.

All WSE layer checksums and newly committed row counters cover decode only.
Total fabric traffic includes initial import; `prefill_import_bytes` separates
that cost from `decode_bytes_transferred`. External prefill computation time
is not included. These byte counters cover numerical payloads; scalar arguments,
token-history metadata and request descriptors are part of the preloaded control
trace and are not charged as fabric packets. Their fixed scratch allowance is
not a complete physical metadata capacity model. Logical ticks still have no
calibrated hardware-time meaning.
