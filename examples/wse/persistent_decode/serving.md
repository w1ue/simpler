# Live requests and capacity admission

The persistent logical service accepts external prefill snapshots during a
single launch. Its device actors continue decode without a Host launch for each
token. Input submission and close are Host control operations; owner request
state drives admission, KV allocation, transfers and retirement.

## C++ lifecycle

The public interface is in `src/wse/runtime/persistent/decode.h`:

```cpp
DecodeConfig config;
config.max_slots = 16;
config.max_pending = 64;
DecodeSimulation service(config);
service.start();

RequestSpec request{"A", automatic_slot, 0,
                    PrefillState::load("prefill/A.wsepf", config),
                    4, "", 0, 128};
bool accepted = service.submit_request(request);
// A false result leaves the request with the caller for retry.
service.close_input();
while (!service.step()) {}
service.write_report("decode-output");
```

`submit_request()` may be called before `start()` or between `step()` calls.
The API is confined to one service thread; concurrent producers must serialize
calls or use the stdin adapter. Program images and operation registries remain
immutable after `start()`. `add_request()` remains the prelaunch finite-trace
convenience interface; `run()` closes input, starts once, and steps until drain.

`step()` advances the service and returns true after input is closed and every
accepted request and transfer has drained. Open, empty input does not terminate
the service. `waiting_for_input()` allows the caller to block on input readiness
when no device actor is active. `output_token(identity)` retrieves a sampled
token associated with an `OUTPUT_TOKEN` event, including completed generations.

The pending queue holds at most `max_pending` unadmitted requests. Queue-full
submission returns false and does not register the request. Invalid snapshots,
duplicate names, invalid slots, impossible storage contracts and submission
after close throw exceptions. Request names stay unique throughout a recorded
service run. `close_input()` prevents further submissions and drains accepted
work. A finite trace with unresolved admission dependencies fails its progress
guard rather than silently dropping requests.

## Slots and resource contracts

`max_slots` bounds globally concurrent request identities; valid explicit slot
IDs are `[0, max_slots)`. `automatic_slot` selects a free slot on admission.
Occupied explicit slots wait. Generation increments when a request is admitted;
a slot becomes reusable only after the previous generation fully retires.
Each request supplies its fixed `home_chip`, whose home AICore manages its KV.

`RequestSpec::context_limit` is the promised maximum processed length, including
prefill. Zero selects `DecodeConfig::max_context`. Both must cover prefill length
plus the configured decode input count, and the request limit cannot exceed the
service limit. Unused context credits remain available to that request until
retirement; physical KV allocations are created as needed.

For page size P and promised context L, the maximum number of history pages is:

```text
pages(L) = 3 * ceil(floor(L / 2) / P) + ceil(L / P)
```

Admission atomically reserves a vector of aligned byte budgets across all
involved memory regions. The budget covers:

- Home SRAM: request frame, all layer windows, three compressor state rings,
  maximum key pages and KV tails, two gather pages, and bounded expert/lookup
  communication buffers.
- Home Bailu: one history page for every possible logical page at the promised
  context, including partial tails written during final drain.
- Host staging: aligned import KV/key pages, window rings and compressor state.
- Expert/shared/table SRAM: request input and result buffers at every possible
  service owner.

The SRAM bound conservatively allows all possible tails to coexist with their
Bailu copies. It does not assume a measured transfer latency or early tail
reclamation. Its admission count can therefore be lower than a future allocator
with an explicitly bounded tail pipeline. Host import credit is also retained
until retirement, even after the staging allocations are freed.

A request that cannot fit alongside permanent weights and control allocations
is rejected at submission. A request that fits alone but not alongside active
reservations waits with `WAIT_CAPACITY`. An occupied slot produces `WAIT_SLOT`.
Wait events record transitions rather than every service tick. Other eligible
requests can continue; there is no global first-request blockage.

`Memory::try_reserve()` grants all requested regions or none. Every dynamic
request allocation, including buffers allocated by an expert on another core,
charges that request's reservation. Unreserved allocation cannot consume the
unused reserved capacity. Releasing an allocation restores its reservation
credit; retiring a request returns its complete grant after transfers and
service work drain. Release while an allocation or transfer still owns the
resource is rejected.

In reports, `used` is actual allocator occupancy and `reserved` is granted but
currently unused capacity. `committed_peak` measures the peak of `used + reserved`.
All regions must satisfy `used + reserved <= capacity`; final `reserved` is zero.
The logical C++ request/actor objects, directories and verification records are
not a complete device metadata ABI. The service retains historical records for
validation; this executable is not an unbounded production inference server.

## External stdin interface

Start the executable in Linux or WSL:

```bash
build/wse-persistent/wse_decode_sim --serve \
    --max-slots 16 --max-pending 64 --output build/live-decode
```

It emits JSON lines on stdout, starting with `SERVICE_LAUNCH`. A producer writes
one command per line to stdin:

```text
SUBMIT A auto 0 128 4 "build/prefill/A.wsepf"
SUBMIT B 1 1 128 4 "build/prefill/B.wsepf"
CLOSE
```

The fields are request name, explicit slot or `auto`, home chip, context limit,
decode input count, and snapshot path. Names and paths may be quoted. A context
limit of zero selects the service maximum. Commands have a 64 KiB line bound.
Partial lines wait for completion; EOF accepts a final unterminated line and
closes input. `CLOSE` prevents further submissions but does not cancel accepted
requests. The process can drain after `CLOSE` while the producer's pipe stays open.

Input responses are `INPUT_ACCEPTED`, `INPUT_BACKPRESSURE` or `INPUT_REJECTED`.
Acceptance means queued, not yet admitted. Queue backpressure requires caller
retry; rejected requests do not poison the service. Other emitted events include
`ADMIT`, `WAIT_SLOT`, `WAIT_CAPACITY`, `PREFILL_IMPORTED`, `OUTPUT_TOKEN` with its
token ID, `REQUEST_DRAINED`, and `SERVICE_DRAINED`. Full transfer events and
numerical state are saved in the output directory. Human diagnostics use stderr.
The adapter blocks on pipe readiness when idle; it uses no sleep-based polling.

Launch options `--max-context`, `--max-slots`, `--max-pending`, `--sram-bytes`,
`--bailu-bytes` and `--host-bytes` define the resource contract. Defaults are
128 positions, 16 slots, 64 pending requests, 16 MiB/core SRAM, 64 MiB/chip Bailu,
and 128 MiB/chip Host staging. Batcher storage retains its fixed 64 KiB/endpoint
region and 8 KiB control allowance. These are fixture capacities.

## Validation

`bash examples/wse/persistent_decode/run.sh` runs protocol, finite-trace and live
admission CTest entries. The live checks use explicit output/drain events and
the public stepping interface to establish ordering. They cover submission after
decode starts, bounded ingress and retry, capacity/slot waits, generation-safe
reuse, EOF/close, impossible-capacity rejection followed by a valid request,
and numerical execution to all 128 positions with 320 compressed KV rows.
Independent NumPy checks compare tokens, logits, layer checksums, KV/key,
windows and compressor state. Results are in
`build/wse-persistent/live-validation/validation.json`; `stdin/live-output.jsonl`
records the external producer interaction.
