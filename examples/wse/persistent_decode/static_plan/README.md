# Static decode plan and KV page quotas

This standalone CPU protocol fixture links the current WSE `Memory/Fabric`
backend. It models bounded generated AICore control flow, shared Bailu KV page
leases, Host renewal and draining without a generic resident AICore runtime.
All simulated physical allocations exist before launch. The payload is integer
arithmetic with an independent per-layer oracle, not full attention or model
weights. The existing numerical fixture remains a separate regression.

## Run

Use a repository-local virtual environment with NumPy and a C++17 compiler
accepting Clang/GCC flags. From the repository root:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install numpy
.venv/bin/python examples/wse/persistent_decode/static_plan/run_validation.py \
  --source-root . --compiler /usr/bin/clang++ --build build/wse-static-plan
```

Windows with LLVM and an installed MSVC/Windows SDK toolchain:

```powershell
python -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install numpy
.venv\Scripts\python.exe examples/wse/persistent_decode/static_plan/run_validation.py `
  --source-root . --compiler 'C:\Program Files\LLVM\bin\clang++.exe' `
  --build build/wse-static-plan
```

The driver rebuilds source files directly; it does not consume an installed
simpler wheel. `finite_trace_input.cpp` supplies an explicitly throwing live
stdin entry for this finite-trace build. Live admission is tested separately
by the parent fixture's `run.sh`.

Summaries and source hashes go to `build/wse-static-plan/summary/`; detailed
outputs, events, prepared layouts and build logs remain under the corresponding
`numeric/`, `protocol/`, `deployment/` and `quota/` subdirectories. Running the
suite does not update tracked source or input files.

## Deployment and execution

`deployment.min.json` is the current v2 deployment contract. Its six top-level
entries contain 19 required leaves. Unknown or duplicate keys are rejected.
The manifest binds the main C++ source, quota header, fixed bounds, layout
recipe and protocol; incompatible contracts are rejected rather than adapted.

| Input | Effect |
| ----- | ------ |
| `format`, `program.manifest`, `program.sha256` | Select and verify the v2 program contract; artifact/header bytes are checked separately |
| `placement.home_core_by_slot` | Four request homes; this backend supports Core 0 or 8 |
| `placement.routed_expert_core` | Eight distinct routed-expert owners |
| `placement.shared_expert_core_by_chip` | One shared expert on each home chip |
| `capacity_bytes.sram_per_core` | Per-Core SRAM budget, checked independently |
| `capacity_bytes.bailu_per_chip` | Budget for each preallocated shared KV pool |
| `capacity_bytes.host_per_chip` | Budget for Host input, output and control buffers |
| `capacity_bytes.batcher_per_endpoint` | Budget for the fixed control allowance; not a complete native descriptor layout |
| `transport.inbox_entries_per_core` | Data notification credits |
| `transport.queue_entries_per_endpoint` | Bounded data send queue |
| `kv_quota.pool_pages_per_chip` | Physical shared cells allocated before launch, at most 32 per chip |
| `kv_quota.initial_pages_per_request` | Initial lease, including one history cell; at most 16 |
| `kv_quota.refill_pages` | Atomic batch renewal increment, clipped to the 16-entry table |
| `kv_quota.low_watermark_pages` | Notify when installed minus claimed pages is at most this value; below the initial lease |
| `kv_quota.exhaustion_policy` | `wait` for released pages or `deny` renewal |
| `kv_quota.control_inbox_entries_per_core` | Independently reserved control notification credits |
| `kv_quota.control_queue_entries_per_endpoint` | Bounded control send queue |

`prepare_deployment.py` checks hashes, roles, budgets and quota constraints,
then emits `prepared_deployment.json` and `deployment_config.h`. Configuration
is compiled into the probe; edits require preparation and rebuild. Handles and
region offsets describe this simulator, not physical WSE addresses.

The fixed program has two chips, eight Cores per chip, four request slots,
40 layers, eight routed experts selected top-6 plus one shared expert, and two
layer banks per slot. A request advances one layer at a time. Each chip owns
a shared physical pool, while each request has a fixed 16-entry page table.
Each cell is 128 bytes. Default deployment creates 524 allocations; the count
is `460 + 2 * pool_pages_per_chip` and remains constant during execution.

| Responsibility | Implementation |
| -------------- | -------------- |
| Validate and expand deployment | `prepare_deployment.py::validate/prepare` |
| Create fixed resources and bind requests | `Probe` constructor and `bind()` |
| Host pool ledger, Low deduplication, Grant and ACK | `Controller::host_step()` |
| Install grants, claim pages, report Low, pause/resume | `Controller::device_step()/claim()` |
| Generated request/expert stages and local token feedback | `Probe::step()/service()` |
| Ordered payload/control transfers and notification credits | Data/control instances of the existing `Fabric` |
| Output consumption and request retirement | `host_consume_outputs()`, `RequestRetire`, `Controller::retire()` |

## Quota protocol and reclamation

Host reserves pages before sending a Grant. Pending grants count against the
shared free pool but cannot be used by the device. Only installation after
delivery, publication and validation increases device `granted`; ACK closes
the Host transaction. One request has at most one grant awaiting ACK.

`claimed` includes the initial history cell and the page reserved for the
token in progress. Low-water is `granted - claimed <= low_watermark_pages`,
with at most one Low per installed grant sequence. Claiming at the hard bound
pauses only that request at `FetchSend`; it cannot access an ungranted page.
Other request and service stages continue. A legal new grant rearms reporting.

Low, Grant, Denied and ACK traverse a separately provisioned bounded control
Fabric over the same Memory model. Epoch, slot, generation and sequence checks
discard stale messages and make repeated grants/events idempotent. An identical
duplicate grant may repeat its ACK; conflicting payloads are rejected.

With `wait`, a request resumes when another runnable owner finishes and returns
pages. A closed trace with only blocked owners and no remaining progress source
is rejected structurally. With `deny`, the request uses its remaining installed
quota and terminates at the first token boundary that needs another page,
recording `capacity_denied` rather than full-generation success.

Retirement waits for data/control transfers, pending authorization ACKs, expert
services and output responsibility to drain before returning every leased page,
including unused pages. Allocations remain; a reused slot gets a new generation.
Default output flow waits for Host ACK per token. `host_silent_burst` separately
validates three generated tokens before the first output ACK using fixed cells.

## Validation and scope

The full driver runs 11 numerical/13 expected rejection cases, 11 static/8
rejection cases, 5 deployment/30 rejection cases, and 13 quota/3 rejection cases.
Quota cases cover refill, inclusive watermark, delayed Host, delayed install,
hard pause, duplicate Low/Grant, old grant sequence, stale generation/epoch,
pool wait/recovery, capacity denial and bounded control congestion. Explicit
event gates establish delayed states and same-Core peer progress.

Every positive quota case checks the integer oracle, grant/claim bounds,
per-generation page ownership and return, and transfer order
`SEND -> COPY_BEGIN -> DELIVER -> PUBLISH -> CONSUMED` in both ID domains.
Physical allocation/release during service is forbidden. Rejection cases cover
closed-trace deadlock, oversized grants and conflicting duplicate grants.

This is a functional model. Admission and retirement use simulator function
boundaries; the Host ledger and portions of fixed control state have CPU shadow
objects. Independent control progress, device memory ordering and full metadata
placement require hardware contracts. Native command/CQ ABI, automatic PyPTO
lowering, full-model static execution, physical pool growth, profile migration,
multi-owner atomic renewal and hardware timing are outside the verified scope.
