# Request-level persistent decode protocol

This standalone Python fixture models a complete bounded generation request
across serving, Host simpler, fixed Core control flow and transport components.
It checks resource ownership and progress through a persistent session using
a 40-layer integer workload. It does not run the product serving stack, the
simpler Worker, PyPTO code generation or native WSE hardware.

The [component walkthrough](../../../../docs/zh-cn/wse-persistent-decode/request-service.md)
and [memory, transport and validation reference](../../../../docs/zh-cn/wse-persistent-decode/resources-and-validation.md)
describe the executed example in Chinese. The [C++ static-plan fixture](../static_plan/README.md)
uses a separate entry point and retains its own validation results.

## Run

Use Python 3.10 or newer and the repository-local virtual environment. The
fixture itself requires only the standard library and no runtime rebuild.
From the repository root on Linux or WSL:

```bash
.venv/bin/python examples/wse/persistent_decode/request_service/run_checks.py --output build/wse-request-service
.venv/bin/python examples/wse/persistent_decode/request_service/simulate.py --input examples/wse/persistent_decode/request_service/requests.json --scenario normal --output build/wse-request-service-cli
```

From PowerShell:

```powershell
& .venv/Scripts/python.exe examples/wse/persistent_decode/request_service/run_checks.py --output build/wse-request-service
& .venv/Scripts/python.exe examples/wse/persistent_decode/request_service/simulate.py --input examples/wse/persistent_decode/request_service/requests.json --scenario normal --output build/wse-request-service-cli
```

Each scenario writes `results.json`, `events.jsonl` and `memory_layout.json`.
The suite writes a `validation.json` containing source and input SHA-256 hashes.
An assertion failure or protocol violation produces a nonzero exit status.
`run_checks.py` uses the fixed three-request validation trace; `simulate.py`
reads the input file and accepts at most 16 requests within the same profile.

## Components and contract

| Component | Responsibility | Implementation |
| --------- | -------------- | -------------- |
| Serving | Request selection, queuing, cancellation, streamed tokens and terminal response | `model.py: Serving` |
| Model adapter | Supported sampling, context, generation and cache-group contract | `model.py: ModelAdapter` |
| Host simpler | Fixed layout, slot generations, sole physical page allocator, grants, output handoff and drain | `model.py: SimplerHost`, `CachePool`, `Layout` |
| Core program | Bounded phases, all 40 layers, expert work, sampling, local feedback and finite rings | `device.py` |
| Transport | Payload copies, stable sources, publication, credits and shared resource limits | `transport.py` |
| Experiment driver | Session launch/exit, explicit fault gates and result export | `simulate.py` |
| Independent reference and checks | Per-layer integer oracle, protocol and lifetime assertions | `model.py: oracle`, `run_checks.py` |

Serving submits one complete request with a generation limit and stop policy.
It sends no per-token `StepCommand` or physical block IDs. Host simpler is the
single page allocation authority in this proposed variant. This differs from
the existing product step protocol and requires an explicit ownership change
before integration with a serving `KvCacheManager`.

The profile has two logical chips with eight Cores each and two request slots.
Requests A and B start together; C waits and reuses A's slot at generation 2.
One persistent launch generates A=`[61,25]`, B=`[0,59,47,23]` and C=`[39,66]`.
The normal trace executes 320 layers for eight tokens, with 246 memory cells
created before launch and no service-time physical allocation.

Core control flow is handwritten Python representing a bounded compiled
program. It includes a complete token loop, sampling and local token feedback;
it is not a generic device runtime or an executed compiler artifact. All data
paths copy payloads through the memory model. Host output/drain observations
and Core grant installation use explicit transported packets.

## Validation and evidence

| Scenario | Observable condition |
| -------- | -------------------- |
| `normal` | EOS/length termination, grouped page renewal, local latest state, slot reuse and complete drain |
| `slow_output` | Full output ring stalls the next token until Host ownership and credit return |
| `slow_writeback` | Full dirty ring stalls safely until writeback can retire |
| `slow_client` | Device output credit returns before delayed client delivery |
| `cancel_import` | Accepted prefill DMA drains before leases return; cancelled request never activates |
| `cancel_running` | Transported cancellation takes effect at a token boundary; pending output is consumed and discarded |
| `deny_refill` | A denied grant prevents unauthorized context writes and terminates with drain |

Twelve protection checks cover unsupported sampling, generation overflow,
missing groups, submit after input close, allocation after launch, duplicate
request IDs, premature lease return, unstable DMA sources, stale output credit,
replacement of a committed page prefix, exit before drain and atomic grouped
reservation exhaustion. All 7 scenarios and 12 protection checks pass.

The normal case deliberately gates B's first writeback until its second token
commits. It tests local SRAM feedback under delayed external progress and is
not an uncongested performance baseline. Generation completion can precede the
last output packet; serving waits for all emitted outputs and its send queue
before publishing the terminal client response. Request drain, session drain,
kernel exit and unload have separate events.

[Validation](evidence/validation.json), [normal results](evidence/normal_results.json),
[memory layout](evidence/memory_layout.json), [milestones](evidence/milestones.json)
and the [evidence index](evidence/index.json) are checked-in functional reports.
The index identifies full compressed traces in `evidence/cases/` and records
both compressed and original JSONL hashes. Build directories and executables
are not included. Local Git attributes preserve LF endings for source and JSON
files so checked-in source/input hashes survive Windows and Linux checkouts.
Rerun the suite to obtain new uncompressed traces.

## Model limits

- Timing is logical. Same-chip CorePush is visible two ticks after injection;
  queueing and credit waits are separate. External traffic shares modeled
  ingress, Bailu banks and batcher service. These values do not establish TPOT.
- Host-to-Bailu import bypasses batcher. External-to-SRAM and output paths use
  batcher; Core direct Bailu access is excluded from this critical path.
- Each endpoint has four waiting contexts, at most three for data. Background
  output and writeback each allow one job per home endpoint, leaving room for
  critical data as well as control traffic.
- Window/C2A/C1A use source-token coverage of 128/256/128; each page is a 128-byte
  scalar surrogate, not real compressed KV storage or partial-tail publication.
- The arithmetic uses scalar state and `scalar_mod97` sampling, with no real
  weights, attention, position encoding, DSpark or production sampling.
- CPU metadata shadows and declared memory allowances are not a verified native
  wire layout. Multichip startup/exit fences, multi-Host operation, timeout
  quarantine, migration and cross-request collective masks remain outside scope.
