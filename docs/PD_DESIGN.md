# Phase 4 design: prefill/decode disaggregation with transfer faults

Pinned to vLLM `0e5b2ba118` (same as [VLLM_INTEGRATION.md](VLLM_INTEGRATION.md)).

## 1. Why this phase

In disaggregated serving, a *prefill* engine computes a prompt's KV cache and
a *decode* engine receives it and generates. The transfer between them is
asynchronous and can fail, arrive late, or race with a client abort. That is
where SGLang's October 2026 isolation bug lived: an aborted request's
in-flight KV write landed in a slot already given to another request.

## 2. How KV moves between engines in vLLM (CPU-friendly path)

vLLM abstracts transfers as **KV connectors** (`KVConnectorBase_V1`). The fast
ones (NIXL, Mooncake) need RDMA hardware. `ExampleConnector`
(`vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`) moves KV
through files and runs on a laptop:

| Side | Hook | What it does |
|---|---|---|
| scheduler | `get_num_new_matched_tokens()` | a request whose `prompt[:-1]` has a saved folder is an external hit for `(len-1)//block*block` tokens |
| worker, decode | `start_load_kv()` (called by the model runner before the forward pass) | reads `<storage>/<hash(tokens)>/<layer>.safetensors` and writes it into this engine's slots via its slot mapping |
| worker, prefill | `save_kv_layer()` (called after attention) | gathers the layer's KV for the request by slot mapping and saves it |

**Integration detail that mattered.** `save_kv_layer()` and
`wait_for_layer_load()` are called by the `maybe_transfer_kv_layer` decorator
(`vllm/model_executor/layers/attention/kv_transfer_utils.py`), which wraps
vLLM's *attention op*. Phantom never calls that op, so as a prefill engine it
would have saved nothing. Phantom's write-and-check step is now a function
decorated with vLLM's own `maybe_transfer_kv_layer`, so it waits for incoming
KV before reading and saves after writing, exactly like a real model.

## 3. What Phantom checks

Tags encode `(pos, token, chain_hash)`, which are independent of which engine or
slot holds them. So after transfer, the decode engine's Phantom re-walks each
context through *its own* block table and checks:

* every transferred slot holds the right position and an unbroken chain;
* the **seam**: the first locally computed token chains from the last
  transferred one;
* every output token equals the reference for `prompt + t1`.

Evidence counter `context_not_computed_here`: positions already present on a
request's first step in the decode engine. With prefix caching off in both
engines, they can only have arrived through the connector.

## 4. Fault injection (Days 13-14)

A `PhantomFaultConnector` subclass of `ExampleConnector`, selected via
`KVTransferConfig(kv_connector=..., kv_connector_module_path=...)`, with
seeded faults:

| Fault | Implementation | Correct engine behaviour |
|---|---|---|
| drop | delete a layer file before load | treat as load failure (recompute, see `kv_load_failure_policy`), never read garbage |
| truncate | save fewer tokens than claimed | detect or recompute |
| swap | serve another prompt's file for this hash | the bug class Phantom must catch (a cross-request leak) |
| delay + abort | load lands after the request was aborted and its blocks reused | the SGLang Oct 2026 bug class |

Each fault is first used as a *planted bug* (Phantom must catch it), then
correct handling in vLLM is checked (Phantom must stay silent).

## 5. Milestones

| Day | Deliverable | Done when |
|---|---|---|
| 12 | Phantom through `maybe_transfer_kv_layer`; `scripts/vllm_pd_check.py` | ✅ 864 transferred positions (exactly the connector's expected count) verified, 12,624 slot reads, 0 violations, all outputs match |
| 13 | `PhantomFaultConnector` with swap + drop | swap caught as a cross-request leak; drop handled |
| 14 | delay + abort mid-transfer | SGLang Oct-2026 bug class reproduced as a plant and caught |
| 15-16 | seeded PD fuzzing + shrinking across both engines | `phantomkv pd-fuzz` |
