# Transfer backends

TorchSpec moves inference tensors through a backend-neutral contract. Mooncake
is the default; UCCL-P2P is available for fabrics such as HPE Slingshot/CXI
that Mooncake does not support.

## Contract

`TransferBackend` has an explicit producer or consumer lifecycle:

1. `setup(role, device)` initializes the process-local client.
2. A producer calls `put(object_id, tensors)` and then `flush()`.
3. It sends the returned `TransferRef`, not backend-specific keys, through Ray.
4. A consumer calls `get(ref, device)` and, after successful materialization,
   `release(ref)`.
5. Each process calls `close()` during shutdown.

`TransferRef` is immutable, Ray/cloudpickle-safe, and contains tensor shape and
dtype metadata plus an opaque backend locator. It can also contain per-rank
shard references. Controllers use `TransferRef.nbytes` for backpressure without
knowing how a backend stores or addresses bytes.

Mooncake is implemented by `MooncakeTransferBackend`, which adapts the existing
`EagleMooncakeStore`. Legacy `mooncake_key`, shape, and dtype fields remain
accepted and emitted during migration.

## UCCL-P2P

UCCL is a peer-to-peer transport rather than a key/value store. The producer
registers pinned host buffers (the safe default), advertises its endpoint and
memory descriptors in the `TransferRef`, and retains those registrations until
the consumer sends an acknowledgement. A TTL releases abandoned registrations.
The consumer explicitly connects, registers destination buffers, performs an
asynchronous READ, polls completion, and acknowledges only after `get()` has
materialized the tensors.

The adapter intentionally uses only UCCL's low-level `Endpoint` API:
`regv`, `advertisev`, `readv_async`, and `poll_async`. The high-level convenience
API currently assumes verbs registration metadata and is not CXI-safe. Use UCCL
commit `51d91bceabf27be88fc9198cf79b6d4e702bed73` or newer; the Isambard scripts pin
`4312141c93c4b2f3ff7b9f274ad58c355c14fab6`.

Select the backend in YAML or with equivalent command-line overrides:

```yaml
transfer:
  backend: uccl-p2p
uccl:
  transport: cxi
  use_gpu_direct: false
```

The patched SGLang child process receives its selection through
`TORCHSPEC_TRANSFER_BACKEND` and the `UCCL_P2P_*` environment. See
`examples/uccl-isambard/` for the required Isambard Apptainer binds and short
single-/two-node validation jobs.

Current UCCL support covers HF and patched SGLang inference. The vLLM connector
and USP-sharded transfers remain Mooncake-only. GPU-direct CXI registration is
opt-in until it has been validated on Isambard; pinned host staging is the
default.
