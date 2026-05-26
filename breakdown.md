# IBGDA RDMA Write Latency Breakdown Plan

This note describes the code changes needed to break down the latency measured by
`perftest/device/pt-to-pt/shmem_put_latency.cu` on the IBGDA path.

The target interval is:

```text
GPU issues put/quiet
  -> GPU writes WQE
  -> GPU writes DBR / BlueFlame doorbell
  -> NIC observes doorbell and fetches/decodes WQE
  -> NIC DMA reads local GPU payload
  -> packets go over link
  -> remote NIC writes remote GPU memory and sends ACK
  -> local NIC generates requester CQE
  -> GPU polling thread observes CQE
```

The important boundary is that ConnectX does not expose per-WQE firmware-stage
timestamps. The reliable way to proceed is to add timestamps around observable
boundaries, then use controlled experiments to subtract common terms.

## What Already Exists

The branch already has useful profiling under `NVSHMEM_IBGDA_LAT_PROFILE`.

Files:

- `src/include/host/nvshmemx_ibgda_lat_profile.h`
- `src/device/init/init_device.cu`
- `src/include/non_abi/device/pt-to-pt/ibgda_device.cuh`
- `src/modules/transport/ibgda/ibgda.cpp`
- `perftest/device/pt-to-pt/shmem_put_latency.cu`

Current measurements:

```text
wqe_prep:
  ibgda_rma_thread entry -> just before ibgda_submit_requests()

doorbell_submit:
  just before ibgda_submit_requests() -> return from ibgda_submit_requests()

nic_cqe_to_gpu:
  CQE timestamp -> GPU thread observes the CQE in ibgda_poll_cq()

gpu_db_store:
  immediately before ibgda_ring_db() -> immediately after ibgda_ring_db()

db_to_cqe:
  GPU doorbell-store boundary -> requester CQE generation, matched by
  prod_idx/completed_idx
```

`doorbell_submit` is still a GPU-side submit-path time. It includes GPU fences,
`ready_head` CAS/spin, DBR write, and the BlueFlame MMIO store. It is not the
NIC firmware doorbell-processing time. Use `gpu_db_store` and `db_to_cqe` for
the finer split around the real doorbell boundary.

## Print New Metrics in shmem_put_latency.cu

Edit `perftest/device/pt-to-pt/shmem_put_latency.cu`.

The file already calibrates GPU `%globaltimer` to the mlx5 raw clock domain.
Reuse:

```c
map_gpu_globaltimer_to_nic_ns()
calibrate_gpu_globaltimer_to_nic_ns()
nvshmemx_ibgda_query_mlx5_raw_clock_ns()
```

Before each measured run:

```c
nvshmemx_ibgda_lat_profile_reset();
nvshmemx_ibgda_lat_profile_cqe_ts_reset();
nvshmemx_ibgda_lat_profile_db_ts_reset();
```

After the measured run:

```c
nvshmemx_ibgda_lat_profile_db_ts_get(db_pairs, iter, &db_pair_count);
nvshmemx_ibgda_lat_profile_cqe_ts_get(cqe_pairs, iter, &cqe_pair_count);
```

For each matched pair:

```text
db_before_nic_ns = map_gpu_globaltimer_to_nic_ns(calib, db_before_gpu_ns)
db_after_nic_ns  = map_gpu_globaltimer_to_nic_ns(calib, db_after_gpu_ns)
cqe_nic_ns       = decoded CQE timestamp in the same mlx5 raw-clock ns domain
gpu_seen_nic_ns  = map_gpu_globaltimer_to_nic_ns(calib, cqe_gpu_now_ns)

gpu_db_store_ns  = db_after_nic_ns - db_before_nic_ns
db_to_cqe_ns     = cqe_nic_ns - db_after_nic_ns
cqe_to_gpu_ns    = gpu_seen_nic_ns - cqe_nic_ns
```

Print at least:

```text
shmem_put_latency_ibgda_db_to_cqe:
  size
  avg / p50 / p95 / max
  pair_count
  valid_count
  negative_count
  over_cap_count

shmem_put_latency_ibgda_gpu_db_store:
  size
  avg / p50 / p95 / max
```

Interpretation:

```text
wqe_prep_ns:
  GPU-side WQE construction.

doorbell_submit_ns:
  GPU-side submit path, including fence/CAS/DBR/BF store.

gpu_db_store_ns:
  timestamped duration around the actual BF doorbell store.

db_to_cqe_ns:
  everything after the GPU doorbell store until requester CQE generation:
  NIC doorbell processing + WQE fetch/decode + local GPU payload DMA read +
  TX/network + remote HCA processing + ACK return + requester CQE generation.

cqe_to_gpu_ns:
  GPU polling visibility overhead after NIC writes the CQE.
```

## Normal Put vs Inline WQE vs BlueFlame

You can use normal put and inline WQE as a differential experiment, but the
result is an estimate, not a direct hardware counter.

Define:

```text
T_normal(size) = db_to_cqe for normal RDMA_WRITE with data_seg
T_inline(size) = db_to_cqe for RDMA_WRITE inline WQE
T_nop          = db_to_cqe for signaled NOP or minimal signaled WQE
```

Approximate:

```text
GPU payload DMA read extra(size) ~= T_normal(size) - T_inline(size)
```

This is most meaningful for very small payloads where inline WQE overhead is
small. For larger payloads, inline data increases WQE size, so the difference is:

```text
T_normal - T_inline
  ~= GPU payload DMA read cost
     - extra inline WQE/BF cost
```

So do not label the difference as a pure data-read latency. Label it as
`gpu_payload_dma_extra_vs_inline`.

### Important Inline Detail

`shmem_put_latency.cu` currently calls:

```c
nvshmem_int_put_nbi(data_d, data_d, len, peer);
nvshmem_quiet();
```

This is the bulk put path and normally reaches `ibgda_rma_thread()` with a normal
RDMA_WRITE WQE containing a data segment.

The inline WQE helper exists:

```c
ibgda_write_rdma_write_inl_wqe()
```

but the easiest existing perftest path for inline scalar writes is:

```text
perftest/device/pt-to-pt/shmem_p_latency.cu
```

because it calls:

```c
nvshmem_int_p(data_d + j, *(data_d + j), peer);
```

and the IBGDA scalar `p` path uses inline RDMA write WQEs for small values.

### Implementation Decision

Use one enhanced `shmem_put_latency.cu` binary with selectable modes. Do not
create two new peer files such as `shmem_put_latency_inline.cu` and
`shmem_put_latency_blueflame.cu`.

Reason:

```text
normal / inline / nop / BlueFlame variants must share:
  - the same warmup and measured iteration loop
  - the same GPU->NIC clock calibration
  - the same DB timestamp collection
  - the same CQE timestamp collection
  - the same prod_idx -> completed_idx matching logic
  - the same output and accounting tables
```

Forking the test into separate files would almost certainly make the results
drift because the calibration, reset, matching, and print paths would need to be
kept in sync in multiple places.

Recommended interface:

```text
./shmem_put_latency ... --ibgda-write-mode normal|inline|nop
```

Also keep an environment-variable fallback for scripts and old perftest harnesses:

```text
NVSHMEM_PERFTEST_IBGDA_WRITE_MODE=normal|inline|nop
```

If adding a new common perftest command-line option is too invasive, implement
the environment variable first. It still gives the important property: one
binary, one code path, one measurement pipeline.

### Phase 1: Normal vs Inline in shmem_put_latency.cu

Add a local enum:

```c
typedef enum shmem_put_latency_ibgda_write_mode_e {
    SHMEM_PUT_LATENCY_IBGDA_WRITE_NORMAL = 0,
    SHMEM_PUT_LATENCY_IBGDA_WRITE_INLINE = 1,
    SHMEM_PUT_LATENCY_IBGDA_WRITE_NOP    = 2,
} shmem_put_latency_ibgda_write_mode_t;
```

Parse it on the host before the measurement loop:

```text
1. default: normal
2. if --ibgda-write-mode is present, use it
3. else if NVSHMEM_PERFTEST_IBGDA_WRITE_MODE is present, use it
4. reject unknown strings with a clear error
```

Pass the mode into the single-thread kernel and cubin launch:

```c
__global__ void latency_kern(int *data_d, int len, int pe, int iter, int write_mode)
```

Update `test_latency()` and the `cuLaunchKernel()` arg list to include
`write_mode`. Keep the current warp/block kernels normal-only at first. The
`db_to_cqe` experiment needs one clean requester stream; warp/block modes add
coalescing and extra WQEs that make the differential harder to interpret.

Route the single-thread measured operation as:

```text
normal:
  current path:
    nvshmem_int_put_nbi(data_d, data_d, len, peer)
    nvshmem_quiet()

inline:
  first implementation: 4B only
    require len == 1
    nvshmem_int_p(data_d, *data_d, peer)
    nvshmem_quiet()

  optional extension: 8B only
    use a correctly typed 8B symmetric buffer and nvshmem_long_p /
    nvshmem_longlong_p, depending on the NVSHMEM API type available in this tree

  do not loop over many ints in one iteration for the primary comparison,
  because that changes the unit from "one WQE" to "many scalar WQEs plus one
  quiet".

nop:
  phase 2 helper, described below
```

For inline mode, enforce:

```text
min_size == max_size == 4   for the first patch
threads_per_block does not matter because only latency_kern is used
no warp/block mode output, or print those tables only for normal mode
```

`perftest/device/pt-to-pt/shmem_p_latency.cu` is useful as a sanity check that
the scalar `p` path reaches inline WQE helpers, but it should not be the primary
measurement file because it does not share the new DB/CQE timestamp, calibration,
and accounting code.

### Phase 2: Add NOP Baseline

The `nop` mode is valuable because it estimates the minimum:

```text
doorbell + NIC WQE fetch/decode + CQE generation
```

without local payload DMA read and without remote memory write.

Implement it after normal/inline is working:

```text
1. Add an NVSHMEM_IBGDA_LAT_PROFILE-only device helper that reserves one WQE.
2. Write a signaled NOP WQE with MLX5_WQE_CTRL_CQ_UPDATE.
3. Submit it through the same ibgda_submit_requests() / ibgda_post_send() path.
4. Wait with the same quiet/poll path so CQE timestamp matching still works.
5. Keep it private to the profiling build; do not expose it as a public NVSHMEM API.
```

Relevant existing helper:

```c
ibgda_write_nop_wqe()
```

Expected measurements:

```text
T_normal(4B) - T_inline(4B)
  -> gpu_payload_dma_extra_vs_inline_4B

T_inline(4B) - T_nop
  -> inline payload / remote write / ACK extra over minimal signaled WQE

T_normal(4B) - T_nop
  -> normal RDMA_WRITE data-segment path extra over minimal signaled WQE
```

### Output Plan

Run one mode per process invocation and print the selected mode once before the
tables:

```text
ibgda_write_mode: normal
```

For machine-readable output, include the mode in the metric name or as an extra
field:

```text
shmem_put_latency_ibgda_db_to_cqe___mode__normal___size__4___avg
shmem_put_latency_ibgda_db_to_cqe___mode__inline___size__4___avg
shmem_put_latency_ibgda_db_to_cqe___mode__nop___size__4___avg
```

Do the subtraction in a small post-processing script or spreadsheet first. Do
not make one run execute all modes back-to-back until the one-mode path is
stable; separate invocations make it easier to spot calibration and warmup
issues.

Example run shape:

```text
NVSHMEM_PERFTEST_IBGDA_WRITE_MODE=normal ./shmem_put_latency ...
NVSHMEM_PERFTEST_IBGDA_WRITE_MODE=inline ./shmem_put_latency ...
NVSHMEM_PERFTEST_IBGDA_WRITE_MODE=nop    ./shmem_put_latency ...
```

## BlueFlame On/Off Caveat

Current IBGDA code always rings the doorbell through:

```c
ibgda_ring_db(qp, new_prod_idx)
```

which writes to `qp->tx_wq.bf`.

That does not automatically mean "full WQE is delivered through BlueFlame and
the NIC does not fetch WQE from SQ memory". In this code the BF store is only the
doorbell/control value:

```c
ibgda_ctrl_seg_t ctrl_seg = {
    .opmod_idx_opcode = HTOBE32(prod_idx << 8),
    .qpn_ds = HTOBE32(qp->qpn << 8),
};
ibgda_store_release(bf_ptr, *((uint64_t *)&ctrl_seg));
```

Therefore:

```text
non-BF vs current BF
```

can estimate doorbell submission path differences, but it should not be called
pure WQE-read latency unless the alternative path truly changes whether the NIC
fetches the WQE body from SQ memory.

If you want to estimate WQE fetch latency, implement two valid modes:

```text
doorbell_only:
  WQE body resides in SQ memory; doorbell tells NIC to fetch it.

full_bf:
  enough of the WQE is pushed through the BF/MMIO path so the NIC need not fetch
  the same WQE body from SQ memory, if supported for this WQE type and UAR mode.
```

Then:

```text
WQE fetch extra ~= T_doorbell_only_nop - T_full_bf_nop
```

### NOP and BlueFlame A/B

Yes: the NOP operation should be run with the BlueFlame/db-mode switch. NOP is
the cleanest operation for this A/B because it removes the local payload DMA read
and the remote memory write payload from the normal put path.

The useful comparison is:

```text
T_nop_sq_fetch:
  doorbell/control + NIC DMA fetch of one NOP WQE from SQ memory +
  WQE decode + requester CQE generation

T_nop_full_bf:
  BlueFlame push of the NOP WQE/control bytes +
  WQE decode + requester CQE generation
```

Then:

```text
T_nop_sq_fetch - T_nop_full_bf
  ~= one WQE fetch DMA cost - extra full-BF push/MMIO cost
```

So the difference is the best available estimate of one WQE fetch DMA extra, but
do not call it an exact hardware counter. It is a differential estimate and still
contains the cost difference between the two doorbell delivery mechanisms.

This interpretation only holds if `full_bf_wqe` really pushes enough WQE bytes
through BlueFlame that the NIC does not fetch the same WQE body from SQ memory.
The current `ibgda_ring_db()` path writes only a 64-bit doorbell/control value to
`qp->tx_wq.bf`; comparing that path against a proxy/non-BF control path measures
doorbell/control-path differences, not one WQE DMA fetch time.

### BlueFlame Implementation Plan

Treat BlueFlame as a second axis, not as a second test file. The matrix should
eventually be:

```text
write_mode: normal | inline | nop
db_mode:    current_bf_control | non_bf_control_or_proxy | full_bf_wqe
```

Do not start by implementing all combinations. Use this order:

```text
1. normal + current_bf_control
2. inline + current_bf_control
3. nop + current_bf_control
4. nop + an alternate doorbell path
5. nop + full_bf_wqe, only if the HCA/UAR mode supports pushing the WQE body
```

Important naming:

```text
current_bf_control:
  the current 64-bit store to qp->tx_wq.bf in ibgda_ring_db().
  This is a BlueFlame/UAR doorbell store, not proof that the full WQE body was
  delivered through BlueFlame.

non_bf_control_or_proxy:
  an alternate control path that avoids the same GPU UAR MMIO store. If it uses
  the existing proxy mechanism, label it as proxy/control-path comparison, not
  as pure BlueFlame-off latency.

full_bf_wqe:
  a mode that pushes enough WQE bytes through the BF/MMIO region that the NIC
  does not need to fetch the same WQE body from SQ memory. Only call a result
  "WQE fetch extra" after this is verified.
```

The BlueFlame control may need to be an NVSHMEM/IBGDA environment variable
rather than a per-kernel argument because QP doorbell pointers are populated
during transport initialization:

```text
NVSHMEM_PERFTEST_IBGDA_WRITE_MODE=nop
NVSHMEM_IBGDA_LAT_PROFILE_DB_MODE=current_bf_control|proxy_control|full_bf_wqe
```

Implementation steps for `current_bf_control` are already mostly done by the DB
timestamp work:

```text
1. timestamp immediately before ibgda_ring_db()
2. execute the 64-bit store to qp->tx_wq.bf
3. timestamp immediately after ibgda_ring_db()
4. match the DB record to the CQE by prod_idx/completed_idx
```

Implementation steps for a safe alternate doorbell experiment:

```text
1. Add an IBGDA profiling-only db_mode parsed during transport init.
2. Keep the WQE body construction identical.
3. Change only the post-send doorbell/control mechanism.
4. Reject unsupported combinations at startup instead of silently falling back.
5. Print db_mode next to write_mode in every table.
```

Implementation steps for `full_bf_wqe`, if supported:

```text
1. Start with nop, not normal put, so the WQE is small and deterministic.
2. Add ibgda_ring_db_full_bf(qp, prod_idx, wqe_ptr, wqe_bytes).
3. Copy the required WQEBB bytes to the BF/UAR region with the ordering required
   by mlx5 for this UAR mode.
4. Keep the existing SQ-memory WQE valid until verification proves the NIC did
   not fetch it.
5. Compare nop/current_bf_control against nop/full_bf_wqe.
```

Only after `nop/full_bf_wqe` is verified should normal and inline be run with
that DB mode. Otherwise the result mixes too many effects: WQE construction,
payload DMA, remote write, ACK, CQE generation, and doorbell delivery.

If full BF is not implemented, use `T_nop` only as a lower-bound baseline for
doorbell/FW/WQE-decode/CQE generation.

## Suggested Experiment Matrix

Use one PE pair, one issuing GPU thread, one outstanding request, and one CQE per
iteration.

Build/run mode:

```bash
cmake -S . -B build-cqe-ts \
  -DNVSHMEM_IBGDA_SUPPORT=ON \
  -DNVSHMEM_IBGDA_LAT_PROFILE=ON

cmake --build build-cqe-ts -j
```

Runtime env:

```bash
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH=1
export NVSHMEM_IBGDA_CQE_TS_BUF_CAP=1048576
export NVSHMEM_IBGDA_DB_TS_BUF_CAP=1048576
export NVSHMEM_PERFTEST_CQE_TS_DEBUG=1
```

Run these modes:

```text
1. nop + current_bf_control:
   estimates minimal doorbell/FW/WQE/CQE baseline for the current 64-bit BF
   control-doorbell path.

2. nop + full_bf_wqe, if supported:
   compare against nop/current_bf_control or nop/sq_fetch to estimate one WQE
   fetch-DMA extra. This is the cleanest BlueFlame A/B.

3. inline scalar 4B and 8B:
   use shmem_p_latency or the new inline mode.

4. normal put 4B, 8B, 16B, 32B, 64B, ..., max_size:
   current shmem_put_latency normal mode.

5. loopback / same-host back-to-back:
   estimates local path without full fabric distance.

6. two-host path:
   normal experiment across the target fabric.
```

Compute:

```text
T_db_to_cqe_normal(size)
T_db_to_cqe_inline(size)
T_db_to_cqe_nop
T_db_to_cqe_nop_sq_fetch
T_db_to_cqe_nop_full_bf

payload_dma_extra_vs_inline(size)
  = T_db_to_cqe_normal(size) - T_db_to_cqe_inline(size)

wqe_fetch_dma_extra_estimate
  ~= T_db_to_cqe_nop_sq_fetch - T_db_to_cqe_nop_full_bf

non_payload_base(size)
  ~= T_db_to_cqe_inline(size)

network_remote_extra(size)
  ~= T_db_to_cqe_two_host(size) - T_db_to_cqe_loopback(size)
```

Use medians/p50 first. Average is sensitive to CQ polling jitter and occasional
firmware/PCIe stalls.

## ibdump and mlx5 Counters

Use `ibdump` only to validate wire behavior and ACK timing. It cannot see PCIe
doorbells or NIC DMA reads from GPU memory.

Recommended captures:

```bash
sudo ibdump -d mlx5_0 -i 1 -w sender.pcap
sudo ibdump -d mlx5_1 -i 1 -w receiver.pcap
```

Useful checks:

```text
first RDMA_WRITE packet timestamp
last RDMA_WRITE packet timestamp
ACK/NAK timestamp
PSN sequence
retransmission / NAK / RNR
```

mlx5 counters are sanity checks, not per-WQE latency sources:

```bash
grep . /sys/class/infiniband/mlx5_*/ports/*/hw_counters/*
ethtool -S <netdev> | egrep 'pci|stall|rdma|retrans|timeout|ecn|cnp|discard|error'
```

If PCIe stall counters grow with GPU-source normal put but not inline put, that
supports the interpretation that `T_normal - T_inline` is dominated by GPU
payload DMA read / PCIe effects.

## Final Naming

Use conservative names in printed output:

```text
wqe_prep_ns
doorbell_submit_gpu_path_ns
gpu_db_store_ns
db_to_cqe_ns
cqe_to_gpu_poll_ns
payload_dma_extra_vs_inline_ns
wqe_fetch_extra_vs_full_bf_ns
network_remote_extra_vs_loopback_ns
```

Avoid printing names like `firmware_doorbell_ns` or `gpu_dma_read_ns` unless they
are clearly marked as estimates. The hardware does not expose those exact
per-WQE boundaries.
