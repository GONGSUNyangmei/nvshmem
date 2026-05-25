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
```

These are useful, but `doorbell_submit` is still a GPU-side submit-path time. It
includes GPU fences, `ready_head` CAS/spin, DBR write, and the BlueFlame MMIO
store. It is not the NIC firmware doorbell-processing time.

The missing timestamp is the one immediately around the actual doorbell store in
`ibgda_post_send()`.

## Add Doorbell Timestamp Records

Add a second device-side ring buffer for doorbell timestamps.

Record one entry per real post-send:

```c
typedef struct nvshmemx_ibgda_db_ts_pair_s {
    unsigned long long prod_idx;
    unsigned long long db_before_gpu_ns;
    unsigned long long db_after_gpu_ns;
} nvshmemx_ibgda_db_ts_pair_t;
```

`prod_idx` must be the full `new_prod_idx`, not only the low 16-bit value passed
to `ibgda_ring_db()`. It is needed to match the doorbell with the CQE's completed
WQE index.

### 1. Header API

Edit `src/include/host/nvshmemx_ibgda_lat_profile.h`.

Add:

```c
typedef struct nvshmemx_ibgda_db_ts_pair_s {
    unsigned long long prod_idx;
    unsigned long long db_before_gpu_ns;
    unsigned long long db_after_gpu_ns;
} nvshmemx_ibgda_db_ts_pair_t;

int nvshmemx_ibgda_lat_profile_db_ts_reset(void);
int nvshmemx_ibgda_lat_profile_db_ts_get(nvshmemx_ibgda_db_ts_pair_t *out,
                                         size_t max_pairs, size_t *out_count);
```

Also extend the existing CQE pair with a completed WQE index:

```c
typedef struct nvshmemx_ibgda_cqe_ts_pair_s {
    unsigned long long completed_idx;
    unsigned long long cqe_ts_cycles;
    unsigned long long gpu_now_ns;
} nvshmemx_ibgda_cqe_ts_pair_t;
```

If ABI stability matters, create a new `nvshmemx_ibgda_cqe_ts_pair_v2_t` instead
of changing the existing struct.

### 2. Device Symbols and Host Accessors

Edit `src/device/init/init_device.cu`.

Add device symbols next to the existing CQE timestamp symbols:

```c
__device__ __attribute__((used)) unsigned long long *nvshmemi_ibgda_db_prod_idx_buf = nullptr;
__device__ __attribute__((used)) unsigned long long *nvshmemi_ibgda_db_before_buf   = nullptr;
__device__ __attribute__((used)) unsigned long long *nvshmemi_ibgda_db_after_buf    = nullptr;
__device__ __attribute__((used)) unsigned long long  nvshmemi_ibgda_db_ts_cap       = 0;
__device__ __attribute__((used)) unsigned long long  nvshmemi_ibgda_db_ts_count     = 0;
```

Implement `nvshmemx_ibgda_lat_profile_db_ts_reset()` and
`nvshmemx_ibgda_lat_profile_db_ts_get()` by copying the pattern already used by:

```c
nvshmemx_ibgda_lat_profile_cqe_ts_reset()
nvshmemx_ibgda_lat_profile_cqe_ts_get()
```

Use a separate env cap:

```text
NVSHMEM_IBGDA_DB_TS_BUF_CAP
```

Default can be `65536`, same as the CQE buffer.

Also update the no-IBGDA stubs in the `#else /* !NVSHMEM_IBGDA_SUPPORT */`
section.

If symbol exports are required by this build, add the new APIs to
`nvshmem_host.sym`.

### 3. Device Recorder

Edit `src/include/non_abi/device/pt-to-pt/ibgda_device.cuh`.

Add extern declarations next to the CQE timestamp externs:

```c
extern __device__ unsigned long long *nvshmemi_ibgda_db_prod_idx_buf;
extern __device__ unsigned long long *nvshmemi_ibgda_db_before_buf;
extern __device__ unsigned long long *nvshmemi_ibgda_db_after_buf;
extern __device__ unsigned long long  nvshmemi_ibgda_db_ts_cap;
extern __device__ unsigned long long  nvshmemi_ibgda_db_ts_count;
```

Add a device helper under `#ifdef NVSHMEM_IBGDA_LAT_PROFILE`:

```c
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_lat_prof_record_db_ts(
    unsigned long long prod_idx,
    unsigned long long db_before_gpu_ns,
    unsigned long long db_after_gpu_ns) {
    unsigned long long cap = nvshmemi_ibgda_db_ts_cap;
    unsigned long long slot = atomicAdd_system(&nvshmemi_ibgda_db_ts_count, 1ULL);
    if (cap == 0 || nvshmemi_ibgda_db_prod_idx_buf == nullptr ||
        nvshmemi_ibgda_db_before_buf == nullptr ||
        nvshmemi_ibgda_db_after_buf == nullptr) return;

    if (slot < cap) {
        nvshmemi_ibgda_db_prod_idx_buf[slot] = prod_idx;
        nvshmemi_ibgda_db_before_buf[slot] = db_before_gpu_ns;
        nvshmemi_ibgda_db_after_buf[slot] = db_after_gpu_ns;
    }
}
```

### 4. Timestamp the Real Doorbell Store

Edit `ibgda_post_send()` in
`src/include/non_abi/device/pt-to-pt/ibgda_device.cuh`.

Current code:

```c
if (likely(new_prod_idx > old_prod_idx)) {
    IBGDA_MEMBAR();
    ibgda_update_dbr(qp, new_prod_idx);
    IBGDA_MEMBAR();
    ibgda_ring_db(qp, new_prod_idx);
}
```

Change to:

```c
if (likely(new_prod_idx > old_prod_idx)) {
    IBGDA_MEMBAR();
    ibgda_update_dbr(qp, new_prod_idx);
    IBGDA_MEMBAR();

#ifdef NVSHMEM_IBGDA_LAT_PROFILE
    unsigned long long db_before_gpu_ns = 0;
    unsigned long long db_after_gpu_ns = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(db_before_gpu_ns)::"memory");
#endif

    ibgda_ring_db(qp, new_prod_idx);

#ifdef NVSHMEM_IBGDA_LAT_PROFILE
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(db_after_gpu_ns)::"memory");
    ibgda_lat_prof_record_db_ts((unsigned long long)new_prod_idx,
                                db_before_gpu_ns,
                                db_after_gpu_ns);
#endif
}
```

`db_after_gpu_ns` is the best available "GPU has completed the doorbell store"
boundary. After calibrating GPU `%globaltimer` to the mlx5 raw clock domain, use
this as the left edge of `doorbell -> CQE`.

### 5. Record CQE completed_idx

In `ibgda_poll_cq()` the code already computes:

```c
completed_idx = ...
record_target_cqe = (completed_idx >= idx);
```

Extend `ibgda_lat_prof_record_cqe_ts()` to store `completed_idx` as well:

```c
ibgda_lat_prof_record_cqe_ts((unsigned long long)completed_idx,
                             (unsigned long long)prof_cqe_ts,
                             (unsigned long long)prof_now_ns);
```

This allows matching:

```text
db record prod_idx == cqe record completed_idx
```

or, for coalesced CQEs:

```text
db record prod_idx <= cqe record completed_idx
and the CQE is the first later CQE covering that prod_idx
```

For the cleanest per-iteration experiment, force one outstanding request and one
CQE per iteration.

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

For a clean apples-to-apples comparison inside `shmem_put_latency.cu`, add a
new mode:

```text
NVSHMEM_PERFTEST_IBGDA_WRITE_MODE=normal|inline|nop
```

Then route:

```text
normal:
  current nvshmem_int_put_nbi + quiet

inline:
  for 4B/8B only, call nvshmem_int_p / nvshmem_long_p in the kernel
  or add a controlled IBGDA-only inline write path for bytes <= 12

nop:
  enqueue a signaled NOP WQE with CQ update, then quiet/poll it
```

The `nop` mode is valuable because it estimates the minimum:

```text
doorbell + NIC WQE fetch/decode + CQE generation
```

without local payload DMA read and without remote memory write.

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
1. nop, 4B-equivalent signaled:
   estimates minimal doorbell/FW/WQE/CQE baseline.

2. inline scalar 4B and 8B:
   use shmem_p_latency or the new inline mode.

3. normal put 4B, 8B, 16B, 32B, 64B, ..., max_size:
   current shmem_put_latency normal mode.

4. loopback / same-host back-to-back:
   estimates local path without full fabric distance.

5. two-host path:
   normal experiment across the target fabric.
```

Compute:

```text
T_db_to_cqe_normal(size)
T_db_to_cqe_inline(size)
T_db_to_cqe_nop

payload_dma_extra_vs_inline(size)
  = T_db_to_cqe_normal(size) - T_db_to_cqe_inline(size)

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
