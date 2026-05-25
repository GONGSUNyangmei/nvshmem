/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

/*
 * IBGDA single-thread put latency profiling API.
 *
 * When NVSHMEM is built with -DNVSHMEM_IBGDA_LAT_PROFILE=ON, the IBGDA
 * single-thread put path (ibgda_rma_thread) records, per WQE, two GPU clock
 * cycle deltas:
 *
 *   wqe_cycles  = clock64() at "ibgda_submit_requests" entry  - clock64() at
 *                 ibgda_rma_thread entry
 *   db_cycles   = clock64() at "ibgda_submit_requests" return - clock64() at
 *                 "ibgda_submit_requests" entry
 *
 * The first delta covers QP/lkey/rkey lookup + WQE byte writes (the "WQE
 * preparation" phase). The second delta covers IBGDA_MEMBAR + atomicCAS spin
 * on ready_head + ibgda_post_send (DBR record write + BlueFlame MMIO ring) -
 * the "doorbell submit" phase.
 *
 * The host reads the accumulated counters via the get/reset APIs below.
 *
 * Note: clock64() is per-SM. For the single-thread perftest kernel
 * (1 block x 1 thread) the deltas are well-defined. gpu_clock_hz is taken
 * from cudaGetDeviceProperties().clockRate (kHz) and may differ from the
 * actual boost clock at runtime - treat returned ns values as estimates.
 *
 * When NVSHMEM is built without NVSHMEM_IBGDA_LAT_PROFILE, the on-device
 * sampling code is omitted, so all counters stay at zero, but the host
 * symbols are still exported (returns success with count == 0).
 */

#ifndef _NVSHMEMX_IBGDA_LAT_PROFILE_H_
#define _NVSHMEMX_IBGDA_LAT_PROFILE_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nvshmemx_ibgda_lat_profile_s {
    /* Sum / max are in raw GPU clock cycles. */
    unsigned long long wqe_cycles_sum;
    unsigned long long wqe_cycles_max;
    unsigned long long db_cycles_sum;
    unsigned long long db_cycles_max;
    /* Number of recorded WQE submissions. */
    unsigned long long count;
    /* GPU SM clock rate in Hz, suitable for converting cycles to ns. */
    double gpu_clock_hz;
} nvshmemx_ibgda_lat_profile_t;

/* Snapshot the current counters into *out. Returns 0 on success, non-zero on
 * a CUDA error (check the most recent CUDA error). gpu_clock_hz is filled in
 * from cudaGetDeviceProperties() of the current device. */
int nvshmemx_ibgda_lat_profile_get(nvshmemx_ibgda_lat_profile_t *out);

/* Zero all counters. Returns 0 on success. */
int nvshmemx_ibgda_lat_profile_reset(void);

/* ------------------------------------------------------------------ */
/*  CQE-timestamp pair API (NIC CQE-gen -> GPU thread CQE-read).      */
/*                                                                    */
/*  When NVSHMEM is built with NVSHMEM_IBGDA_LAT_PROFILE=ON, every    */
/*  successful ibgda_poll_cq() observation stores one pair of:        */
/*    - cqe_ts_cycles: raw HCA core-clock cycles from                 */
/*                     cqe64->timestamp (already byte-swapped to      */
/*                     host endianness)                               */
/*    - gpu_now_ns:    %globaltimer ns read by the GPU thread right   */
/*                     after observing the new CQE                    */
/*  into a host-allocated device ring buffer (default cap 65536       */
/*  pairs, overridable via NVSHMEM_IBGDA_CQE_TS_BUF_CAP).             */
/*                                                                    */
/*  Convert cqe_ts_cycles to CLOCK_REALTIME ns with mlx5dv_ts_to_ns() */
/*  using the mlx5dv_clock_info fetched via                           */
/*  nvshmemx_ibgda_get_mlx5_clock_info(). The two clocks need one     */
/*  caller-side calibration (clock_gettime REALTIME vs MONOTONIC_RAW) */
/*  to align with %globaltimer for an absolute ns delta.              */
/* ------------------------------------------------------------------ */
typedef struct nvshmemx_ibgda_cqe_ts_pair_s {
    /* Full completed WQE index represented by the CQE. */
    unsigned long long completed_idx;
    /* HCA HW cycles from cqe64->timestamp (host-endian, FREE_RUNNING). */
    unsigned long long cqe_ts_cycles;
    /* GPU %globaltimer ns when the GPU thread observed the CQE. */
    unsigned long long gpu_now_ns;
} nvshmemx_ibgda_cqe_ts_pair_t;

typedef struct nvshmemx_ibgda_db_ts_pair_s {
    /* Full producer index passed to ibgda_post_send(), not the low 16 bits. */
    unsigned long long prod_idx;
    /* GPU %globaltimer ns immediately before the real doorbell store. */
    unsigned long long db_before_gpu_ns;
    /* GPU %globaltimer ns immediately after the real doorbell store. */
    unsigned long long db_after_gpu_ns;
} nvshmemx_ibgda_db_ts_pair_t;

/* Snapshot up to max_pairs most recent pairs into out[]. On success returns
 * 0 and writes the number of pairs actually copied to *out_count (or 0 if
 * the device buffer hasn't been initialized via _reset yet). */
int nvshmemx_ibgda_lat_profile_cqe_ts_get(nvshmemx_ibgda_cqe_ts_pair_t *out,
                                          size_t max_pairs, size_t *out_count);

int nvshmemx_ibgda_lat_profile_db_ts_get(nvshmemx_ibgda_db_ts_pair_t *out,
                                         size_t max_pairs, size_t *out_count);

/* Lazily allocate (cap from NVSHMEM_IBGDA_CQE_TS_BUF_CAP env, default 64K
 * pairs) the device ring buffer and zero the recorded count. Must be
 * called at least once before the GPU starts recording. */
int nvshmemx_ibgda_lat_profile_cqe_ts_reset(void);

int nvshmemx_ibgda_lat_profile_db_ts_reset(void);

/* Copy the cached mlx5dv_clock_info captured by the IBGDA transport into
 * *out. *out must point at a struct mlx5dv_clock_info (the header avoids
 * dragging in <infiniband/mlx5dv.h>; callers cast). Returns 0 if the
 * transport has published a value, non-zero (cudaErrorNotReady) if not. */
int nvshmemx_ibgda_get_mlx5_clock_info(void *out);

/* Query a fresh mlx5dv_clock_info snapshot from the live IBGDA transport
 * context instead of using the cached copy captured at first CQ creation.
 * Returns 0 on success, non-zero if the transport is unavailable or the
 * provider query fails. */
int nvshmemx_ibgda_query_mlx5_clock_info(void *out);

/* Query the provider's current HCA raw clock in nanoseconds (the same domain
 * reported by ibv_query_rt_values_ex(raw_clock)). This can be used to align
 * CQE timestamps with GPU %globaltimer without relying on CLOCK_REALTIME/PTP
 * wall-clock synchronization. */
int nvshmemx_ibgda_query_mlx5_raw_clock_ns(unsigned long long *out_ns);

/* Internal: called by the IBGDA transport. Not intended for application use,
 * but exported so the transport .so can resolve it at dlopen time. */
int nvshmemx_ibgda_publish_mlx5_clock_info(const void *src, size_t sz);
int nvshmemx_ibgda_publish_mlx5_context(const void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NVSHMEMX_IBGDA_LAT_PROFILE_H_ */
