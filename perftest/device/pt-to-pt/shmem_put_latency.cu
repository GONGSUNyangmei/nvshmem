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

#define CUMODULE_NAME "shmem_put_latency.cubin"

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils.h"
#include "host/nvshmemx_ibgda_lat_profile.h"

/* mlx5dv.h declares struct mlx5dv_clock_info used by the optional host-side
 * transport diagnostics in main(). The bitcode/cubin device-only pass
 * (NVSHMEM_HOSTLIB_ONLY) only emits device code, so the references inside
 * the host-only main() are parsed but never codegen'd; the header itself
 * is plain C and clean to parse under --cuda-device-only. */
#include <infiniband/mlx5dv.h>

#define THREADS_PER_WARP 32
#define NSEC_PER_SEC 1000000000ULL
#define CQE_REALTIME_SEC_WINDOW 3600ULL

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
extern "C" {
#endif

__global__ void latency_kern(int *data_d, int len, int pe, int iter) {
    int i, peer;

    peer = !pe;

    for (i = 0; i < iter; i++) {
        nvshmem_int_put_nbi(data_d, data_d, len, peer);
        nvshmem_quiet();
    }
}

#define LATENCY_THREADGROUP(group)                                                 \
    __global__ void latency_kern_##group(int *data_d, int len, int pe, int iter) { \
        int i, tid, peer;                                                          \
                                                                                   \
        peer = !pe;                                                                \
        tid = threadIdx.x;                                                         \
                                                                                   \
        for (i = 0; i < iter; i++) {                                               \
            nvshmemx_int_put_##group(data_d, data_d, len, peer);                   \
                                                                                   \
            __syncthreads();                                                       \
            if (!tid) nvshmem_quiet();                                             \
            __syncthreads();                                                       \
        }                                                                          \
    }

LATENCY_THREADGROUP(warp)
LATENCY_THREADGROUP(block)

#if defined __cplusplus || defined NVSHMEM_HOSTLIB_ONLY
}
#endif

#define DEFINE_TEST_LATENCY(TG)                                                               \
                                                                                              \
    void test_latency##TG(int *data_d, int len, int pe, int iter, CUfunction kernel,          \
                          int threads) {                                                      \
        if (use_cubin) {                                                                      \
            void *arglist[] = {(void *)&data_d, (void *)&len, (void *)&pe, (void *)&iter};    \
            CU_CHECK(cuLaunchKernel(kernel, 1, 1, 1, threads, 1, 1, 0, NULL, arglist, NULL)); \
        } else {                                                                              \
            latency_kern##TG<<<1, threads>>>(data_d, len, pe, iter);                          \
        }                                                                                     \
    }

DEFINE_TEST_LATENCY()
DEFINE_TEST_LATENCY(_warp)
DEFINE_TEST_LATENCY(_block)

struct gpu_calib_signal_t {
    volatile unsigned int request;
    volatile unsigned int ack;
    unsigned long long gpu_now_ns;
};

static __global__ void sample_globaltimer_on_signal_kern(volatile unsigned int *request,
                                                         volatile unsigned int *ack,
                                                         unsigned long long *gpu_now_ns) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned int seen = *ack;
    while (1) {
        unsigned int req = *request;
        if (req != seen) {
            if (req == 0xffffffffu) {
                *ack = req;
                __threadfence_system();
                return;
            }

            unsigned long long now_ns;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now_ns)::"memory");
            *gpu_now_ns = now_ns;
            __threadfence_system();
            *ack = req;
            __threadfence_system();
            seen = req;
        }
    }
}

static inline uint64_t timespec_to_ns(const struct timespec *ts) {
    return ((uint64_t)ts->tv_sec * NSEC_PER_SEC) + (uint64_t)ts->tv_nsec;
}

static inline void normalize_mlx5_clock_info_mask(struct mlx5dv_clock_info *ci) {
    (void)ci;
}

static inline bool cqe_timestamp_realtime_ns(uint64_t realtime_timestamp, uint64_t *out_ns,
                                             uint32_t *out_sec, uint32_t *out_nsec) {
    uint64_t sec = realtime_timestamp >> 32;
    uint64_t nsec = realtime_timestamp & 0xffffffffULL;

    if (out_sec) *out_sec = (uint32_t)sec;
    if (out_nsec) *out_nsec = (uint32_t)nsec;
    if (nsec >= NSEC_PER_SEC) return false;
    if (sec > UINT64_MAX / NSEC_PER_SEC) return false;
    if (out_ns) *out_ns = (sec * NSEC_PER_SEC) + nsec;
    return true;
}

static inline bool cqe_realtime_sec_in_window(uint64_t sec, uint64_t ref_sec) {
    return sec + CQE_REALTIME_SEC_WINDOW >= ref_sec &&
           sec <= ref_sec + CQE_REALTIME_SEC_WINDOW;
}

static int compare_u64(const void *lhs, const void *rhs) {
    uint64_t a = *(const uint64_t *)lhs;
    uint64_t b = *(const uint64_t *)rhs;
    return (a > b) - (a < b);
}

static uint64_t percentile_nearest_rank_u64(const uint64_t *sorted, size_t count,
                                            unsigned percentile) {
    if (!sorted || count == 0) return 0;
    size_t rank = ((size_t)percentile * count + 99) / 100;
    if (rank == 0) rank = 1;
    if (rank > count) rank = count;
    return sorted[rank - 1];
}

static void print_table_cqe_to_thread(uint64_t *size, double *avg_ns, double *max_ns,
                                      double *p50_ns, double *p95_ns, size_t *pair_count,
                                      size_t *valid_count, size_t *negative_count,
                                      size_t *over_cap_count, size_t *raw_clock_count,
                                      size_t *invalid_format_count, int num_entries) {
    bool machine_readable = false;
    char *env_value = getenv("NVSHMEM_MACHINE_READABLE_OUTPUT");
    if (env_value) machine_readable = atoi(env_value);

    if (machine_readable) {
        printf("%s\n", "shmem_put_latency_cqe_to_thread");
        for (int i = 0; i < num_entries; ++i) {
            if (size[i] == 0 || pair_count[i] == 0) continue;
            printf("&&&& PERF shmem_put_latency_cqe_to_thread___Thread___size__%lu___nic_cqe_to_gpu_avg %lf -ns\n",
                   size[i], avg_ns[i]);
            printf("&&&& PERF shmem_put_latency_cqe_to_thread___Thread___size__%lu___nic_cqe_to_gpu_max %lf -ns\n",
                   size[i], max_ns[i]);
            printf("&&&& PERF shmem_put_latency_cqe_to_thread___Thread___size__%lu___nic_cqe_to_gpu_p50 %lf -ns\n",
                   size[i], p50_ns[i]);
            printf("&&&& PERF shmem_put_latency_cqe_to_thread___Thread___size__%lu___nic_cqe_to_gpu_p95 %lf -ns\n",
                   size[i], p95_ns[i]);
            printf("&&&& PERF shmem_put_latency_cqe_to_thread___Thread___size__%lu___pair_count %zu -count\n",
                   size[i], pair_count[i]);
            printf("&&&& PERF shmem_put_latency_cqe_to_thread___Thread___size__%lu___valid_count %zu -count\n",
                   size[i], valid_count[i]);
            printf("&&&& PERF shmem_put_latency_cqe_to_thread___Thread___size__%lu___negative_count %zu -count\n",
                   size[i], negative_count[i]);
            printf("&&&& PERF shmem_put_latency_cqe_to_thread___Thread___size__%lu___over_cap_count %zu -count\n",
                   size[i], over_cap_count[i]);
            printf("&&&& PERF shmem_put_latency_cqe_to_thread___Thread___size__%lu___raw_clock_count %zu -count\n",
                   size[i], raw_clock_count[i]);
            printf("&&&& PERF shmem_put_latency_cqe_to_thread___Thread___size__%lu___invalid_format_count %zu -count\n",
                   size[i], invalid_format_count[i]);
        }
        return;
    }

    printf("#%10s\n", "shmem_put_latency_cqe_to_thread");
    printf("%-10s  %-8s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s  %-15s  %-14s  %-18s  %-20s\n",
           "size(B)", "scope", "avg(ns)", "max(ns)", "p50(ns)", "p95(ns)",
           "pair_count", "valid_count", "negative_count", "over_cap_count",
           "raw_clock_count", "invalid_format_count");
    for (int i = 0; i < num_entries; ++i) {
        if (size[i] == 0 || pair_count[i] == 0) continue;
        printf("%-10lu  %-8s  %-12.6lf  %-12.6lf  %-12.6lf  %-12.6lf  %-12zu  %-12zu  %-15zu  %-14zu  %-18zu  %-20zu\n",
               size[i], "Thread", avg_ns[i], max_ns[i], p50_ns[i], p95_ns[i],
               pair_count[i], valid_count[i], negative_count[i], over_cap_count[i],
               raw_clock_count[i], invalid_format_count[i]);
    }
}

typedef struct gpu_nic_clock_calibration_s {
    unsigned long long gpu_ref;
    long double nic_ref_ns;
    long double nic_per_gpu_tick_ns;
    /* Conservative NIC timestamp clock <-> GPU %globaltimer sync error bound, in ns. */
    uint64_t fit_error_ns;
} gpu_nic_clock_calibration_t;

static inline void init_gpu_nic_clock_calibration(gpu_nic_clock_calibration_t *calib) {
    memset(calib, 0, sizeof(*calib));
    calib->nic_per_gpu_tick_ns = 1.0L;
    calib->fit_error_ns = UINT64_MAX;
}

static inline int64_t map_gpu_globaltimer_to_nic_ns(
    const gpu_nic_clock_calibration_t *calib, unsigned long long gpu_timestamp) {
    long double gpu_delta = (long double)((long long)gpu_timestamp - (long long)calib->gpu_ref);
    long double mapped_ns = calib->nic_ref_ns + gpu_delta * calib->nic_per_gpu_tick_ns;
    return (int64_t)llroundl(mapped_ns);
}

typedef int (*query_clock_ns_fn_t)(unsigned long long *out_ns);

static int query_host_realtime_ns(unsigned long long *out_ns) {
    if (!out_ns) return (int)cudaErrorInvalidValue;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    *out_ns = (unsigned long long)timespec_to_ns(&ts);
    return 0;
}

static int query_mlx5_raw_clock_ns(unsigned long long *out_ns) {
    if (!out_ns) return (int)cudaErrorInvalidValue;
    return nvshmemx_ibgda_query_mlx5_raw_clock_ns(out_ns);
}

static int calibrate_gpu_globaltimer_to_clock_ns(cudaStream_t stream,
                                                 query_clock_ns_fn_t query_clock_ns,
                                                 const char *debug_tag,
                                                 gpu_nic_clock_calibration_t *calib_out) {
    const int warmup_samples = 4;
    const int measure_samples = 32;
    const int total_samples = warmup_samples + measure_samples;
    const char *debug_env = getenv("NVSHMEM_PERFTEST_CQE_TS_DEBUG");
    int debug = debug_env ? atoi(debug_env) : 0;
    gpu_calib_signal_t *signal_h = NULL;
    gpu_calib_signal_t *signal_d = NULL;
    unsigned long long gpu_samples[measure_samples];
    unsigned long long nic_midpoints_ns[measure_samples];
    uint64_t window_samples_ns[measure_samples];
    size_t sample_count = 0;
    cudaError_t err = cudaSuccess;

    if (!stream || !query_clock_ns || !calib_out) return (int)cudaErrorInvalidValue;
    if (!debug_tag) debug_tag = "calib_clock";

    init_gpu_nic_clock_calibration(calib_out);

    err = cudaHostAlloc((void **)&signal_h, sizeof(*signal_h),
                        cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) return (int)err;
    memset((void *)signal_h, 0, sizeof(*signal_h));

    err = cudaHostGetDevicePointer((void **)&signal_d, (void *)signal_h, 0);
    if (err != cudaSuccess) {
        cudaFreeHost(signal_h);
        return (int)err;
    }

    sample_globaltimer_on_signal_kern<<<1, 1, 0, stream>>>(&signal_d->request, &signal_d->ack,
                                                           &signal_d->gpu_now_ns);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFreeHost(signal_h);
        return (int)err;
    }

    for (unsigned int sample = 1; sample <= (unsigned int)total_samples; ++sample) {
        unsigned long long before_ns = 0ULL;
        unsigned long long after_ns = 0ULL;

        int rc = query_clock_ns(&before_ns);
        if (rc != 0) {
            err = (cudaError_t)rc;
            break;
        }

        signal_h->request = sample;
        __sync_synchronize();

        while (signal_h->ack != sample) {
        }

        rc = query_clock_ns(&after_ns);
        if (rc != 0) {
            err = (cudaError_t)rc;
            break;
        }
        if (after_ns <= before_ns) continue;

        if (sample > (unsigned int)warmup_samples) {
            uint64_t window_ns = after_ns - before_ns;
            uint64_t midpoint_ns = before_ns + (window_ns / 2);
            if (sample_count < (size_t)measure_samples) {
                gpu_samples[sample_count] = signal_h->gpu_now_ns;
                nic_midpoints_ns[sample_count] = midpoint_ns;
                window_samples_ns[sample_count] = window_ns;
                sample_count++;
            }

            if (debug >= 2) {
                printf("CQE_TS_DEBUG %s sample=%u before_ns=%llu gpu_ns=%llu after_ns=%llu window_ns=%llu\n",
                       debug_tag,
                       sample - (unsigned int)warmup_samples,
                       (unsigned long long)before_ns,
                       (unsigned long long)signal_h->gpu_now_ns,
                       (unsigned long long)after_ns,
                       (unsigned long long)window_ns);
            }
        }
    }

    signal_h->request = 0xffffffffu;
    __sync_synchronize();
    while (signal_h->ack != 0xffffffffu) {
    }
    err = cudaStreamSynchronize(stream);

    if (err == cudaSuccess && sample_count >= 2) {
        const unsigned long long gpu_ref = gpu_samples[0];
        const unsigned long long nic_ref = nic_midpoints_ns[0];
        long double sum_x = 0.0L;
        long double sum_y = 0.0L;
        long double sum_xx = 0.0L;
        long double sum_xy = 0.0L;
        long double slope = 1.0L;
        long double intercept = 0.0L;
        long double max_residual_ns = 0.0L;
        uint64_t best_window_ns = UINT64_MAX;

        for (size_t i = 0; i < sample_count; ++i) {
            long double x =
                (long double)((long long)gpu_samples[i] - (long long)gpu_ref);
            long double y =
                (long double)((long long)nic_midpoints_ns[i] - (long long)nic_ref);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            if (window_samples_ns[i] < best_window_ns) best_window_ns = window_samples_ns[i];
        }

        long double n = (long double)sample_count;
        long double denom = (n * sum_xx) - (sum_x * sum_x);
        if (fabsl(denom) > 0.0L) {
            slope = ((n * sum_xy) - (sum_x * sum_y)) / denom;
            intercept = (sum_y - slope * sum_x) / n;
        }

        for (size_t i = 0; i < sample_count; ++i) {
            long double x =
                (long double)((long long)gpu_samples[i] - (long long)gpu_ref);
            long double predicted_ns = ((long double)nic_ref + intercept) + (x * slope);
            long double residual_ns = fabsl(predicted_ns - (long double)nic_midpoints_ns[i]);
            if (residual_ns > max_residual_ns) max_residual_ns = residual_ns;
            if (debug >= 2) {
                printf("CQE_TS_DEBUG %s_fit sample=%zu x=%0.0Lf predicted_ns=%0.0Lf observed_ns=%llu residual_ns=%0.3Lf\n",
                       debug_tag,
                       i + 1, x, predicted_ns,
                       (unsigned long long)nic_midpoints_ns[i], residual_ns);
            }
        }

        calib_out->gpu_ref = gpu_ref;
        calib_out->nic_ref_ns = (long double)nic_ref + intercept;
        calib_out->nic_per_gpu_tick_ns = slope;
        calib_out->fit_error_ns = (uint64_t)ceill(max_residual_ns);
        if (best_window_ns != UINT64_MAX &&
            calib_out->fit_error_ns < (best_window_ns / 2ULL)) {
            calib_out->fit_error_ns = best_window_ns / 2ULL;
        }
        if (debug >= 2) {
            printf("CQE_TS_DEBUG %s best_window_ns=%llu gpu_ref=%llu nic_ref_ns=%0.3Lf slope=%0.12Lf fit_error_ns=%llu\n",
                   debug_tag, (unsigned long long)best_window_ns, gpu_ref, calib_out->nic_ref_ns,
                   calib_out->nic_per_gpu_tick_ns,
                   (unsigned long long)calib_out->fit_error_ns);
        }
    } else if (err == cudaSuccess) {
        err = cudaErrorUnknown;
    }

    cudaFreeHost(signal_h);
    return (int)err;
}

static int calibrate_gpu_globaltimer_to_nic_ns(cudaStream_t stream,
                                               gpu_nic_clock_calibration_t *calib_out) {
    return calibrate_gpu_globaltimer_to_clock_ns(stream, query_mlx5_raw_clock_ns,
                                                 "calib_nic_raw", calib_out);
}

static int calibrate_gpu_globaltimer_to_realtime_ns(cudaStream_t stream,
                                                    gpu_nic_clock_calibration_t *calib_out) {
    return calibrate_gpu_globaltimer_to_clock_ns(stream, query_host_realtime_ns,
                                                 "calib_realtime", calib_out);
}

static int combine_gpu_nic_calibrations(const gpu_nic_clock_calibration_t *before,
                                        const gpu_nic_clock_calibration_t *after,
                                        gpu_nic_clock_calibration_t *combined) {
    if (!before || !after || !combined) return (int)cudaErrorInvalidValue;
    if (before->fit_error_ns == UINT64_MAX || after->fit_error_ns == UINT64_MAX) {
        return (int)cudaErrorInvalidValue;
    }
    if (after->gpu_ref <= before->gpu_ref || after->nic_ref_ns <= before->nic_ref_ns) {
        return (int)cudaErrorUnknown;
    }

    memset(combined, 0, sizeof(*combined));
    combined->gpu_ref = before->gpu_ref;
    combined->nic_ref_ns = before->nic_ref_ns;
    combined->nic_per_gpu_tick_ns =
        (after->nic_ref_ns - before->nic_ref_ns) /
        (long double)(after->gpu_ref - before->gpu_ref);
    combined->fit_error_ns =
        before->fit_error_ns > after->fit_error_ns ? before->fit_error_ns : after->fit_error_ns;
    return 0;
}

static int select_gpu_nic_calibration(int before_status,
                                      const gpu_nic_clock_calibration_t *before,
                                      int after_status,
                                      const gpu_nic_clock_calibration_t *after,
                                      gpu_nic_clock_calibration_t *selected) {
    int status = cudaErrorInvalidValue;

    init_gpu_nic_clock_calibration(selected);
    if (before_status == 0 && after_status == 0) {
        status = combine_gpu_nic_calibrations(before, after, selected);
    }
    if (status != 0 && after_status == 0) {
        *selected = *after;
        status = after_status;
    } else if (status != 0 && before_status == 0) {
        *selected = *before;
        status = before_status;
    }
    return status;
}

int main(int argc, char *argv[]) {
    int mype, npes, size;
    int *data_d = NULL;

    read_args(argc, argv);
    int iter = iters;
    int skip = warmup_iters;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;

    float milliseconds;
    cudaEvent_t start, stop;
    cudaStream_t calib_stream = NULL;
    CUfunction test_cubin = NULL;
    CUfunction test_cubin_warp = NULL;
    CUfunction test_cubin_block = NULL;

    /* Per-size breakdown of the IBGDA single-thread put path into "WQE prep"
     * and "doorbell submit" phases. Filled in only on PE 0 in the thread
     * loop below, then dumped after the existing latency table. Declared
     * here (above any `goto finalize`) so they aren't bypassed; allocated
     * just before the measurement loop. Values stay 0 if NVSHMEM was not
     * built with NVSHMEM_IBGDA_LAT_PROFILE=ON. */
    double *h_wqe_avg_ns = NULL;
    double *h_wqe_max_ns = NULL;
    double *h_db_avg_ns  = NULL;
    double *h_db_max_ns  = NULL;

    /* Per-size breakdown of "NIC CQE generation -> GPU thread CQE read"
     * latency. The device-side recorder writes (cqe_ts, gpu_now_ns) pairs
     * from inside ibgda_poll_cq; here on the host we decode the CQE
     * REAL_TIME timestamp as sec[63:32] + nsec[31:0] or consume it directly
     * when the CQE field is already in HCA raw_clock ns. The GPU
     * %globaltimer is independently calibrated into the mlx5/NIC timestamp
     * ns domain.
     * Values stay 0 if NVSHMEM was not built with
     * NVSHMEM_IBGDA_LAT_PROFILE=ON or if the GPU clock calibration is too
     * noisy to trust. */
    double *h_cqe2gpu_avg_ns = NULL;
    double *h_cqe2gpu_max_ns = NULL;
    double *h_cqe2gpu_p50_ns = NULL;
    double *h_cqe2gpu_p95_ns = NULL;
    double *h_nic_gpu_sync_precision_ns = NULL;
    size_t *h_cqe_pair_count = NULL;
    size_t *h_cqe_valid_count = NULL;
    size_t *h_cqe_negative_count = NULL;
    size_t *h_cqe_over_cap_count = NULL;
    size_t *h_cqe_raw_clock_count = NULL;
    size_t *h_cqe_invalid_format_count = NULL;
    nvshmemx_ibgda_cqe_ts_pair_t *h_cqe_ts_pairs = NULL;
    uint64_t *h_cqe_delta_ns = NULL;

    /* Declared (without initializers) above any `goto finalize` so C++'s
     * goto-crosses-initialized-declaration rule doesn't fire. Filled in
     * later, before the timed run. */
    struct mlx5dv_clock_info ci;
    uint64_t ci_blob[8];
    memset(&ci, 0, sizeof(ci));
    memset(ci_blob, 0, sizeof(ci_blob));

    init_wrapper(&argc, &argv);

    if (use_cubin) {
        init_cumodule(CUMODULE_NAME);
        init_test_case_kernel(&test_cubin, "latency_kern");
        init_test_case_kernel(&test_cubin_warp, "latency_kern_warp");
        init_test_case_kernel(&test_cubin_block, "latency_kern_block");
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStreamCreateWithFlags(&calib_stream, cudaStreamNonBlocking);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    if (use_mmap) {
        data_d = (int *)allocate_mmap_buffer(max_size, mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        data_d = (int *)nvshmem_malloc(max_size);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
        CUDA_CHECK(cudaMemset(data_d, 0, max_size));
    }

    array_size = max_size_log;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_lat = (double *)h_tables[1];

    h_wqe_avg_ns = (double *)calloc(array_size, sizeof(double));
    h_wqe_max_ns = (double *)calloc(array_size, sizeof(double));
    h_db_avg_ns  = (double *)calloc(array_size, sizeof(double));
    h_db_max_ns  = (double *)calloc(array_size, sizeof(double));

    h_cqe2gpu_avg_ns = (double *)calloc(array_size, sizeof(double));
    h_cqe2gpu_max_ns = (double *)calloc(array_size, sizeof(double));
    h_cqe2gpu_p50_ns = (double *)calloc(array_size, sizeof(double));
    h_cqe2gpu_p95_ns = (double *)calloc(array_size, sizeof(double));
    h_nic_gpu_sync_precision_ns = (double *)calloc(array_size, sizeof(double));
    h_cqe_pair_count = (size_t *)calloc(array_size, sizeof(size_t));
    h_cqe_valid_count = (size_t *)calloc(array_size, sizeof(size_t));
    h_cqe_negative_count = (size_t *)calloc(array_size, sizeof(size_t));
    h_cqe_over_cap_count = (size_t *)calloc(array_size, sizeof(size_t));
    h_cqe_raw_clock_count = (size_t *)calloc(array_size, sizeof(size_t));
    h_cqe_invalid_format_count = (size_t *)calloc(array_size, sizeof(size_t));
    /* Up to `iter` pairs per measurement; allocate enough for the largest
     * expected sample. */
    h_cqe_ts_pairs =
        (nvshmemx_ibgda_cqe_ts_pair_t *)calloc((size_t)iter, sizeof(*h_cqe_ts_pairs));
    h_cqe_delta_ns = (uint64_t *)calloc((size_t)iter, sizeof(*h_cqe_delta_ns));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());

    i = 0;
    for (size = min_size; size <= max_size; size *= step_factor) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            test_latency(data_d, nelems, mype, skip, test_cubin, 1);

            /* The IBGDA CQ (and therefore the cached mlx5dv_clock_info) can be
             * created lazily by the first put/quiet. Fetch it after warmup,
             * before converting CQE timestamps. */
            if (ci.mask == 0 && nvshmemx_ibgda_get_mlx5_clock_info(ci_blob) == 0) {
                memcpy(&ci, ci_blob, sizeof(ci));
                normalize_mlx5_clock_info_mask(&ci);
            }
            if (nvshmemx_ibgda_query_mlx5_clock_info(ci_blob) == 0) {
                memcpy(&ci, ci_blob, sizeof(ci));
                normalize_mlx5_clock_info_mask(&ci);
            }

            const char *debug_cqe_ts = getenv("NVSHMEM_PERFTEST_CQE_TS_DEBUG");
            gpu_nic_clock_calibration_t gpu_nic_calib_before;
            gpu_nic_clock_calibration_t gpu_nic_calib_after;
            gpu_nic_clock_calibration_t gpu_nic_calib;
            gpu_nic_clock_calibration_t gpu_realtime_calib_before;
            gpu_nic_clock_calibration_t gpu_realtime_calib_after;
            gpu_nic_clock_calibration_t gpu_realtime_calib;
            init_gpu_nic_clock_calibration(&gpu_nic_calib_before);
            init_gpu_nic_clock_calibration(&gpu_nic_calib_after);
            init_gpu_nic_clock_calibration(&gpu_nic_calib);
            init_gpu_nic_clock_calibration(&gpu_realtime_calib_before);
            init_gpu_nic_clock_calibration(&gpu_realtime_calib_after);
            init_gpu_nic_clock_calibration(&gpu_realtime_calib);
            int calib_before_status = cudaErrorInvalidValue;
            int calib_after_status = cudaErrorInvalidValue;
            int calib_status = cudaErrorInvalidValue;
            int realtime_calib_before_status = cudaErrorInvalidValue;
            int realtime_calib_after_status = cudaErrorInvalidValue;
            int realtime_calib_status = cudaErrorInvalidValue;

            if (calib_stream) {
                calib_before_status =
                    calibrate_gpu_globaltimer_to_nic_ns(calib_stream, &gpu_nic_calib_before);
                realtime_calib_before_status = calibrate_gpu_globaltimer_to_realtime_ns(
                    calib_stream, &gpu_realtime_calib_before);
            }

            /* Reset the IBGDA latency profiling counters before the timed
             * run so the average reflects only the `iter` measured calls. */
            nvshmemx_ibgda_lat_profile_reset();
            /* Reset the CQE-timestamp ring buffer counter as well so the
             * pairs we fetch after the run correspond exactly to this size. */
            nvshmemx_ibgda_lat_profile_cqe_ts_reset();

            cudaEventRecord(start);
            test_latency(data_d, nelems, mype, iter, test_cubin, 1);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            /* give latency in us */
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_lat[i] = (milliseconds * 1000) / iter;

            if (calib_stream) {
                calib_after_status =
                    calibrate_gpu_globaltimer_to_nic_ns(calib_stream, &gpu_nic_calib_after);
                realtime_calib_after_status = calibrate_gpu_globaltimer_to_realtime_ns(
                    calib_stream, &gpu_realtime_calib_after);
            }
            calib_status = select_gpu_nic_calibration(
                calib_before_status, &gpu_nic_calib_before,
                calib_after_status, &gpu_nic_calib_after, &gpu_nic_calib);
            realtime_calib_status = select_gpu_nic_calibration(
                realtime_calib_before_status, &gpu_realtime_calib_before,
                realtime_calib_after_status, &gpu_realtime_calib_after, &gpu_realtime_calib);

            if (debug_cqe_ts && atoi(debug_cqe_ts) != 0) {
                printf("CQE_TS_DEBUG size=%d calib_status=%d calib_before_status=%d calib_after_status=%d realtime_calib_status=%d realtime_before_status=%d realtime_after_status=%d gpu_ref=%llu gpu_nic_ref_ns=%0.3Lf gpu_tick_ns=%0.12Lf nic_gpu_sync_precision_ns=%llu realtime_sync_precision_ns=%llu\n",
                       size, calib_status, calib_before_status, calib_after_status,
                       realtime_calib_status, realtime_calib_before_status,
                       realtime_calib_after_status,
                       (unsigned long long)gpu_nic_calib.gpu_ref,
                       gpu_nic_calib.nic_ref_ns,
                       gpu_nic_calib.nic_per_gpu_tick_ns,
                       (unsigned long long)gpu_nic_calib.fit_error_ns,
                       (unsigned long long)gpu_realtime_calib.fit_error_ns);
            }
            if (calib_status == 0 && gpu_nic_calib.fit_error_ns != UINT64_MAX) {
                h_nic_gpu_sync_precision_ns[i] = (double)gpu_nic_calib.fit_error_ns;
            }

            /* Snapshot the IBGDA WQE-prep / doorbell-submit cycle counters
             * accumulated by ibgda_rma_thread during this measurement. */
            nvshmemx_ibgda_lat_profile_t prof;
            int prof_status = nvshmemx_ibgda_lat_profile_get(&prof);
            if (prof_status == 0 && prof.count > 0 && prof.gpu_clock_hz > 0.0) {
                double cycle_to_ns = 1.0e9 / prof.gpu_clock_hz;
                h_wqe_avg_ns[i] =
                    ((double)prof.wqe_cycles_sum / (double)prof.count) * cycle_to_ns;
                h_db_avg_ns[i] =
                    ((double)prof.db_cycles_sum / (double)prof.count) * cycle_to_ns;
                h_wqe_max_ns[i] = (double)prof.wqe_cycles_max * cycle_to_ns;
                h_db_max_ns[i]  = (double)prof.db_cycles_max * cycle_to_ns;
            }

            /* Drain the CQE-timestamp pair buffer and convert each pair to a
             * NIC->GPU thread CQE-read ns delta in the NIC timestamp domain.
             * Skipped quietly if NVSHMEM was built without
             * NVSHMEM_IBGDA_LAT_PROFILE (count stays 0). */
            size_t pair_count = 0;
            if (h_cqe_ts_pairs && iter > 0) {
                int ts_status = nvshmemx_ibgda_lat_profile_cqe_ts_get(
                    h_cqe_ts_pairs, (size_t)iter, &pair_count);
                if (ts_status == 0) h_cqe_pair_count[i] = pair_count;
                if (debug_cqe_ts && atoi(debug_cqe_ts) != 0) {
                    printf("CQE_TS_DEBUG size=%d ts_status=%d pair_count=%zu ci.mask=%#lx\n",
                           size, ts_status, pair_count, (unsigned long)ci.mask);
                }
                if (ts_status == 0 && pair_count > 0) {
                    /* Refresh clock_info after the measured run for diagnostics. */
                    if (nvshmemx_ibgda_query_mlx5_clock_info(ci_blob) == 0) {
                        memcpy(&ci, ci_blob, sizeof(ci));
                        normalize_mlx5_clock_info_mask(&ci);
                    }
                    if (debug_cqe_ts && atoi(debug_cqe_ts) != 0) {
                        struct timespec host_realtime_ts;
                        clock_gettime(CLOCK_REALTIME, &host_realtime_ts);
                        printf("CQE_TS_DEBUG size=%d ci.nsec=%llu ci.last_cycles=%llu ci.frac=%llu ci.mult=%u ci.shift=%u ci.mask=%#lx host_realtime_ns=%llu\n",
                               size, (unsigned long long)ci.nsec,
                               (unsigned long long)ci.last_cycles,
                               (unsigned long long)ci.frac, ci.mult, ci.shift,
                               (unsigned long)ci.mask,
                               (unsigned long long)timespec_to_ns(&host_realtime_ts));
                    }
                    if (calib_status == 0 || realtime_calib_status == 0) {
                        struct timespec wall_ref_ts;
                        clock_gettime(CLOCK_REALTIME, &wall_ref_ts);
                        uint64_t wall_ref_sec = (uint64_t)wall_ref_ts.tv_sec;
                        uint64_t raw_ref_sec =
                            calib_status == 0
                                ? (uint64_t)(gpu_nic_calib.nic_ref_ns /
                                             (long double)NSEC_PER_SEC)
                                : 0;
                        long double sum_ns = 0.0L;
                        uint64_t max_ns = 0;
                        size_t valid_count = 0;
                        size_t negative_count = 0;
                        size_t over_cap_count = 0;
                        size_t raw_clock_count = 0;
                        size_t invalid_format_count = 0;
                        uint64_t active_fit_error_ns =
                            calib_status == 0 ? gpu_nic_calib.fit_error_ns : UINT64_MAX;
                        if (realtime_calib_status == 0 &&
                            (active_fit_error_ns == UINT64_MAX ||
                             gpu_realtime_calib.fit_error_ns > active_fit_error_ns)) {
                            active_fit_error_ns = gpu_realtime_calib.fit_error_ns;
                        }
                        uint64_t sanity_cap_ns =
                            (uint64_t)((h_lat[i] > 0.0 ? h_lat[i] * 1000.0 : 0.0) * 64.0);
                        uint64_t min_reasonable_cap_ns =
                            (active_fit_error_ns == UINT64_MAX ? 0ULL
                                                               : active_fit_error_ns * 8ULL) +
                            1000ULL;
                        if (sanity_cap_ns < min_reasonable_cap_ns) {
                            sanity_cap_ns = min_reasonable_cap_ns;
                        }

                        for (size_t k = 0; k < pair_count; ++k) {
                            uint64_t cqe_realtime_ns = 0;
                            uint64_t cqe_timestamp_ns = 0;
                            uint32_t cqe_sec = 0;
                            const gpu_nic_clock_calibration_t *timestamp_calib = NULL;
                            bool cqe_is_realtime = cqe_timestamp_realtime_ns(
                                h_cqe_ts_pairs[k].cqe_ts_cycles, &cqe_realtime_ns,
                                &cqe_sec, NULL) &&
                                cqe_realtime_sec_in_window((uint64_t)cqe_sec, wall_ref_sec);
                            if (cqe_is_realtime && realtime_calib_status == 0) {
                                cqe_timestamp_ns = cqe_realtime_ns;
                                timestamp_calib = &gpu_realtime_calib;
                            } else if (!cqe_is_realtime && calib_status == 0) {
                                uint64_t raw_clock_ns = h_cqe_ts_pairs[k].cqe_ts_cycles;
                                if (cqe_realtime_sec_in_window(raw_clock_ns / NSEC_PER_SEC,
                                                               raw_ref_sec)) {
                                    cqe_timestamp_ns = raw_clock_ns;
                                    timestamp_calib = &gpu_nic_calib;
                                    raw_clock_count++;
                                } else {
                                    invalid_format_count++;
                                    continue;
                                }
                            } else {
                                invalid_format_count++;
                                continue;
                            }
                            int64_t gpu_nic_ns = map_gpu_globaltimer_to_nic_ns(
                                timestamp_calib, h_cqe_ts_pairs[k].gpu_now_ns);
                            int64_t delta_ns = gpu_nic_ns - (int64_t)cqe_timestamp_ns;

                            if (delta_ns < 0) {
                                negative_count++;
                                continue;
                            }
                            if ((uint64_t)delta_ns > sanity_cap_ns) {
                                over_cap_count++;
                                continue;
                            }

                            sum_ns += (long double)delta_ns;
                            if ((uint64_t)delta_ns > max_ns) max_ns = (uint64_t)delta_ns;
                            if (h_cqe_delta_ns) h_cqe_delta_ns[valid_count] = (uint64_t)delta_ns;
                            valid_count++;
                        }

                        h_cqe_valid_count[i] = valid_count;
                        h_cqe_negative_count[i] = negative_count;
                        h_cqe_over_cap_count[i] = over_cap_count;
                        h_cqe_raw_clock_count[i] = raw_clock_count;
                        h_cqe_invalid_format_count[i] = invalid_format_count;
                        if (valid_count > 0) {
                            h_cqe2gpu_avg_ns[i] = (double)(sum_ns / (long double)valid_count);
                            h_cqe2gpu_max_ns[i] = (double)max_ns;
                            if (h_cqe_delta_ns) {
                                qsort(h_cqe_delta_ns, valid_count, sizeof(*h_cqe_delta_ns),
                                      compare_u64);
                                h_cqe2gpu_p50_ns[i] = (double)percentile_nearest_rank_u64(
                                    h_cqe_delta_ns, valid_count, 50);
                                h_cqe2gpu_p95_ns[i] = (double)percentile_nearest_rank_u64(
                                    h_cqe_delta_ns, valid_count, 95);
                            }
                        }
                        if (debug_cqe_ts && atoi(debug_cqe_ts) != 0) {
                            printf("CQE_TS_DEBUG size=%d valid_count=%zu negative_count=%zu over_cap_count=%zu raw_clock_count=%zu invalid_format_count=%zu wall_ref_sec=%llu raw_ref_sec=%llu sanity_cap_ns=%llu\n",
                                   size, valid_count, negative_count, over_cap_count,
                                   raw_clock_count, invalid_format_count,
                                   (unsigned long long)wall_ref_sec,
                                   (unsigned long long)raw_ref_sec,
                                   (unsigned long long)sanity_cap_ns);
                            for (size_t k = 0; k < pair_count && k < 4; ++k) {
                                uint32_t cqe_sec = 0;
                                uint32_t cqe_nsec = 0;
                                uint64_t cqe_realtime_ns = 0;
                                bool cqe_ts_ok = cqe_timestamp_realtime_ns(
                                    h_cqe_ts_pairs[k].cqe_ts_cycles, &cqe_realtime_ns,
                                    &cqe_sec, &cqe_nsec);
                                bool cqe_sec_ok =
                                    cqe_realtime_sec_in_window((uint64_t)cqe_sec,
                                                               wall_ref_sec);
                                uint64_t cqe_raw_clock_ns = h_cqe_ts_pairs[k].cqe_ts_cycles;
                                bool cqe_raw_clock_ok =
                                    calib_status == 0 &&
                                    cqe_realtime_sec_in_window(cqe_raw_clock_ns / NSEC_PER_SEC,
                                                               raw_ref_sec);
                                bool cqe_realtime_ok =
                                    cqe_ts_ok && cqe_sec_ok && realtime_calib_status == 0;
                                const gpu_nic_clock_calibration_t *timestamp_calib =
                                    cqe_realtime_ok ? &gpu_realtime_calib : &gpu_nic_calib;
                                int64_t gpu_nic_ns =
                                    (cqe_realtime_ok || cqe_raw_clock_ok)
                                        ? map_gpu_globaltimer_to_nic_ns(
                                              timestamp_calib, h_cqe_ts_pairs[k].gpu_now_ns)
                                        : 0;
                                uint64_t chosen_cqe_ns =
                                    cqe_realtime_ok ? cqe_realtime_ns : cqe_raw_clock_ns;
                                int64_t delta_ns = cqe_realtime_ok || cqe_raw_clock_ok
                                                       ? gpu_nic_ns - (int64_t)chosen_cqe_ns
                                                       : 0;
                                printf("CQE_TS_DEBUG sample[%zu] cqe_ts_raw=%llu cqe_sec=%u cqe_nsec=%u cqe_decode_ok=%d cqe_sec_ok=%d cqe_realtime_ns=%llu cqe_raw_clock_ok=%d cqe_raw_clock_ns=%llu gpu_ns=%llu gpu_nic_ns=%lld delta_ns=%lld\n",
                                       k,
                                       (unsigned long long)h_cqe_ts_pairs[k].cqe_ts_cycles,
                                       cqe_sec, cqe_nsec, cqe_ts_ok ? 1 : 0,
                                       cqe_sec_ok ? 1 : 0,
                                       (unsigned long long)cqe_realtime_ns,
                                       cqe_raw_clock_ok ? 1 : 0,
                                       (unsigned long long)cqe_raw_clock_ns,
                                       (unsigned long long)h_cqe_ts_pairs[k].gpu_now_ns,
                                       (long long)gpu_nic_ns,
                                       (long long)delta_ns);
                            }
                        }
                    } else if (debug_cqe_ts && atoi(debug_cqe_ts) != 0) {
                        printf("CQE_TS_DEBUG size=%d skipping CQE->GPU stats because calibration failed\n",
                               size);
                    }
                }
            }
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_table_basic("shmem_put_latency", "Thread", "size (Bytes)", "latency", "us", '-',
                          h_size_arr, h_lat, i);

        /* Print the IBGDA single-thread put path breakdown. If NVSHMEM was
         * not built with NVSHMEM_IBGDA_LAT_PROFILE=ON the rows will all be
         * zero - in that case nothing is printed (print_table_basic skips
         * zero-valued rows). */
        print_table_basic("shmem_put_latency_ibgda_breakdown", "wqe_avg",
                          "size (Bytes)", "wqe_prep_avg", "ns", '-',
                          h_size_arr, h_wqe_avg_ns, i);
        print_table_basic("shmem_put_latency_ibgda_breakdown", "wqe_max",
                          "size (Bytes)", "wqe_prep_max", "ns", '-',
                          h_size_arr, h_wqe_max_ns, i);
        print_table_basic("shmem_put_latency_ibgda_breakdown", "db_avg",
                          "size (Bytes)", "doorbell_submit_avg", "ns", '-',
                          h_size_arr, h_db_avg_ns, i);
        print_table_basic("shmem_put_latency_ibgda_breakdown", "db_max",
                          "size (Bytes)", "doorbell_submit_max", "ns", '-',
                          h_size_arr, h_db_max_ns, i);

        /* Print the NIC CQE-generation -> GPU-thread CQE-read deltas. Rows
         * stay at 0 if NVSHMEM was not built with NVSHMEM_IBGDA_LAT_PROFILE=ON,
         * GPU clock calibration failed, or the CQE timestamp format did not
         * match REAL_TIME sec/nsec or HCA raw_clock ns. */
        print_table_cqe_to_thread(h_size_arr, h_cqe2gpu_avg_ns, h_cqe2gpu_max_ns,
                                  h_cqe2gpu_p50_ns, h_cqe2gpu_p95_ns,
                                  h_cqe_pair_count, h_cqe_valid_count,
                                  h_cqe_negative_count, h_cqe_over_cap_count,
                                  h_cqe_raw_clock_count, h_cqe_invalid_format_count, i);
        print_table_basic("shmem_put_latency_nic_gpu_clock_sync", "precision",
                          "size (Bytes)", "nic_gpu_sync_precision", "ns", '-',
                          h_size_arr, h_nic_gpu_sync_precision_ns, i);
    }

    i = 0;
    for (size = min_size; size <= max_size; size *= step_factor) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            test_latency_warp(data_d, nelems, mype, skip, test_cubin_warp, THREADS_PER_WARP);
            cudaEventRecord(start);
            test_latency_warp(data_d, nelems, mype, iter, test_cubin_warp, THREADS_PER_WARP);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            /* give latency in us */
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_lat[i] = (milliseconds * 1000) / iter;
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_table_basic("shmem_put_latency", "Warp", "size (Bytes)", "latency", "us", '-',
                          h_size_arr, h_lat, i);
    }

    i = 0;
    for (size = min_size; size <= max_size; size *= step_factor) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            test_latency_block(data_d, nelems, mype, skip, test_cubin_block, threads_per_block);
            cudaEventRecord(start);
            test_latency_block(data_d, nelems, mype, iter, test_cubin_block, threads_per_block);
            cudaEventRecord(stop);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));

            /* give latency in us */
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_lat[i] = (milliseconds * 1000) / iter;
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_table_basic("shmem_put_latency", "Block", "size (Bytes)", "latency", "us", '-',
                          h_size_arr, h_lat, i);
    }

finalize:

    if (calib_stream) cudaStreamDestroy(calib_stream);

    if (data_d) {
        if (use_mmap) {
            free_mmap_buffer(data_d);
        } else {
            nvshmem_free(data_d);
        }
    }
    free_tables(h_tables, 2);

    free(h_wqe_avg_ns);
    free(h_wqe_max_ns);
    free(h_db_avg_ns);
    free(h_db_max_ns);

    free(h_cqe2gpu_avg_ns);
    free(h_cqe2gpu_max_ns);
    free(h_cqe2gpu_p50_ns);
    free(h_cqe2gpu_p95_ns);
    free(h_nic_gpu_sync_precision_ns);
    free(h_cqe_pair_count);
    free(h_cqe_valid_count);
    free(h_cqe_negative_count);
    free(h_cqe_over_cap_count);
    free(h_cqe_raw_clock_count);
    free(h_cqe_invalid_format_count);
    free(h_cqe_ts_pairs);
    free(h_cqe_delta_ns);

    finalize_wrapper();

    return 0;
}
