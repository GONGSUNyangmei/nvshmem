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

static __global__ void sample_globaltimer_kern(unsigned long long *out_ns) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long now_ns;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now_ns)::"memory");
        *out_ns = now_ns;
    }
}

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
    return ((uint64_t)ts->tv_sec * 1000000000ULL) + (uint64_t)ts->tv_nsec;
}

static inline void normalize_mlx5_clock_info_mask(struct mlx5dv_clock_info *ci) {
    (void)ci;
}

static inline uint64_t cqe_timestamp_raw_clock_ns(uint64_t raw_timestamp) {
    return raw_timestamp;
}

static void CUDART_CB record_realtime_ns_cb(void *user_data) {
    uint64_t *out_ns = (uint64_t *)user_data;
    struct timespec ts;

    clock_gettime(CLOCK_REALTIME, &ts);
    *out_ns = timespec_to_ns(&ts);
}

/* Keep the raw-clock helper for transport diagnostics. The CQE timestamp
 * field observed with this DevX CQ configuration tracks the same raw-clock
 * domain as ibv_query_rt_values_ex(...raw_clock...), so the production
 * calibration path lands in that domain directly instead of feeding the CQE
 * timestamp into mlx5dv_ts_to_ns(). */
static int query_hca_raw_clock_ns_cb(void *user_data) {
    unsigned long long *out_ns = (unsigned long long *)user_data;
    return nvshmemx_ibgda_query_mlx5_raw_clock_ns(out_ns);
}

static int query_mlx5_clock_info_cb(struct mlx5dv_clock_info *out_ci) {
    if (!out_ci) return (int)cudaErrorInvalidValue;
    return nvshmemx_ibgda_query_mlx5_clock_info((void *)out_ci);
}

static int calibrate_realtime_to_mlx5_ns(int64_t *offset_out_ns, uint64_t *half_window_out_ns) {
    const int warmup_samples = 4;
    const int measure_samples = 32;
    const int total_samples = warmup_samples + measure_samples;
    uint64_t best_window_ns = UINT64_MAX;
    int64_t best_offset_ns = 0;

    if (!offset_out_ns || !half_window_out_ns) return (int)cudaErrorInvalidValue;

    *offset_out_ns = 0;
    *half_window_out_ns = UINT64_MAX;

    for (int sample = 0; sample < total_samples; ++sample) {
        struct timespec before_ts, after_ts;
        struct mlx5dv_clock_info ci;
        memset(&ci, 0, sizeof(ci));

        clock_gettime(CLOCK_REALTIME, &before_ts);
        int rc = query_mlx5_clock_info_cb(&ci);
        clock_gettime(CLOCK_REALTIME, &after_ts);
        if (rc != 0) return rc;

        uint64_t before_ns = timespec_to_ns(&before_ts);
        uint64_t after_ns = timespec_to_ns(&after_ts);
        if (after_ns <= before_ns) continue;

        if (sample >= warmup_samples) {
            uint64_t window_ns = after_ns - before_ns;
            uint64_t midpoint_ns = before_ns + (window_ns / 2);
            int64_t candidate_offset_ns = (int64_t)ci.nsec - (int64_t)midpoint_ns;

            if (window_ns < best_window_ns) {
                best_window_ns = window_ns;
                best_offset_ns = candidate_offset_ns;
            }
        }
    }

    if (best_window_ns == UINT64_MAX) return (int)cudaErrorUnknown;
    *offset_out_ns = best_offset_ns;
    *half_window_out_ns = best_window_ns / 2;
    return 0;
}

static int calibrate_raw_clock_to_realtime_ns(int64_t *offset_out_ns,
                                              uint64_t *half_window_out_ns) {
    const int warmup_samples = 4;
    const int measure_samples = 32;
    const int total_samples = warmup_samples + measure_samples;
    uint64_t best_window_ns = UINT64_MAX;
    int64_t best_offset_ns = 0;

    if (!offset_out_ns || !half_window_out_ns) return (int)cudaErrorInvalidValue;

    *offset_out_ns = 0;
    *half_window_out_ns = UINT64_MAX;

    for (int sample = 0; sample < total_samples; ++sample) {
        struct timespec before_ts, after_ts;
        unsigned long long raw_clock_ns = 0ULL;

        clock_gettime(CLOCK_REALTIME, &before_ts);
        int rc = nvshmemx_ibgda_query_mlx5_raw_clock_ns(&raw_clock_ns);
        clock_gettime(CLOCK_REALTIME, &after_ts);
        if (rc != 0) return rc;

        uint64_t before_ns = timespec_to_ns(&before_ts);
        uint64_t after_ns = timespec_to_ns(&after_ts);
        if (after_ns <= before_ns) continue;

        if (sample >= warmup_samples) {
            uint64_t window_ns = after_ns - before_ns;
            uint64_t midpoint_ns = before_ns + (window_ns / 2);
            int64_t candidate_offset_ns = (int64_t)midpoint_ns - (int64_t)raw_clock_ns;

            if (window_ns < best_window_ns) {
                best_window_ns = window_ns;
                best_offset_ns = candidate_offset_ns;
            }
        }
    }

    if (best_window_ns == UINT64_MAX) return (int)cudaErrorUnknown;
    *offset_out_ns = best_offset_ns;
    *half_window_out_ns = best_window_ns / 2;
    return 0;
}

typedef struct gpu_raw_clock_calibration_s {
    unsigned long long gpu_ref;
    long double raw_ref_ns;
    long double raw_per_gpu_tick_ns;
    uint64_t fit_error_ns;
} gpu_raw_clock_calibration_t;

static inline int64_t map_gpu_globaltimer_to_raw_clock_ns(
    const gpu_raw_clock_calibration_t *calib, unsigned long long gpu_timestamp) {
    long double gpu_delta = (long double)((long long)gpu_timestamp - (long long)calib->gpu_ref);
    long double mapped_ns = calib->raw_ref_ns + gpu_delta * calib->raw_per_gpu_tick_ns;
    return (int64_t)llroundl(mapped_ns);
}

static int calibrate_gpu_globaltimer_to_raw_ns(cudaStream_t stream,
                                               gpu_raw_clock_calibration_t *calib_out) {
    const int warmup_samples = 4;
    const int measure_samples = 32;
    const int total_samples = warmup_samples + measure_samples;
    const char *debug_env = getenv("NVSHMEM_PERFTEST_CQE_TS_DEBUG");
    int debug = debug_env ? atoi(debug_env) : 0;
    gpu_calib_signal_t *signal_h = NULL;
    gpu_calib_signal_t *signal_d = NULL;
    unsigned long long gpu_samples[measure_samples];
    unsigned long long raw_midpoints_ns[measure_samples];
    uint64_t window_samples_ns[measure_samples];
    size_t sample_count = 0;
    cudaError_t err = cudaSuccess;

    if (!stream || !calib_out) return (int)cudaErrorInvalidValue;

    memset(calib_out, 0, sizeof(*calib_out));
    calib_out->raw_per_gpu_tick_ns = 1.0L;
    calib_out->fit_error_ns = UINT64_MAX;

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

        int rc = nvshmemx_ibgda_query_mlx5_raw_clock_ns(&before_ns);
        if (rc != 0) {
            err = (cudaError_t)rc;
            break;
        }

        signal_h->request = sample;
        __sync_synchronize();

        while (signal_h->ack != sample) {
        }

        rc = nvshmemx_ibgda_query_mlx5_raw_clock_ns(&after_ns);
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
                raw_midpoints_ns[sample_count] = midpoint_ns;
                window_samples_ns[sample_count] = window_ns;
                sample_count++;
            }

            if (debug >= 2) {
                printf("CQE_TS_DEBUG calib_raw sample=%u before_ns=%llu gpu_ns=%llu after_ns=%llu window_ns=%llu\n",
                       sample - (unsigned int)warmup_samples,
                       before_ns,
                       (unsigned long long)signal_h->gpu_now_ns,
                       after_ns,
                       (unsigned long long)window_ns);
            }
        }
    }

    signal_h->request = 0xffffffffu;
    __sync_synchronize();
    while (signal_h->ack != 0xffffffffu) {
    }
    cudaStreamSynchronize(stream);

    if (err == cudaSuccess && sample_count >= 2) {
        const unsigned long long gpu_ref = gpu_samples[0];
        const unsigned long long raw_ref = raw_midpoints_ns[0];
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
                (long double)((long long)raw_midpoints_ns[i] - (long long)raw_ref);
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
            long double predicted_ns =
                ((long double)raw_ref + intercept) + (x * slope);
            long double residual_ns =
                fabsl(predicted_ns - (long double)raw_midpoints_ns[i]);
            if (residual_ns > max_residual_ns) max_residual_ns = residual_ns;
            if (debug >= 2) {
                printf("CQE_TS_DEBUG calib_raw_fit sample=%zu x=%0.0Lf predicted_ns=%0.0Lf observed_ns=%llu residual_ns=%0.3Lf\n",
                       i + 1, x, predicted_ns,
                       (unsigned long long)raw_midpoints_ns[i], residual_ns);
            }
        }

        calib_out->gpu_ref = gpu_ref;
        calib_out->raw_ref_ns = (long double)raw_ref + intercept;
        calib_out->raw_per_gpu_tick_ns = slope;
        calib_out->fit_error_ns = (uint64_t)ceill(max_residual_ns);
        if (best_window_ns != UINT64_MAX &&
            calib_out->fit_error_ns < (best_window_ns / 2ULL)) {
            calib_out->fit_error_ns = best_window_ns / 2ULL;
        }
        if (debug >= 2) {
            printf("CQE_TS_DEBUG calib_raw best_window_ns=%llu gpu_ref=%llu raw_ref_ns=%0.3Lf slope=%0.12Lf fit_error_ns=%llu\n",
                   (unsigned long long)best_window_ns, gpu_ref, calib_out->raw_ref_ns,
                   calib_out->raw_per_gpu_tick_ns,
                   (unsigned long long)calib_out->fit_error_ns);
        }
    } else if (err == cudaSuccess) {
        err = cudaErrorUnknown;
    }

    cudaFreeHost(signal_h);
    return (int)err;
}

static int calibrate_gpu_globaltimer_to_realtime_ns(cudaStream_t stream, int64_t *offset_out_ns,
                                                    uint64_t *half_window_out_ns) {
    const int warmup_samples = 4;
    const int measure_samples = 32;
    const int total_samples = warmup_samples + measure_samples;
    gpu_calib_signal_t *signal_h = NULL;
    gpu_calib_signal_t *signal_d = NULL;
    uint64_t best_window_ns = UINT64_MAX;
    int64_t best_offset_ns = 0;
    cudaError_t err = cudaSuccess;

    if (!stream || !offset_out_ns || !half_window_out_ns) return (int)cudaErrorInvalidValue;

    *offset_out_ns = 0;
    *half_window_out_ns = UINT64_MAX;

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
        struct timespec before_ts, after_ts;

        clock_gettime(CLOCK_REALTIME, &before_ts);
        signal_h->request = sample;
        __sync_synchronize();

        while (signal_h->ack != sample) {
        }

        clock_gettime(CLOCK_REALTIME, &after_ts);

        uint64_t before_ns = timespec_to_ns(&before_ts);
        uint64_t after_ns = timespec_to_ns(&after_ts);
        if (after_ns <= before_ns) continue;

        if (sample > (unsigned int)warmup_samples) {
            uint64_t window_ns = after_ns - before_ns;
            uint64_t midpoint_ns = before_ns + (window_ns / 2);
            int64_t candidate_offset_ns =
                (int64_t)midpoint_ns - (int64_t)signal_h->gpu_now_ns;

            if (window_ns < best_window_ns) {
                best_window_ns = window_ns;
                best_offset_ns = candidate_offset_ns;
            }
        }
    }

    signal_h->request = 0xffffffffu;
    __sync_synchronize();
    while (signal_h->ack != 0xffffffffu) {
    }
    cudaStreamSynchronize(stream);

    if (err == cudaSuccess && best_window_ns != UINT64_MAX) {
        *offset_out_ns = best_offset_ns;
        *half_window_out_ns = best_window_ns / 2;
    } else if (err == cudaSuccess) {
        err = cudaErrorUnknown;
    }

    cudaFreeHost(signal_h);
    return (int)err;
}

static int calibrate_gpu_globaltimer_to_hca_raw_ns(cudaStream_t stream, int64_t *offset_out_ns,
                                                   uint64_t *half_window_out_ns) {
    const int warmup_samples = 4;
    const int measure_samples = 32;
    const int total_samples = warmup_samples + measure_samples;
    unsigned long long *d_gpu_now_ns = NULL;
    unsigned long long *h_gpu_now_ns = NULL;
    uint64_t best_window_ns = UINT64_MAX;
    int64_t best_offset_ns = 0;
    cudaError_t err = cudaSuccess;

    if (!stream || !offset_out_ns || !half_window_out_ns) return (int)cudaErrorInvalidValue;

    *offset_out_ns = 0;
    *half_window_out_ns = UINT64_MAX;

    err = cudaMalloc((void **)&d_gpu_now_ns, sizeof(*d_gpu_now_ns));
    if (err != cudaSuccess) return (int)err;

    err = cudaMallocHost((void **)&h_gpu_now_ns, sizeof(*h_gpu_now_ns));
    if (err != cudaSuccess) {
        cudaFree(d_gpu_now_ns);
        return (int)err;
    }

    for (int sample = 0; sample < total_samples; ++sample) {
        unsigned long long before_ns = 0;
        unsigned long long after_ns = 0;

        int rc = query_hca_raw_clock_ns_cb(&before_ns);
        if (rc != 0) {
            err = (cudaError_t)rc;
            break;
        }

        sample_globaltimer_kern<<<1, 1, 0, stream>>>(d_gpu_now_ns);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;

        err = cudaMemcpyAsync(h_gpu_now_ns, d_gpu_now_ns, sizeof(*h_gpu_now_ns),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) break;

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) break;

        rc = query_hca_raw_clock_ns_cb(&after_ns);
        if (rc != 0) {
            err = (cudaError_t)rc;
            break;
        }
        if (after_ns <= before_ns) continue;

        if (sample >= warmup_samples) {
            uint64_t window_ns = after_ns - before_ns;
            uint64_t midpoint_ns = before_ns + (window_ns / 2);
            int64_t candidate_offset_ns = (int64_t)midpoint_ns - (int64_t)(*h_gpu_now_ns);

            if (window_ns < best_window_ns) {
                best_window_ns = window_ns;
                best_offset_ns = candidate_offset_ns;
            }
        }
    }

    if (err == cudaSuccess && best_window_ns != UINT64_MAX) {
        *offset_out_ns = best_offset_ns;
        *half_window_out_ns = best_window_ns / 2;
    } else if (err == cudaSuccess) {
        err = cudaErrorUnknown;
    }

    cudaFreeHost(h_gpu_now_ns);
    cudaFree(d_gpu_now_ns);
    return (int)err;
}

static int calibrate_gpu_globaltimer_to_mlx5_ns(cudaStream_t stream, int64_t *offset_out_ns,
                                                uint64_t *half_window_out_ns) {
    const int warmup_samples = 4;
    const int measure_samples = 32;
    const int total_samples = warmup_samples + measure_samples;
    unsigned long long *d_gpu_now_ns = NULL;
    unsigned long long *h_gpu_now_ns = NULL;
    uint64_t best_window_ns = UINT64_MAX;
    int64_t best_offset_ns = 0;
    cudaError_t err = cudaSuccess;

    if (!stream || !offset_out_ns || !half_window_out_ns) return (int)cudaErrorInvalidValue;

    *offset_out_ns = 0;
    *half_window_out_ns = UINT64_MAX;

    err = cudaMalloc((void **)&d_gpu_now_ns, sizeof(*d_gpu_now_ns));
    if (err != cudaSuccess) return (int)err;

    err = cudaMallocHost((void **)&h_gpu_now_ns, sizeof(*h_gpu_now_ns));
    if (err != cudaSuccess) {
        cudaFree(d_gpu_now_ns);
        return (int)err;
    }

    for (int sample = 0; sample < total_samples; ++sample) {
        struct mlx5dv_clock_info before_ci;
        struct mlx5dv_clock_info after_ci;
        memset(&before_ci, 0, sizeof(before_ci));
        memset(&after_ci, 0, sizeof(after_ci));

        int rc = query_mlx5_clock_info_cb(&before_ci);
        if (rc != 0) {
            err = (cudaError_t)rc;
            break;
        }

        sample_globaltimer_kern<<<1, 1, 0, stream>>>(d_gpu_now_ns);
        err = cudaGetLastError();
        if (err != cudaSuccess) break;

        err = cudaMemcpyAsync(h_gpu_now_ns, d_gpu_now_ns, sizeof(*h_gpu_now_ns),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) break;

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) break;

        rc = query_mlx5_clock_info_cb(&after_ci);
        if (rc != 0) {
            err = (cudaError_t)rc;
            break;
        }
        if (after_ci.nsec <= before_ci.nsec) continue;

        if (sample >= warmup_samples) {
            uint64_t window_ns = after_ci.nsec - before_ci.nsec;
            uint64_t midpoint_ns = before_ci.nsec + (window_ns / 2);
            int64_t candidate_offset_ns = (int64_t)midpoint_ns - (int64_t)(*h_gpu_now_ns);

            if (window_ns < best_window_ns) {
                best_window_ns = window_ns;
                best_offset_ns = candidate_offset_ns;
            }
        }
    }

    if (err == cudaSuccess && best_window_ns != UINT64_MAX) {
        *offset_out_ns = best_offset_ns;
        *half_window_out_ns = best_window_ns / 2;
    } else if (err == cudaSuccess) {
        err = cudaErrorUnknown;
    }

    cudaFreeHost(h_gpu_now_ns);
    cudaFree(d_gpu_now_ns);
    return (int)err;
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
     * from inside ibgda_poll_cq; here on the host we compare the CQE
     * timestamp directly against the HCA raw-clock domain reported by
     * ibv_query_rt_values_ex(...raw_clock...) and independently calibrate
     * the GPU %globaltimer into that same raw-clock ns domain with a narrow
     * stream-ordered host/device bracket. Values stay 0 if NVSHMEM was not
     * built with NVSHMEM_IBGDA_LAT_PROFILE=ON or if the GPU clock
     * calibration is too noisy to trust. */
    double *h_cqe2gpu_avg_ns = NULL;
    double *h_cqe2gpu_max_ns = NULL;
    nvshmemx_ibgda_cqe_ts_pair_t *h_cqe_ts_pairs = NULL;

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
    /* Up to `iter` pairs per measurement; allocate enough for the largest
     * expected sample. */
    h_cqe_ts_pairs =
        (nvshmemx_ibgda_cqe_ts_pair_t *)calloc((size_t)iter, sizeof(*h_cqe_ts_pairs));

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
             * NIC->GPU thread CQE-read ns delta in the HCA raw-clock domain.
             * Skipped quietly if NVSHMEM was built without
             * NVSHMEM_IBGDA_LAT_PROFILE (count stays 0). */
            size_t pair_count = 0;
            if (h_cqe_ts_pairs && iter > 0) {
                int ts_status = nvshmemx_ibgda_lat_profile_cqe_ts_get(
                    h_cqe_ts_pairs, (size_t)iter, &pair_count);
                const char *debug_cqe_ts = getenv("NVSHMEM_PERFTEST_CQE_TS_DEBUG");
                if (debug_cqe_ts && atoi(debug_cqe_ts) != 0) {
                    printf("CQE_TS_DEBUG size=%d ts_status=%d pair_count=%zu ci.mask=%#lx\n",
                           size, ts_status, pair_count, (unsigned long)ci.mask);
                }
                if (ts_status == 0 && pair_count > 0) {
                    /* Refresh clock_info after the measured run for
                     * diagnostics only. The CQE timestamp path below compares
                     * directly against the raw HCA clock domain. */
                    if (nvshmemx_ibgda_query_mlx5_clock_info(ci_blob) == 0) {
                        memcpy(&ci, ci_blob, sizeof(ci));
                        normalize_mlx5_clock_info_mask(&ci);
                    }
                    if (debug_cqe_ts && atoi(debug_cqe_ts) != 0) {
                        unsigned long long raw_clock_ns = 0ULL;
                        int raw_clock_status = nvshmemx_ibgda_query_mlx5_raw_clock_ns(&raw_clock_ns);
                        printf("CQE_TS_DEBUG size=%d ci.nsec=%llu ci.last_cycles=%llu ci.frac=%llu ci.mult=%u ci.shift=%u ci.mask=%#lx raw_clock_status=%d raw_clock_ns=%llu\n",
                               size, (unsigned long long)ci.nsec,
                               (unsigned long long)ci.last_cycles,
                               (unsigned long long)ci.frac, ci.mult, ci.shift,
                               (unsigned long)ci.mask, raw_clock_status, raw_clock_ns);
                    }

                    gpu_raw_clock_calibration_t gpu_raw_calib;
                    memset(&gpu_raw_calib, 0, sizeof(gpu_raw_calib));
                    gpu_raw_calib.raw_per_gpu_tick_ns = 1.0L;
                    gpu_raw_calib.fit_error_ns = UINT64_MAX;
                    int calib_status = cudaErrorInvalidValue;

                    if (calib_stream) {
                        calib_status =
                            calibrate_gpu_globaltimer_to_raw_ns(calib_stream, &gpu_raw_calib);
                    }
                    if (debug_cqe_ts && atoi(debug_cqe_ts) != 0) {
                        printf("CQE_TS_DEBUG size=%d calib_status=%d gpu_ref=%llu gpu_raw_ref_ns=%0.3Lf gpu_tick_ns=%0.12Lf gpu_fit_error_ns=%llu\n",
                               size, calib_status,
                               (unsigned long long)gpu_raw_calib.gpu_ref,
                               gpu_raw_calib.raw_ref_ns,
                               gpu_raw_calib.raw_per_gpu_tick_ns,
                               (unsigned long long)gpu_raw_calib.fit_error_ns);
                    }
                    if (calib_status == 0) {
                        long double sum_ns = 0.0L;
                        uint64_t max_ns = 0;
                        size_t valid_count = 0;
                        uint64_t sanity_cap_ns =
                            (uint64_t)((h_lat[i] > 0.0 ? h_lat[i] * 1000.0 : 0.0) * 64.0);
                        uint64_t min_reasonable_cap_ns =
                            (gpu_raw_calib.fit_error_ns * 8ULL) + 1000ULL;
                        if (sanity_cap_ns < min_reasonable_cap_ns) {
                            sanity_cap_ns = min_reasonable_cap_ns;
                        }

                        for (size_t k = 0; k < pair_count; ++k) {
                            int64_t cqe_raw_ns = (int64_t)cqe_timestamp_raw_clock_ns(
                                h_cqe_ts_pairs[k].cqe_ts_cycles);
                            int64_t gpu_raw_ns = map_gpu_globaltimer_to_raw_clock_ns(
                                &gpu_raw_calib, h_cqe_ts_pairs[k].gpu_now_ns);
                            int64_t delta_ns = gpu_raw_ns - cqe_raw_ns;

                            if (delta_ns < 0) continue;
                            if ((uint64_t)delta_ns > sanity_cap_ns) continue;

                            sum_ns += (long double)delta_ns;
                            if ((uint64_t)delta_ns > max_ns) max_ns = (uint64_t)delta_ns;
                            valid_count++;
                        }

                        if (valid_count > 0) {
                            h_cqe2gpu_avg_ns[i] = (double)(sum_ns / (long double)valid_count);
                            h_cqe2gpu_max_ns[i] = (double)max_ns;
                        }
                        if (debug_cqe_ts && atoi(debug_cqe_ts) != 0) {
                            printf("CQE_TS_DEBUG size=%d valid_count=%zu sanity_cap_ns=%llu\n",
                                   size, valid_count, (unsigned long long)sanity_cap_ns);
                            for (size_t k = 0; k < pair_count && k < 4; ++k) {
                                int64_t cqe_raw_ns = (int64_t)cqe_timestamp_raw_clock_ns(
                                    h_cqe_ts_pairs[k].cqe_ts_cycles);
                                int64_t gpu_raw_ns = map_gpu_globaltimer_to_raw_clock_ns(
                                    &gpu_raw_calib, h_cqe_ts_pairs[k].gpu_now_ns);
                                int64_t delta_ns = gpu_raw_ns - cqe_raw_ns;
                                printf("CQE_TS_DEBUG sample[%zu] cqe_ts_raw=%llu cqe_raw_ns=%lld gpu_ns=%llu gpu_raw_ns=%lld delta_ns=%lld\n",
                                       k,
                                       (unsigned long long)h_cqe_ts_pairs[k].cqe_ts_cycles,
                                       (long long)cqe_raw_ns,
                                       (unsigned long long)h_cqe_ts_pairs[k].gpu_now_ns,
                                       (long long)gpu_raw_ns,
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
         * stay at 0 (and print_table_basic skips them) if NVSHMEM was not
         * built with NVSHMEM_IBGDA_LAT_PROFILE=ON, or if the IBGDA transport
         * never published mlx5dv_clock_info (e.g. running on a non-mlx5
         * NIC). */
        print_table_basic("shmem_put_latency_cqe_to_thread", "avg",
                          "size (Bytes)", "nic_cqe_to_gpu_avg", "ns", '-',
                          h_size_arr, h_cqe2gpu_avg_ns, i);
        print_table_basic("shmem_put_latency_cqe_to_thread", "max",
                          "size (Bytes)", "nic_cqe_to_gpu_max", "ns", '-',
                          h_size_arr, h_cqe2gpu_max_ns, i);
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
    free(h_cqe_ts_pairs);

    finalize_wrapper();

    return 0;
}
