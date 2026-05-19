/*
 * Copyright (c) 2017-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */
#ifndef _NVSHMEM_INIT_DEVICE_CUH_
#define _NVSHMEM_INIT_DEVICE_CUH_

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <dlfcn.h>

#if defined(__clang_llvm_bitcode_lib__) || defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
#if !defined(assert)
#define assert(...)
#endif
#include "nvshmem.h"
#endif

#include "non_abi/nvshmem_build_options.h"
#include "non_abi/nvshmem_version.h"
#include "non_abi/nvshmemx_error.h"
#include "internal/device/nvshmemi_device.h"
#include "internal/common/error_codes_internal.h"
#include "non_abi/device/pt-to-pt/proxy_device.cuh"
#include "device_host/nvshmem_common.cuh"
#include "device_host/nvshmem_types.h"
#include "host/nvshmemx_ibgda_lat_profile.h"
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#ifdef NVSHMEM_IBGDA_SUPPORT
#include "device_host_transport/nvshmem_common_ibgda.h"
#if defined(__clang_llvm_bitcode_lib__)
#if defined(__CUDACC__)
// Clang CUDA mode: use __constant__ only (no address_space to avoid LLVM21 conflict)
__constant__ __attribute__((used)) nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d = {};
#else
// Plain Clang-to-NVPTX bitcode: use address_space(4) only (no __constant__)
__attribute__((address_space(4),
               used)) nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d = {};
#endif
#else
// Normal CUDA/nvcc build
__constant__ __attribute__((used)) nvshmemi_ibgda_device_state_t nvshmemi_ibgda_device_state_d;
#endif

/* IBGDA latency profiling counters. Always defined so the host symbol is
   stable regardless of whether the on-device sampling code (gated by
   NVSHMEM_IBGDA_LAT_PROFILE in ibgda_device.cuh) is compiled in. The host
   reads/resets them through nvshmemx_ibgda_lat_profile_get() / _reset(). */
__device__ __attribute__((used)) unsigned long long nvshmemi_ibgda_lat_prof_wqe_sum = 0;
__device__ __attribute__((used)) unsigned long long nvshmemi_ibgda_lat_prof_wqe_max = 0;
__device__ __attribute__((used)) unsigned long long nvshmemi_ibgda_lat_prof_db_sum  = 0;
__device__ __attribute__((used)) unsigned long long nvshmemi_ibgda_lat_prof_db_max  = 0;
__device__ __attribute__((used)) unsigned long long nvshmemi_ibgda_lat_prof_count   = 0;

/* Per-iter CQE-timestamp ring buffer used by ibgda_poll_cq when built with
   NVSHMEM_IBGDA_LAT_PROFILE. The buffers themselves are host-allocated
   (cudaMalloc) and published via cudaMemcpyToSymbol from
   nvshmemx_ibgda_lat_profile_cqe_ts_reset(); the slot count cap is also
   pushed once. The device-side recorder writes (cqe_ts_cycles, gpu_now_ns)
   pairs into matching slots while count < cap. */
__device__ __attribute__((used)) unsigned long long *nvshmemi_ibgda_cqe_ts_buf  = nullptr;
__device__ __attribute__((used)) unsigned long long *nvshmemi_ibgda_gpu_ns_buf  = nullptr;
__device__ __attribute__((used)) unsigned long long  nvshmemi_ibgda_cqe_ts_cap   = 0;
__device__ __attribute__((used)) unsigned long long  nvshmemi_ibgda_cqe_ts_count = 0;

/* Host-published mlx5dv_clock_info, populated by the IBGDA transport via
   nvshmemx_ibgda_publish_mlx5_clock_info() the first time a CQ is created.
   Stored as 8x u64 (== 64 bytes), which is >= sizeof(struct mlx5dv_clock_info)
   in any current libmlx5; we never reinterpret on device, only memcpy back
   on host via nvshmemx_ibgda_get_mlx5_clock_info(). */
__device__ __attribute__((used)) unsigned long long nvshmemi_ibgda_mlx5_clock_info[8] = {0};
__device__ __attribute__((used)) int                nvshmemi_ibgda_mlx5_clock_info_valid = 0;
#endif

nvshmemi_device_state_t nvshmemi_device_only_state;

#if defined(__clang_llvm_bitcode_lib__) || defined(NVSHMEM_BUILD_LTOIR_LIBRARY)
#if defined(__CUDACC__)
// Clang CUDA mode: use __constant__ only (no address_space to avoid LLVM21 conflict)
__constant__ __attribute__((used)) nvshmemi_device_host_state_t nvshmemi_device_state_d = {};

__constant__ __attribute__((used)) nvshmemi_version_t nvshmemi_device_lib_version_d = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};
#else
// Plain Clang-to-NVPTX bitcode: use address_space(4) only (no __constant__)
__attribute__((address_space(4), used)) nvshmemi_device_host_state_t nvshmemi_device_state_d = {};

__attribute__((address_space(4), used)) nvshmemi_version_t nvshmemi_device_lib_version_d = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};
#endif
#else
// Normal CUDA/nvcc build
__constant__ nvshmemi_device_host_state_t nvshmemi_device_state_d;

__constant__ nvshmemi_version_t nvshmemi_device_lib_version_d = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};
#endif

const nvshmemi_version_t nvshmemi_device_lib_version = {
    NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION, NVSHMEM_VENDOR_PATCH_VERSION};

extern nvshmemi_selected_device_transport_t nvshmem_selected_device_transport;

#ifdef __CUDA_ARCH__
#ifdef __cplusplus
extern "C" {
#endif
NVSHMEMI_DEVICE_PREFIX void nvshmem_global_exit(int status);
#ifdef __cplusplus
}
#endif

NVSHMEMI_DEVICE_PREFIX void nvshmem_global_exit(int status) {
    if (nvshmemi_device_state_d.proxy > NVSHMEMI_PROXY_NONE) {
        nvshmemi_proxy_global_exit(status);
    } else {
        /* TODO: Add device side printing macros */
        printf(
            "Device side proxy was called, but is not supported under your configuration. "
            "Please unset NVSHMEM_DISABLE_LOCAL_ONLY_PROXY, or set it to false.\n");
        assert(0);
    }
}
#endif

#ifdef __cplusplus
extern "C" {
int nvshmemi_get_device_state_ptrs(void **dev_state_ptr, void **transport_dev_state_ptr);
}
#endif

static int _nvshmemi_init_device_only_state() {
    int status = 0;
    status = nvshmemi_setup_collective_launch();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "_nvshmemi_init_device_only_state failed\n");
    nvshmemi_device_only_state.is_initialized = true;

out:
    return status;
}

int nvshmemi_check_state_and_init_d() {
    int status = NVSHMEMI_SUCCESS;
    int ret;

    if (nvshmemid_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
        fprintf(stderr, "nvshmem API called before nvshmem_init \n");
        return NVSHMEMI_NOT_INITIALIZED;
    }
    if (nvshmemid_init_status() == NVSHMEM_STATUS_IS_BOOTSTRAPPED) {
        /* The fact that we can pass NVSHMEM_THREAD_SERIALIZED
         * here is an implementation detail. It should be fixed
         * if/when NVSHMEM_THREAD_* becomes significant. */
        status = nvshmemid_hostlib_init_attr(NVSHMEM_THREAD_SERIALIZED, &ret, 0, NULL,
                                             nvshmemi_device_lib_version, NULL);
        if (status) {
            fprintf(stderr, "nvshmem initialization failed, exiting \n");
            return NVSHMEMI_NOT_INITIALIZED;
        }

        status = cudaGetDevice(&nvshmemi_device_only_state.cuda_device_id);
        if (status) {
            fprintf(stderr, "nvshmem cuda device query failed, exiting \n");
            return NVSHMEMI_CUDA_GET_DEVICE_FAILED;
        }

        nvshmemid_hostlib_finalize(NULL, NULL);  // for refcounting
    }

    if (!nvshmemi_device_only_state.is_initialized) {
        status = _nvshmemi_init_device_only_state();
        if (status) {
            fprintf(stderr, "nvshmem device initialization failed, exiting \n");
            return NVSHMEMI_INIT_DEVICE_ONLY_STATE_FAILED;
        }
    }

    return status;
}

int nvshmemi_get_device_state_ptrs(void **dev_state_ptr, void **transport_dev_state_ptr) {
    int status = NVSHMEMI_SUCCESS;
    status = cudaGetSymbolAddress(dev_state_ptr, nvshmemi_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to access device state. %d\n", status);
        *dev_state_ptr = NULL;
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    // Get transport device pointer (if required)
    switch (nvshmem_selected_device_transport) {
#ifdef NVSHMEM_IBGDA_SUPPORT
        case NVSHMEMI_DEVICE_TRANSPORT_TYPE_IBGDA:
            status = cudaGetSymbolAddress(transport_dev_state_ptr, nvshmemi_ibgda_device_state_d);
            if (status) {
                NVSHMEMI_ERROR_PRINT("Unable to access ibgda device state. %d\n", status);
                *transport_dev_state_ptr = NULL;
                status = NVSHMEMX_ERROR_INTERNAL;
            }
            break;
#endif
        default:
            *transport_dev_state_ptr = NULL;
            break;
    }

out:
    return status;
}

int nvshmemi_init_thread(int requested_thread_support, int *provided_thread_support,
                         unsigned int bootstrap_flags, nvshmemx_init_attr_t *bootstrap_attr,
                         nvshmemi_version_t nvshmem_app_version) {
    int status = 0;

#ifdef _NVSHMEM_DEBUG
    printf("  %-28s %d\n", "DEVICE CUDA API", CUDART_VERSION);
#endif
    status = nvshmemid_hostlib_init_attr(
        requested_thread_support, provided_thread_support, bootstrap_flags, bootstrap_attr,
        nvshmemi_device_lib_version, &nvshmemi_get_device_state_ptrs);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmem_internal_init_thread failed \n");

    if (nvshmemid_init_status() > NVSHMEM_STATUS_IS_BOOTSTRAPPED) {
        status = _nvshmemi_init_device_only_state();
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmem_internal_init_thread failed at init_device_only_state.\n");

        status = cudaGetDevice(&nvshmemi_device_only_state.cuda_device_id);
        if (status) {
            NVSHMEMI_ERROR_EXIT("nvshmem cuda device query failed, exiting \n");
        }
    }

out:
    return status;
}

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NVSHMEM_IBGDA_SUPPORT
/* Host accessors for the IBGDA latency profiling counters defined above.
 * They live in this TU so cudaMemcpyFromSymbol() / cudaMemcpyToSymbol() can
 * resolve the device variables by direct symbol reference, matching the
 * pattern used by nvshmemi_get_device_state_ptrs() in this file. The
 * functions are still exported from libnvshmem_host.so because nvshmem_device
 * is statically linked into nvshmem_host and the names match nvshmemx_*
 * in nvshmem_host.sym. */
int nvshmemx_ibgda_lat_profile_get(nvshmemx_ibgda_lat_profile_t *out) {
    if (!out) return (int)cudaErrorInvalidValue;
    memset(out, 0, sizeof(*out));

    cudaError_t err;
    unsigned long long val;

    err = cudaMemcpyFromSymbol(&val, nvshmemi_ibgda_lat_prof_wqe_sum, sizeof(val), 0,
                               cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;
    out->wqe_cycles_sum = val;

    err = cudaMemcpyFromSymbol(&val, nvshmemi_ibgda_lat_prof_wqe_max, sizeof(val), 0,
                               cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;
    out->wqe_cycles_max = val;

    err = cudaMemcpyFromSymbol(&val, nvshmemi_ibgda_lat_prof_db_sum, sizeof(val), 0,
                               cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;
    out->db_cycles_sum = val;

    err = cudaMemcpyFromSymbol(&val, nvshmemi_ibgda_lat_prof_db_max, sizeof(val), 0,
                               cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;
    out->db_cycles_max = val;

    err = cudaMemcpyFromSymbol(&val, nvshmemi_ibgda_lat_prof_count, sizeof(val), 0,
                               cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;
    out->count = val;

    int dev = 0;
    err = cudaGetDevice(&dev);
    if (err == cudaSuccess) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err == cudaSuccess) {
            /* prop.clockRate is the SM clock in kHz. */
            out->gpu_clock_hz = (double)prop.clockRate * 1000.0;
        }
    }
    return 0;
}

int nvshmemx_ibgda_lat_profile_reset(void) {
    cudaError_t err;
    const unsigned long long zero = 0;

    err = cudaMemcpyToSymbol(nvshmemi_ibgda_lat_prof_wqe_sum, &zero, sizeof(zero), 0,
                             cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemcpyToSymbol(nvshmemi_ibgda_lat_prof_wqe_max, &zero, sizeof(zero), 0,
                             cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemcpyToSymbol(nvshmemi_ibgda_lat_prof_db_sum, &zero, sizeof(zero), 0,
                             cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemcpyToSymbol(nvshmemi_ibgda_lat_prof_db_max, &zero, sizeof(zero), 0,
                             cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemcpyToSymbol(nvshmemi_ibgda_lat_prof_count, &zero, sizeof(zero), 0,
                             cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  CQE-timestamp ring-buffer host APIs.                              */
/*                                                                    */
/*  The on-device recorder in ibgda_poll_cq writes (cqe_ts_cycles,    */
/*  gpu_now_ns) pairs into matching slots in two parallel uint64_t    */
/*  arrays. The arrays themselves live in plain device memory and are */
/*  lazily allocated on the first call to _reset(). The cap defaults  */
/*  to 64K slots and can be overridden with                           */
/*  NVSHMEM_IBGDA_CQE_TS_BUF_CAP.                                     */
/* ------------------------------------------------------------------ */

static unsigned long long *s_cqe_ts_dev_buf  = nullptr;
static unsigned long long *s_gpu_ns_dev_buf  = nullptr;
static size_t              s_cqe_ts_cap      = 0;
static struct ibv_context *s_ibgda_host_context = nullptr;

typedef int (*nvshmemi_ibgda_transport_query_clock_info_fn_t)(void *out, size_t sz);
typedef int (*nvshmemi_ibgda_transport_query_raw_clock_ns_fn_t)(unsigned long long *out_ns);

static inline int nvshmemi_ibgda_clock_debug_enabled(void) {
    const char *env = getenv("NVSHMEM_PERFTEST_CQE_TS_DEBUG");
    return env && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0');
}

static void *nvshmemi_ibgda_transport_handle(void) {
    static void *s_transport_handle = (void *)(uintptr_t)-1;

    if (s_transport_handle == (void *)(uintptr_t)-1) {
        s_transport_handle = dlopen("nvshmem_transport_ibgda.so.5", RTLD_NOW | RTLD_NOLOAD);
        if (!s_transport_handle) {
            s_transport_handle = dlopen("nvshmem_transport_ibgda.so", RTLD_NOW | RTLD_NOLOAD);
        }
        if (!s_transport_handle) {
            s_transport_handle =
                dlopen("nvshmem_transport_ibgda.so.5.0.0", RTLD_NOW | RTLD_NOLOAD);
        }
        if (!s_transport_handle) s_transport_handle = nullptr;

        if (nvshmemi_ibgda_clock_debug_enabled()) {
            fprintf(stderr, "IBGDA_CLOCK_DEBUG transport_handle=%p\n", s_transport_handle);
        }
    }

    return s_transport_handle;
}

static int nvshmemi_ibgda_query_mlx5_clock_info_from_transport(void *out) {
    if (!out) return (int)cudaErrorInvalidValue;

    void *handle = nvshmemi_ibgda_transport_handle();
    if (!handle) return (int)cudaErrorNotReady;

    dlerror();
    nvshmemi_ibgda_transport_query_clock_info_fn_t fn =
        (nvshmemi_ibgda_transport_query_clock_info_fn_t)dlsym(
            handle, "nvshmemi_ibgda_query_mlx5_clock_info");
    const char *sym_err = dlerror();
    if (fn == nullptr || sym_err != nullptr) {
        if (nvshmemi_ibgda_clock_debug_enabled()) {
            fprintf(stderr, "IBGDA_CLOCK_DEBUG transport clock_info symbol missing: %s\n",
                    sym_err ? sym_err : "(null)");
        }
        return (int)cudaErrorNotReady;
    }

    int rc = fn(out, sizeof(struct mlx5dv_clock_info));
    if (nvshmemi_ibgda_clock_debug_enabled()) {
        fprintf(stderr, "IBGDA_CLOCK_DEBUG transport clock_info rc=%d\n", rc);
    }
    return rc;
}

static int nvshmemi_ibgda_query_mlx5_raw_clock_ns_from_transport(unsigned long long *out_ns) {
    if (!out_ns) return (int)cudaErrorInvalidValue;

    void *handle = nvshmemi_ibgda_transport_handle();
    if (!handle) return (int)cudaErrorNotReady;

    dlerror();
    nvshmemi_ibgda_transport_query_raw_clock_ns_fn_t fn =
        (nvshmemi_ibgda_transport_query_raw_clock_ns_fn_t)dlsym(
            handle, "nvshmemi_ibgda_query_mlx5_raw_clock_ns");
    const char *sym_err = dlerror();
    if (fn == nullptr || sym_err != nullptr) {
        if (nvshmemi_ibgda_clock_debug_enabled()) {
            fprintf(stderr, "IBGDA_CLOCK_DEBUG transport raw_clock symbol missing: %s\n",
                    sym_err ? sym_err : "(null)");
        }
        return (int)cudaErrorNotReady;
    }

    int rc = fn(out_ns);
    if (nvshmemi_ibgda_clock_debug_enabled()) {
        fprintf(stderr, "IBGDA_CLOCK_DEBUG transport raw_clock rc=%d now_ns=%llu\n", rc,
                (unsigned long long)(rc == 0 ? *out_ns : 0ULL));
    }
    return rc;
}

static int nvshmemi_ibgda_cqe_ts_buf_ensure(void) {
    if (s_cqe_ts_dev_buf && s_gpu_ns_dev_buf && s_cqe_ts_cap > 0) return 0;

    size_t cap = 65536;
    const char *env = getenv("NVSHMEM_IBGDA_CQE_TS_BUF_CAP");
    if (env && *env) {
        char *end = nullptr;
        unsigned long v = strtoul(env, &end, 0);
        if (end != env && v > 0) cap = (size_t)v;
    }

    cudaError_t err;
    err = cudaMalloc((void **)&s_cqe_ts_dev_buf, cap * sizeof(unsigned long long));
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void **)&s_gpu_ns_dev_buf, cap * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        cudaFree(s_cqe_ts_dev_buf);
        s_cqe_ts_dev_buf = nullptr;
        return (int)err;
    }
    err = cudaMemset(s_cqe_ts_dev_buf, 0, cap * sizeof(unsigned long long));
    if (err != cudaSuccess) return (int)err;
    err = cudaMemset(s_gpu_ns_dev_buf, 0, cap * sizeof(unsigned long long));
    if (err != cudaSuccess) return (int)err;

    err = cudaMemcpyToSymbol(nvshmemi_ibgda_cqe_ts_buf, &s_cqe_ts_dev_buf,
                             sizeof(s_cqe_ts_dev_buf), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemcpyToSymbol(nvshmemi_ibgda_gpu_ns_buf, &s_gpu_ns_dev_buf,
                             sizeof(s_gpu_ns_dev_buf), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;

    unsigned long long cap_dev = (unsigned long long)cap;
    err = cudaMemcpyToSymbol(nvshmemi_ibgda_cqe_ts_cap, &cap_dev, sizeof(cap_dev), 0,
                             cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;

    s_cqe_ts_cap = cap;
    return 0;
}

int nvshmemx_ibgda_lat_profile_cqe_ts_reset(void) {
    int rc = nvshmemi_ibgda_cqe_ts_buf_ensure();
    if (rc) return rc;

    const unsigned long long zero = 0;
    cudaError_t err = cudaMemcpyToSymbol(nvshmemi_ibgda_cqe_ts_count, &zero, sizeof(zero), 0,
                                         cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    return 0;
}

int nvshmemx_ibgda_lat_profile_cqe_ts_get(nvshmemx_ibgda_cqe_ts_pair_t *out, size_t max_pairs,
                                          size_t *out_count) {
    if (out_count) *out_count = 0;
    if (!out || max_pairs == 0) return (int)cudaErrorInvalidValue;
    if (!s_cqe_ts_dev_buf || !s_gpu_ns_dev_buf || s_cqe_ts_cap == 0) return 0;

    unsigned long long count_dev = 0;
    cudaError_t err = cudaMemcpyFromSymbol(&count_dev, nvshmemi_ibgda_cqe_ts_count,
                                           sizeof(count_dev), 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;

    size_t n = (size_t)count_dev;
    if (n > s_cqe_ts_cap) n = s_cqe_ts_cap;  /* sampling cap, ignore overflow */
    if (n > max_pairs) n = max_pairs;
    if (n == 0) return 0;

    unsigned long long *tmp_cqe = (unsigned long long *)malloc(n * sizeof(unsigned long long));
    unsigned long long *tmp_gpu = (unsigned long long *)malloc(n * sizeof(unsigned long long));
    if (!tmp_cqe || !tmp_gpu) {
        free(tmp_cqe);
        free(tmp_gpu);
        return (int)cudaErrorMemoryAllocation;
    }

    err = cudaMemcpy(tmp_cqe, s_cqe_ts_dev_buf, n * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(tmp_cqe);
        free(tmp_gpu);
        return (int)err;
    }
    err = cudaMemcpy(tmp_gpu, s_gpu_ns_dev_buf, n * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(tmp_cqe);
        free(tmp_gpu);
        return (int)err;
    }

    for (size_t i = 0; i < n; ++i) {
        out[i].cqe_ts_cycles = tmp_cqe[i];
        out[i].gpu_now_ns    = tmp_gpu[i];
    }
    free(tmp_cqe);
    free(tmp_gpu);

    if (out_count) *out_count = n;
    return 0;
}

__attribute__((visibility("default"), used)) int nvshmemx_ibgda_publish_mlx5_clock_info(
    const void *src, size_t sz) {
    if (!src || sz == 0) return (int)cudaErrorInvalidValue;
    if (sz > sizeof(nvshmemi_ibgda_mlx5_clock_info)) sz = sizeof(nvshmemi_ibgda_mlx5_clock_info);
    cudaError_t err = cudaMemcpyToSymbol(nvshmemi_ibgda_mlx5_clock_info, src, sz, 0,
                                         cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    int valid = 1;
    err = cudaMemcpyToSymbol(nvshmemi_ibgda_mlx5_clock_info_valid, &valid, sizeof(valid), 0,
                             cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return (int)err;
    return 0;
}

__attribute__((visibility("default"), used)) int nvshmemx_ibgda_publish_mlx5_context(
    const void *ctx) {
    s_ibgda_host_context = (struct ibv_context *)ctx;
    if (nvshmemi_ibgda_clock_debug_enabled()) {
        fprintf(stderr, "IBGDA_CLOCK_DEBUG publish host_context=%p\n", s_ibgda_host_context);
    }
    return 0;
}

int nvshmemx_ibgda_get_mlx5_clock_info(void *out) {
    if (!out) return (int)cudaErrorInvalidValue;
    int valid = 0;
    cudaError_t err = cudaMemcpyFromSymbol(&valid, nvshmemi_ibgda_mlx5_clock_info_valid,
                                           sizeof(valid), 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;
    if (!valid) return (int)cudaErrorNotReady;
    err = cudaMemcpyFromSymbol(out, nvshmemi_ibgda_mlx5_clock_info,
                               sizeof(nvshmemi_ibgda_mlx5_clock_info), 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return (int)err;
    return 0;
}

int nvshmemx_ibgda_query_mlx5_clock_info(void *out) {
    if (!out) return (int)cudaErrorInvalidValue;
    if (s_ibgda_host_context != nullptr) {
        int rc = mlx5dv_get_clock_info(s_ibgda_host_context, (struct mlx5dv_clock_info *)out);
        if (nvshmemi_ibgda_clock_debug_enabled()) {
            fprintf(stderr, "IBGDA_CLOCK_DEBUG direct clock_info rc=%d host_context=%p\n", rc,
                    s_ibgda_host_context);
        }
        if (rc == 0) return 0;
    } else if (nvshmemi_ibgda_clock_debug_enabled()) {
        fprintf(stderr, "IBGDA_CLOCK_DEBUG direct clock_info skipped: host_context=null\n");
    }

    return nvshmemi_ibgda_query_mlx5_clock_info_from_transport(out);
}

int nvshmemx_ibgda_query_mlx5_raw_clock_ns(unsigned long long *out_ns) {
    if (!out_ns) return (int)cudaErrorInvalidValue;
    if (s_ibgda_host_context != nullptr) {
        struct ibv_values_ex values;
        memset(&values, 0, sizeof(values));
        values.comp_mask = IBV_VALUES_MASK_RAW_CLOCK;
        int rc = ibv_query_rt_values_ex(s_ibgda_host_context, &values);
        if (nvshmemi_ibgda_clock_debug_enabled()) {
            fprintf(stderr, "IBGDA_CLOCK_DEBUG direct raw_clock rc=%d host_context=%p\n", rc,
                    s_ibgda_host_context);
        }
        if (rc == 0) {
            *out_ns = ((unsigned long long)values.raw_clock.tv_sec * 1000000000ULL) +
                      (unsigned long long)values.raw_clock.tv_nsec;
            return 0;
        }
    } else if (nvshmemi_ibgda_clock_debug_enabled()) {
        fprintf(stderr, "IBGDA_CLOCK_DEBUG direct raw_clock skipped: host_context=null\n");
    }

    return nvshmemi_ibgda_query_mlx5_raw_clock_ns_from_transport(out_ns);
}
#else  /* !NVSHMEM_IBGDA_SUPPORT */
int nvshmemx_ibgda_lat_profile_get(nvshmemx_ibgda_lat_profile_t *out) {
    if (!out) return (int)cudaErrorInvalidValue;
    memset(out, 0, sizeof(*out));
    return 0;
}
int nvshmemx_ibgda_lat_profile_reset(void) { return 0; }
int nvshmemx_ibgda_lat_profile_cqe_ts_reset(void) { return 0; }
int nvshmemx_ibgda_lat_profile_cqe_ts_get(nvshmemx_ibgda_cqe_ts_pair_t *out, size_t max_pairs,
                                          size_t *out_count) {
    (void)out;
    (void)max_pairs;
    if (out_count) *out_count = 0;
    return 0;
}
__attribute__((visibility("default"), used)) int nvshmemx_ibgda_publish_mlx5_clock_info(
    const void *src, size_t sz) {
    (void)src;
    (void)sz;
    return 0;
}
__attribute__((visibility("default"), used)) int nvshmemx_ibgda_publish_mlx5_context(
    const void *ctx) {
    (void)ctx;
    return 0;
}
int nvshmemx_ibgda_get_mlx5_clock_info(void *out) {
    (void)out;
    return (int)cudaErrorNotReady;
}
int nvshmemx_ibgda_query_mlx5_clock_info(void *out) {
    (void)out;
    return (int)cudaErrorNotReady;
}
int nvshmemx_ibgda_query_mlx5_raw_clock_ns(unsigned long long *out_ns) {
    (void)out_ns;
    return (int)cudaErrorNotReady;
}
#endif /* NVSHMEM_IBGDA_SUPPORT */

void nvshmemi_finalize() {
    int status;
    void *dev_state_ptr, *transport_dev_state_ptr = NULL;

    status = cudaGetSymbolAddress(&dev_state_ptr, nvshmemi_device_state_d);
    if (status) {
        NVSHMEMI_ERROR_PRINT("Unable to properly unregister device state.\n");
        nvshmemid_hostlib_finalize(NULL, NULL);
        return;
    }
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmem_selected_device_transport == NVSHMEMI_DEVICE_TRANSPORT_TYPE_IBGDA) {
        status = cudaGetSymbolAddress(&transport_dev_state_ptr, nvshmemi_ibgda_device_state_d);
        if (status) {
            NVSHMEMI_ERROR_PRINT("Unable to properly unregister device state.\n");
            nvshmemid_hostlib_finalize(NULL, NULL);
            return;
        }
    }
#endif
    nvshmemid_hostlib_finalize(dev_state_ptr, transport_dev_state_ptr);
}
#ifdef __cplusplus
}
#endif
#endif
