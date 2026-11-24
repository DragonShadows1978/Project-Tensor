/* ENG5-S copy (orders/ENG5S_STATUS_CHECKS.md, G1 and G2) of ENG5's injection shim (eng5 branch,
 * tensor_cuda/tests/support/eng5_cuda_shim.c, read-only there). ENG5-S changes, all in the injection build only:
 *   - cudaMalloc and cudaMallocAsync are wrapped and injectable (the allocator's OOM retry is reached by making the
 *     allocation return cudaErrorMemoryAllocation; ENG3's counter wraps them in the counting build, so they sit in
 *     the #ifndef ENG5_WITH_ENG3_COUNTER block with the other overlapping wrappers);
 *   - up to four injections can be armed at once (the OOM-drain row arms the allocation AND the drain);
 *     eng5_inject_next() adds one armed slot, eng5s_inject_clear() disarms all, eng5_injected() returns how many armed
 *     slots fired, eng5s_fired(name) whether the slot armed for that name fired.
 * The counting interface (eng5_extra_calls, eng5_count, eng5_name, eng5_n) is unchanged.
 *
 * ENG5 instrument (orders/ENG5_HOST_GUARD_AND_STATUS.md, G1 and G3): an LD_PRELOAD shim with exact prototypes.
 *
 * 1. Counting (G1, CPU): it counts the CUDA entry points the engine module imports (nm -D --undefined-only of the
 *    module) that ENG3's counter (cudart_count.c, 32 runtime functions) does not wrap: five more runtime functions
 *    and every cuBLAS and cuBLASLt function. With ENG3's counter preloaded too, the two cover every CUDA runtime,
 *    cuBLAS and cuBLASLt call the module makes through its dynamic imports (the load-time fat-binary registration
 *    calls excepted, as in ENG3).
 * 2. Failure injection (G3, GPU lines): ENG5_INJECT="<function>:<nth>:<code>" makes the nth call (1-based, counted
 *    from process start) of <function> return <code> WITHOUT calling the real function, so the runtime records no
 *    error and cudaGetLastError cannot see it; every other call is forwarded. eng5_injected() reports whether the
 *    injection fired. The memcpy/memset wrappers here overlap ENG3's counter; preload only one of the two for those.
 *
 * Prior art: LD_PRELOAD interposition with dlsym(RTLD_NEXT) (glibc ld.so; ltrace-style call tracing), as in ENG3's
 * cudart_count.c (this repo), whose libcudart resolution fallback (dlopen RTLD_NOLOAD) is taken as is; fault injection
 * by library interposition (e.g. LFI, Marinescu and Candea, 2009 -- unverified, lead to check: "library fault
 * injection LFI interposition"). Ours: the symbol list, exact-prototype forwarding (cuBLAS GEMMs take more than eight
 * arguments, so ENG3's eight-register forwarding cannot be used for them), and the nth-call injection rule. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>

static void* real_sym(const char* n) {
  void* p = dlsym(RTLD_NEXT, n);
  if (p) return p;
  static const char* libs[3] = {"libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"};
  for (int i = 0; i < 3; ++i) {
    void* h = dlopen(libs[i], RTLD_LAZY | RTLD_NOLOAD);
    if (h && (p = dlsym(h, n))) return p;
  }
  return 0;
}

enum { NSYM = 40 };
static const char* g_names[NSYM] = {
  "cudaDeviceGetAttribute", "cudaDriverGetVersion", "cudaFuncGetAttributes", "cudaGetDeviceCount",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
  "cublasCreate_v2", "cublasDestroy_v2", "cublasGemmEx", "cublasGemmStridedBatchedEx", "cublasGetAtomicsMode",
  "cublasGetMathMode", "cublasGetPointerMode_v2", "cublasGetStatusName", "cublasGetStatusString", "cublasGetStream_v2",
  "cublasGetVersion_v2", "cublasSetAtomicsMode", "cublasSetMathMode", "cublasSetStream_v2", "cublasSetWorkspace_v2",
  "cublasSgemmStridedBatched",
  "cublasLtCreate", "cublasLtMatmul", "cublasLtMatmulAlgoGetHeuristic", "cublasLtMatmulDescCreate",
  "cublasLtMatmulDescDestroy", "cublasLtMatmulDescSetAttribute", "cublasLtMatmulPreferenceCreate",
  "cublasLtMatmulPreferenceDestroy", "cublasLtMatmulPreferenceSetAttribute", "cublasLtMatrixLayoutCreate",
  "cublasLtMatrixLayoutDestroy", "cublasLtMatrixLayoutSetAttribute",
  "cudaMemcpy", "cudaMemset", "cudaMemsetAsync", "cudaDeviceSynchronize", "cudaStreamSynchronize",
  "cudaMalloc", "cudaMallocAsync"};
/* The last five overlap ENG3's counter: they are counted here but excluded from eng5_extra_calls(). ENG5-F1 added
 * cudaDeviceSynchronize and cudaStreamSynchronize (P4's synchronize statuses), wrapped only in the injection build. */
enum { NEXTRA = 33 };
static volatile unsigned long g_counts[NSYM];
static volatile unsigned long g_extra_total = 0;
static int g_inject_idx = -2;  /* -2: not parsed yet; -1: no injection (environment form) */
static unsigned long g_inject_nth = 0;
static long g_inject_code = 0;
static volatile int g_injected = 0;
/* ENG5-S: armed slots (in-process form). */
enum { NSLOT = 4 };
static int g_slot_idx[NSLOT] = {-1, -1, -1, -1};
static unsigned long g_slot_nth[NSLOT];
static long g_slot_code[NSLOT];
static volatile int g_slot_fired[NSLOT];

unsigned long eng5_extra_calls(void) { return g_extra_total; }
unsigned long eng5_count(int i) { return (i >= 0 && i < NSYM) ? g_counts[i] : 0; }
const char* eng5_name(int i) { return (i >= 0 && i < NSYM) ? g_names[i] : 0; }
int eng5_n(void) { return NSYM; }
int eng5_injected(void) {
  int n = g_injected;
  for (int s = 0; s < NSLOT; ++s) n += (g_slot_idx[s] >= 0 && g_slot_fired[s]) ? 1 : 0;
  return n;
}
void eng5s_inject_clear(void) {
  for (int s = 0; s < NSLOT; ++s) { g_slot_idx[s] = -1; g_slot_fired[s] = 0; }
  g_injected = 0;
}
int eng5s_fired(const char* name) {
  for (int s = 0; s < NSLOT; ++s)
    if (g_slot_idx[s] >= 0 && strcmp(name, g_names[g_slot_idx[s]]) == 0) return g_slot_fired[s];
  return -1;
}
/* Arm an injection from the process itself: the NEXT call of `name` (counted from now) returns `code`. Returns 0
 * when armed, -1 for a name this shim does not wrap, -2 when every slot is in use. Used by the G2 lines right before
 * the call under test (ENG5-S: one slot per call; several may be armed together). */
int eng5_inject_next(const char* name, long code) {
  for (int i = 0; i < NSYM; ++i)
    if (strcmp(name, g_names[i]) == 0) {
      for (int s = 0; s < NSLOT; ++s)
        if (g_slot_idx[s] < 0) {
          g_slot_nth[s] = g_counts[i] + 1;
          g_slot_code[s] = code;
          g_slot_fired[s] = 0;
          g_slot_idx[s] = i;
          return 0;
        }
      return -2;
    }
  return -1;
}

static void parse_inject(void) {
  g_inject_idx = -1;
  const char* e = getenv("ENG5_INJECT");
  if (!e || !*e) return;
  char buf[160];
  strncpy(buf, e, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = 0;
  char* c1 = strchr(buf, ':');
  if (!c1) return;
  *c1 = 0;
  char* c2 = strchr(c1 + 1, ':');
  if (!c2) return;
  *c2 = 0;
  for (int i = 0; i < NSYM; ++i)
    if (strcmp(buf, g_names[i]) == 0) g_inject_idx = i;
  g_inject_nth = strtoul(c1 + 1, 0, 10);
  g_inject_code = strtol(c2 + 1, 0, 10);
}

/* Returns 1 when this call is the one to fail (its code is then in *code). */
static int hit_code(int idx, long* code) {
  unsigned long c = __atomic_add_fetch(&g_counts[idx], 1, __ATOMIC_SEQ_CST);
  if (idx < NEXTRA) __atomic_add_fetch(&g_extra_total, 1, __ATOMIC_SEQ_CST);
  if (g_inject_idx == -2) parse_inject();
  if (g_inject_idx == idx && c == g_inject_nth) {
    g_injected = 1;
    *code = g_inject_code;
    return 1;
  }
  for (int s = 0; s < NSLOT; ++s)
    if (g_slot_idx[s] == idx && c == g_slot_nth[s] && !g_slot_fired[s]) {
      g_slot_fired[s] = 1;
      *code = g_slot_code[s];
      return 1;
    }
  return 0;
}
static int hit(int idx) {
  long unused;
  return hit_code(idx, &unused);
}

#define FWD(idx, ret, name, params, args)                                  \
  ret name params {                                                         \
    static ret(*real) params = 0;                                           \
    long code_;                                                             \
    if (hit_code(idx, &code_)) return (ret)code_;                          \
    if (!real) real = (ret(*) params)real_sym(#name);                       \
    return real args;                                                       \
  }
#define FWD_NOINJ(idx, ret, name, params, args)                            \
  ret name params {                                                         \
    static ret(*real) params = 0;                                           \
    (void)hit(idx);                                                         \
    if (!real) real = (ret(*) params)real_sym(#name);                       \
    return real args;                                                       \
  }

FWD(0, cudaError_t, cudaDeviceGetAttribute, (int* v, enum cudaDeviceAttr a, int d), (v, a, d))
FWD(1, cudaError_t, cudaDriverGetVersion, (int* v), (v))
FWD(2, cudaError_t, cudaFuncGetAttributes, (struct cudaFuncAttributes* a, const void* f), (a, f))
FWD(3, cudaError_t, cudaGetDeviceCount, (int* c), (c))
FWD(4, cudaError_t, cudaOccupancyMaxActiveBlocksPerMultiprocessor, (int* n, const void* f, int b, size_t s),
    (n, f, b, s))
FWD(5, cublasStatus_t, cublasCreate_v2, (cublasHandle_t* h), (h))
FWD(6, cublasStatus_t, cublasDestroy_v2, (cublasHandle_t h), (h))
FWD(7, cublasStatus_t, cublasGemmEx,
    (cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const void* alpha,
     const void* A, cudaDataType At, int lda, const void* B, cudaDataType Bt, int ldb, const void* beta, void* C,
     cudaDataType Ct, int ldc, cublasComputeType_t ct, cublasGemmAlgo_t algo),
    (h, ta, tb, m, n, k, alpha, A, At, lda, B, Bt, ldb, beta, C, Ct, ldc, ct, algo))
FWD(8, cublasStatus_t, cublasGemmStridedBatchedEx,
    (cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const void* alpha,
     const void* A, cudaDataType At, int lda, long long int sA, const void* B, cudaDataType Bt, int ldb,
     long long int sB, const void* beta, void* C, cudaDataType Ct, int ldc, long long int sC, int bc,
     cublasComputeType_t ct, cublasGemmAlgo_t algo),
    (h, ta, tb, m, n, k, alpha, A, At, lda, sA, B, Bt, ldb, sB, beta, C, Ct, ldc, sC, bc, ct, algo))
FWD(9, cublasStatus_t, cublasGetAtomicsMode, (cublasHandle_t h, cublasAtomicsMode_t* m), (h, m))
FWD(10, cublasStatus_t, cublasGetMathMode, (cublasHandle_t h, cublasMath_t* m), (h, m))
FWD(11, cublasStatus_t, cublasGetPointerMode_v2, (cublasHandle_t h, cublasPointerMode_t* m), (h, m))
FWD_NOINJ(12, const char*, cublasGetStatusName, (cublasStatus_t s), (s))
FWD_NOINJ(13, const char*, cublasGetStatusString, (cublasStatus_t s), (s))
FWD(14, cublasStatus_t, cublasGetStream_v2, (cublasHandle_t h, cudaStream_t* s), (h, s))
FWD(15, cublasStatus_t, cublasGetVersion_v2, (cublasHandle_t h, int* v), (h, v))
FWD(16, cublasStatus_t, cublasSetAtomicsMode, (cublasHandle_t h, cublasAtomicsMode_t m), (h, m))
FWD(17, cublasStatus_t, cublasSetMathMode, (cublasHandle_t h, cublasMath_t m), (h, m))
FWD(18, cublasStatus_t, cublasSetStream_v2, (cublasHandle_t h, cudaStream_t s), (h, s))
FWD(19, cublasStatus_t, cublasSetWorkspace_v2, (cublasHandle_t h, void* w, size_t n), (h, w, n))
FWD(20, cublasStatus_t, cublasSgemmStridedBatched,
    (cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const float* alpha,
     const float* A, int lda, long long int sA, const float* B, int ldb, long long int sB, const float* beta,
     float* C, int ldc, long long int sC, int bc),
    (h, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc))
FWD(21, cublasStatus_t, cublasLtCreate, (cublasLtHandle_t* h), (h))
FWD(22, cublasStatus_t, cublasLtMatmul,
    (cublasLtHandle_t h, cublasLtMatmulDesc_t d, const void* alpha, const void* A, cublasLtMatrixLayout_t Ad,
     const void* B, cublasLtMatrixLayout_t Bd, const void* beta, const void* C, cublasLtMatrixLayout_t Cd, void* D,
     cublasLtMatrixLayout_t Dd, const cublasLtMatmulAlgo_t* algo, void* ws, size_t wsn, cudaStream_t st),
    (h, d, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd, algo, ws, wsn, st))
FWD(23, cublasStatus_t, cublasLtMatmulAlgoGetHeuristic,
    (cublasLtHandle_t h, cublasLtMatmulDesc_t d, cublasLtMatrixLayout_t Ad, cublasLtMatrixLayout_t Bd,
     cublasLtMatrixLayout_t Cd, cublasLtMatrixLayout_t Dd, cublasLtMatmulPreference_t p, int req,
     cublasLtMatmulHeuristicResult_t r[], int* ret),
    (h, d, Ad, Bd, Cd, Dd, p, req, r, ret))
FWD(24, cublasStatus_t, cublasLtMatmulDescCreate, (cublasLtMatmulDesc_t* d, cublasComputeType_t c, cudaDataType_t s),
    (d, c, s))
FWD(25, cublasStatus_t, cublasLtMatmulDescDestroy, (cublasLtMatmulDesc_t d), (d))
FWD(26, cublasStatus_t, cublasLtMatmulDescSetAttribute,
    (cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t a, const void* b, size_t n), (d, a, b, n))
FWD(27, cublasStatus_t, cublasLtMatmulPreferenceCreate, (cublasLtMatmulPreference_t* p), (p))
FWD(28, cublasStatus_t, cublasLtMatmulPreferenceDestroy, (cublasLtMatmulPreference_t p), (p))
FWD(29, cublasStatus_t, cublasLtMatmulPreferenceSetAttribute,
    (cublasLtMatmulPreference_t p, cublasLtMatmulPreferenceAttributes_t a, const void* b, size_t n), (p, a, b, n))
FWD(30, cublasStatus_t, cublasLtMatrixLayoutCreate,
    (cublasLtMatrixLayout_t* l, cudaDataType t, uint64_t r, uint64_t c, int64_t ld), (l, t, r, c, ld))
FWD(31, cublasStatus_t, cublasLtMatrixLayoutDestroy, (cublasLtMatrixLayout_t l), (l))
FWD(32, cublasStatus_t, cublasLtMatrixLayoutSetAttribute,
    (cublasLtMatrixLayout_t l, cublasLtMatrixLayoutAttribute_t a, const void* b, size_t n), (l, a, b, n))
/* Built with -DENG5_WITH_ENG3_COUNTER (G1: preloaded beside ENG3's counter), the three overlapping wrappers are
 * left out so that ENG3's counter alone wraps them. */
#ifndef ENG5_WITH_ENG3_COUNTER
FWD(33, cudaError_t, cudaMemcpy, (void* d, const void* s, size_t n, enum cudaMemcpyKind k), (d, s, n, k))
FWD(34, cudaError_t, cudaMemset, (void* p, int v, size_t n), (p, v, n))
FWD(35, cudaError_t, cudaMemsetAsync, (void* p, int v, size_t n, cudaStream_t s), (p, v, n, s))
FWD(36, cudaError_t, cudaDeviceSynchronize, (void), ())
FWD(37, cudaError_t, cudaStreamSynchronize, (cudaStream_t s), (s))
/* ENG5-S: the allocator's two allocation calls (Storage::Storage), injectable to reach the OOM retry. */
FWD(38, cudaError_t, cudaMalloc, (void** p, size_t n), (p, n))
FWD(39, cudaError_t, cudaMallocAsync, (void** p, size_t n, cudaStream_t s), (p, n, s))
#endif
