/* P1T-FIX4 test shim (LD_PRELOAD) over libcudart.so.12 and libcublas.so.12 for tc_signal's device / CUDA-error sweep
 * tests (test_fix4_findings.py). This host has ONE GPU, so "another device is current" is simulated, as FIX1 did for
 * core's F05: in fake mode the shim reports a simulated current device, logs each interposed call with the simulated
 * device it would run on, and keeps the REAL device at 0. It can fail one named call, or the Nth cudaSetDevice(d), with
 * a GENUINE runtime error (it calls the real cudaSetDevice(-1), so the thread's last-error slot really holds
 * cudaErrorInvalidDevice), and it can fail cublasSetMathMode with CUBLAS_STATUS_INTERNAL_ERROR (14).
 *
 * Use it only in processes that do not import torch: its definitions are global, so torch's own runtime calls
 * (torch 2.11 + cu130 uses libcudart.so.13) would also be routed here, to libcudart.so.12.
 *
 * Prior art: LD_PRELOAD symbol interposition (glibc ld.so; classic malloc/syscall interposers) and the FIX1 F05 shim in
 * tensor_cuda/tests/test_dlpack.py. Ours: the call log, genuine-slot failure injection and the cuBLAS hooks.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef struct { unsigned x, y, z; } dim3_;

static int fake = -1;          /* -1: pass-through; else the simulated current device */
static int fail_set_to = -1, fail_set_nth = 0, set_seen = 0;
static char fail_name[64];
static char logbuf[16384];

static void note(const char* s) {
  size_t n = strlen(logbuf);
  if (n + strlen(s) + 2 < sizeof logbuf) { strcat(logbuf, s); strcat(logbuf, " "); }
}
static void* real_sym(const char* lib, const char* name) {
  void* h = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD);
  if (!h) h = dlopen(lib, RTLD_LAZY);
  return h ? dlsym(h, name) : NULL;
}
#define REAL(R, n, A) static R (*real_##n) A; if (!real_##n) real_##n = (R (*) A)real_sym("libcudart.so.12", #n)
#define REALB(R, n, A) static R (*real_##n) A; if (!real_##n) real_##n = (R (*) A)real_sym("libcublas.so.12", #n)

static int genuine_failure(void) { REAL(int, cudaSetDevice, (int)); return real_cudaSetDevice(-1); }
static int take_fail(const char* n) {
  if (fail_name[0] && strcmp(fail_name, n) == 0) { fail_name[0] = 0; return 1; }
  return 0;
}
static void lognote(const char* what, int failed) {
  if (fake < 0 && !failed) return;
  char b[48];
  snprintf(b, sizeof b, "%s@%d%s", what, fake, failed ? "!" : "");
  note(b);
}

int cudaGetDevice(int* d) {
  REAL(int, cudaGetDevice, (int*));
  if (fake >= 0) { *d = fake; return 0; }
  return real_cudaGetDevice(d);
}
int cudaSetDevice(int d) {
  REAL(int, cudaSetDevice, (int));
  if (fake < 0) return real_cudaSetDevice(d);
  int fail = 0;
  if (d == fail_set_to && ++set_seen == fail_set_nth) { fail = 1; fail_set_to = -1; }
  char b[32];
  snprintf(b, sizeof b, "set%d%s", d, fail ? "!" : "");
  note(b);
  if (fail) return genuine_failure();
  fake = d;
  return real_cudaSetDevice(0);
}
int cudaEventCreateWithFlags(void** e, unsigned f) {
  REAL(int, cudaEventCreateWithFlags, (void**, unsigned));
  if (take_fail("cudaEventCreateWithFlags")) { lognote("create", 1); return genuine_failure(); }
  lognote("create", 0);
  return real_cudaEventCreateWithFlags(e, f);
}
int cudaEventRecord(void* e, void* s) {
  REAL(int, cudaEventRecord, (void*, void*));
  if (take_fail("cudaEventRecord")) { lognote("record", 1); return genuine_failure(); }
  lognote("record", 0);
  return real_cudaEventRecord(e, s);
}
int cudaEventQuery(void* e) {
  REAL(int, cudaEventQuery, (void*));
  if (take_fail("cudaEventQuery")) { lognote("query", 1); return genuine_failure(); }
  return real_cudaEventQuery(e);   /* queries are not logged: their number depends on timing */
}
int cudaEventDestroy(void* e) {
  REAL(int, cudaEventDestroy, (void*));
  if (take_fail("cudaEventDestroy")) { lognote("destroy", 1); real_cudaEventDestroy(e); return genuine_failure(); }
  lognote("destroy", 0);
  return real_cudaEventDestroy(e);
}
int cudaStreamSynchronize(void* s) {
  REAL(int, cudaStreamSynchronize, (void*));
  lognote("sync", 0);
  return real_cudaStreamSynchronize(s);
}
int cudaLaunchKernel(const void* fn, dim3_ g, dim3_ b, void** a, size_t sh, void* s) {
  REAL(int, cudaLaunchKernel, (const void*, dim3_, dim3_, void**, size_t, void*));
  if (take_fail("cudaLaunchKernel")) { lognote("launch", 1); return genuine_failure(); }   /* nothing launched */
  lognote("launch", 0);
  return real_cudaLaunchKernel(fn, g, b, a, sh, s);
}
int cudaMemsetAsync(void* p, int v, size_t n, void* s) {
  REAL(int, cudaMemsetAsync, (void*, int, size_t, void*));
  if (take_fail("cudaMemsetAsync")) { lognote("memset", 1); return genuine_failure(); }
  return real_cudaMemsetAsync(p, v, n, s);
}
int cublasSetMathMode(void* h, int m) {
  REALB(int, cublasSetMathMode, (void*, int));
  if (take_fail("cublasSetMathMode")) { note("cublasSetMathMode!"); return 14; /* CUBLAS_STATUS_INTERNAL_ERROR */ }
  return real_cublasSetMathMode(h, m);
}
int cublasDestroy_v2(void* h) {
  REALB(int, cublasDestroy_v2, (void*));
  note("cublasDestroy");
  return real_cublasDestroy_v2(h);
}

void shim_fake(int d) { fake = d; }
void shim_fail_set(int d, int nth) { fail_set_to = d; fail_set_nth = nth; set_seen = 0; }
void shim_fail(const char* n) { strncpy(fail_name, n, sizeof fail_name - 1); }
int shim_current(void) { return fake; }
int shim_slot_peek(void) { REAL(int, cudaPeekAtLastError, (void)); return real_cudaPeekAtLastError(); }
const char* shim_log(void) { return logbuf; }
void shim_clear(void) { logbuf[0] = 0; }
