// Prior art: FIX8 L2 pthread fault interposer (2026); unchanged fault mechanism.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int (*create_fn)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
typedef int (*mutex_fn)(pthread_mutex_t*);
typedef int (*clockwait_fn)(pthread_cond_t*, pthread_mutex_t*, clockid_t, const struct timespec*);
typedef int (*cond_fn)(pthread_cond_t*);
typedef int (*join_fn)(pthread_t, void**);
typedef int (*detach_fn)(pthread_t);
static create_fn r_create; static mutex_fn r_lock, r_unlock; static clockwait_fn r_clockwait;
static cond_fn r_broadcast; static join_fn r_join; static detach_fn r_detach;
__attribute__((constructor)) static void shim_init(void) {
  r_create = (create_fn)dlsym(RTLD_NEXT, "pthread_create");
  r_lock = (mutex_fn)dlsym(RTLD_NEXT, "pthread_mutex_lock");
  r_unlock = (mutex_fn)dlsym(RTLD_NEXT, "pthread_mutex_unlock");
  r_clockwait = (clockwait_fn)dlsym(RTLD_NEXT, "pthread_cond_clockwait");
  r_broadcast = (cond_fn)dlsym(RTLD_NEXT, "pthread_cond_broadcast");
  r_join = (join_fn)dlsym(RTLD_NEXT, "pthread_join");
  r_detach = (detach_fn)dlsym(RTLD_NEXT, "pthread_detach");
}

static int armed, window, wd_created, wd_known, stop_began, drop_next, detached;
static pthread_t main_thread, wd;
static void* (*wd_fn)(void*);
static int lock_fail_left, drop_notify, join_fail_left, hold_ms, held;
static long long locks_failed, notifies_dropped, joins_failed, returned_at_detach, wd_ops, wd_ops_after_detach,
                 wd_returned, t_stop, t_join_ret, t_detach, t_last_wd_op;

static long long now_ns(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static int from_addon(void* ra) {
  Dl_info info;
  if (!dladdr(ra, &info) || !info.dli_fname) return 0;
  const char* b = strrchr(info.dli_fname, '/'); b = b ? b + 1 : info.dli_fname;
  return strncmp(b, "_tc_signal", 10) == 0;
}
static int on_wd(void) { return __atomic_load_n(&wd_known, __ATOMIC_ACQUIRE) && pthread_equal(pthread_self(), wd); }
static void wd_op(void) {
  __atomic_add_fetch(&wd_ops, 1, __ATOMIC_ACQ_REL);
  __atomic_store_n(&t_last_wd_op, now_ns(), __ATOMIC_RELEASE);
  if (__atomic_load_n(&detached, __ATOMIC_ACQUIRE)) __atomic_add_fetch(&wd_ops_after_detach, 1, __ATOMIC_ACQ_REL);
}
static void* tramp(void* a) {
  wd = pthread_self(); __atomic_store_n(&wd_known, 1, __ATOMIC_RELEASE);
  void* r = wd_fn(a);
  __atomic_store_n(&wd_returned, 1, __ATOMIC_RELEASE);
  return r;
}

int pthread_create(pthread_t* t, const pthread_attr_t* at, void* (*fn)(void*), void* arg) {
  if (__atomic_load_n(&armed, __ATOMIC_ACQUIRE) && !wd_created && pthread_equal(pthread_self(), main_thread)) {
    wd_created = 1; wd_fn = fn;
    int e = r_create(t, at, tramp, arg);
    if (e == 0) { wd = *t; __atomic_store_n(&wd_known, 1, __ATOMIC_RELEASE); __atomic_store_n(&window, 1, __ATOMIC_RELEASE); }
    return e;
  }
  return r_create(t, at, fn, arg);
}
int pthread_mutex_lock(pthread_mutex_t* m) {
  if (__atomic_load_n(&window, __ATOMIC_ACQUIRE) && pthread_equal(pthread_self(), main_thread)
      && from_addon(__builtin_return_address(0))) {
    if (!stop_began) { t_stop = now_ns(); __atomic_store_n(&stop_began, 1, __ATOMIC_RELEASE); }
    if (lock_fail_left > 0) { --lock_fail_left; ++locks_failed; drop_next = drop_notify; return EINVAL; }
  } else if (on_wd() && from_addon(__builtin_return_address(0))) {
    wd_op();
  }
  return r_lock(m);
}
int pthread_mutex_unlock(pthread_mutex_t* m) {
  if (on_wd() && from_addon(__builtin_return_address(0))) wd_op();
  return r_unlock(m);
}
int pthread_cond_clockwait(pthread_cond_t* c, pthread_mutex_t* m, clockid_t ck, const struct timespec* ts) {
  const int w = on_wd() && from_addon(__builtin_return_address(0));
  if (w) wd_op();
  const int e = r_clockwait(c, m, ck, ts);
  if (w && hold_ms > 0 && !held && __atomic_load_n(&stop_began, __ATOMIC_ACQUIRE)) { held = 1; usleep(hold_ms * 1000); }
  if (w) wd_op();
  return e;
}
int pthread_cond_broadcast(pthread_cond_t* c) {
  if (drop_next && pthread_equal(pthread_self(), main_thread)) { drop_next = 0; ++notifies_dropped; return 0; }
  return r_broadcast(c);
}
int pthread_join(pthread_t t, void** ret) {
  const int w = __atomic_load_n(&wd_known, __ATOMIC_ACQUIRE) && pthread_equal(t, wd);
  if (w) {
    __atomic_store_n(&window, 0, __ATOMIC_RELEASE);
    if (join_fail_left > 0) { --join_fail_left; ++joins_failed; return EINVAL; }
  }
  const int e = r_join(t, ret);
  if (w) t_join_ret = now_ns();
  return e;
}
int pthread_detach(pthread_t t) {
  if (__atomic_load_n(&wd_known, __ATOMIC_ACQUIRE) && pthread_equal(t, wd)) {
    __atomic_store_n(&window, 0, __ATOMIC_RELEASE);
    returned_at_detach = __atomic_load_n(&wd_returned, __ATOMIC_ACQUIRE);
    t_detach = now_ns();
    __atomic_store_n(&detached, 1, __ATOMIC_RELEASE);
  }
  return r_detach(t);
}
void shim_arm(int lock_fail, int drop, int join_fail, int hold) {
  main_thread = pthread_self(); lock_fail_left = lock_fail; drop_notify = drop; join_fail_left = join_fail;
  hold_ms = hold; __atomic_store_n(&armed, 1, __ATOMIC_RELEASE);
}
void shim_state(long long* o) {
  usleep(200000);   /* let a detached watchdog finish before reading (its after-detach calls are counted) */
  o[0] = wd_created; o[1] = locks_failed; o[2] = notifies_dropped; o[3] = joins_failed;
  o[4] = __atomic_load_n(&detached, __ATOMIC_ACQUIRE); o[5] = returned_at_detach;
  o[6] = __atomic_load_n(&wd_ops, __ATOMIC_ACQUIRE); o[7] = __atomic_load_n(&wd_ops_after_detach, __ATOMIC_ACQUIRE);
  o[8] = t_stop; o[9] = t_join_ret; o[10] = t_detach; o[11] = __atomic_load_n(&t_last_wd_op, __ATOMIC_ACQUIRE);
  o[12] = __atomic_load_n(&wd_returned, __ATOMIC_ACQUIRE); o[13] = held; o[14] = lock_fail_left; o[15] = join_fail_left;
}
