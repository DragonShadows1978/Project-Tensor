/* Prior art: C++17 mutex serialization and pthreads (POSIX2008), SQLite fault scheduling.
 * Taken: schedule another native writer after a raw snapshot unlock; ours: FIX8 snapshot boundary; also works when a fixed build uses one raw snapshot.
 * No product code is copied: the callback enters the reviewed record writer, from a native thread. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <string.h>
static int (*real_unlock)(pthread_mutex_t*);
static void (*record_fn)(void);
static int armed, fired;
static pthread_t main_thread;
__attribute__((constructor)) static void init(void){real_unlock=dlsym(RTLD_NEXT,"pthread_mutex_unlock");}
static void* writer(void* p){record_fn();return 0;}
void arm(void* p){record_fn=p;main_thread=pthread_self();armed=1;}
int was_fired(void){return fired;}
int pthread_mutex_unlock(pthread_mutex_t* m){
 int result=real_unlock(m);
 Dl_info info;
 if(armed && pthread_equal(pthread_self(),main_thread) && dladdr(__builtin_return_address(0),&info)
    && info.dli_sname && strstr(info.dli_sname,"dlpack_unreported_unraisable") && strstr(info.dli_sname,"_raw")){
  armed=0;pthread_t th;
  if(pthread_create(&th,0,writer,0)==0){pthread_join(th,0);fired=1;}
 }
 return result;
}
