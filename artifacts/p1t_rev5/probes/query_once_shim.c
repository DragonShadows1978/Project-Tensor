/* Prior art: glibc LD_PRELOAD interposition; NVIDIA CUDA12.6 checked query returns.
 * Taken: interposition; ours: one genuine invalid-event query failure on an otherwise valid pending event. */
#include <dlfcn.h>
static int armed=0, failures=0;
void rev5_arm_query_once(void){armed=1;}
int rev5_query_failures(void){return failures;}
int cudaEventQuery(void* event){
 int (*real_query)(void*)=dlsym(RTLD_NEXT,"cudaEventQuery");
 if(armed){armed=0;++failures;return real_query(0);}
 return real_query(event);
}
