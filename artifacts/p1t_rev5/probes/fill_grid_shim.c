/* Prior art: glibc LD_PRELOAD, NVIDIA CUDA12.6 launch geometry. Ours: capture fill geometry without touching data. */
#include <dlfcn.h>
struct dim3 { unsigned x,y,z; };
static int armed=0,calls=0;static unsigned gridx=0;
void rev5_arm_fill(void){armed=1;}
unsigned rev5_fill_grid(void){return gridx;}
int rev5_fill_calls(void){return calls;}
int cudaLaunchKernel(const void* f,struct dim3 g,struct dim3 b,void** a,unsigned long s,void* stream){
 int (*real_launch)(const void*,struct dim3,struct dim3,void**,unsigned long,void*)=dlsym(RTLD_NEXT,"cudaLaunchKernel");
 if(armed){armed=0;++calls;gridx=g.x;return real_launch(0,g,b,a,s,stream);}
 return real_launch(f,g,b,a,s,stream);
}
