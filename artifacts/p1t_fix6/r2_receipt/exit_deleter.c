// R2 receipt (scratch): a DLPack deleter that terminates its calling thread instead of returning.
#include <pthread.h>
#include <stdio.h>
void exit_deleter(void* m) { (void)m; fprintf(stderr, "DELETER calls pthread_exit\n"); fflush(stderr); pthread_exit(NULL); }
