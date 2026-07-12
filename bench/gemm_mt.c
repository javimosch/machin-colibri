#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#define M 8192
#define K 2048
#define B 256
#define GS 64
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int8_t *W; static float *WS; static int8_t *X; static float *XS; static float *OUT;
static int NT;
static inline int32_t gdot(const int8_t*a,const int8_t*b){ int32_t s=0; for(int k=0;k<GS;k++) s+=(int32_t)a[k]*(int32_t)b[k]; return s; }
static void* work(void* arg){
    int tid=(int)(intptr_t)arg;
    int lo=(int64_t)M*tid/NT, hi=(int64_t)M*(tid+1)/NT;
    for(int i=lo;i<hi;i++){
        const int8_t* wr=W+(int64_t)i*K; const float* wsc=WS+(int64_t)i*(K/GS);
        for(int b=0;b<B;b++){
            const int8_t* xr=X+(int64_t)b*K; const float* xsc=XS+(int64_t)b*(K/GS);
            float val=0; for(int g=0;g<K/GS;g++) val+=(float)gdot(xr+g*GS,wr+g*GS)*xsc[g]*wsc[g];
            OUT[(int64_t)b*M+i]=val;
        }
    }
    return 0;
}
static double run(int nt){ NT=nt; pthread_t th[64]; double t=now_s();
    for(int i=0;i<nt;i++) pthread_create(&th[i],0,work,(void*)(intptr_t)i);
    for(int i=0;i<nt;i++) pthread_join(th[i],0); return now_s()-t; }
int main(void){
    W=malloc((int64_t)M*K); WS=malloc((int64_t)M*(K/GS)*4); X=malloc((int64_t)B*K); XS=malloc((int64_t)B*(K/GS)*4); OUT=malloc((int64_t)B*M*4);
    for(int64_t i=0;i<(int64_t)M*K;i++) W[i]=(int8_t)(i*7);
    for(int64_t i=0;i<(int64_t)B*K;i++) X[i]=(int8_t)(i*3);
    for(int64_t i=0;i<(int64_t)M*(K/GS);i++) WS[i]=0.01f;
    for(int64_t i=0;i<(int64_t)B*(K/GS);i++) XS[i]=0.02f;
    double flop=2.0*M*K*B;
    run(1); // warm
    for(int nt=1;nt<=12;nt+=(nt<4?1:4)){ double dt=run(nt); printf("%2d threads: %6.1f GF/s\n", nt, flop/1e9/dt); }
    return 0;
}
