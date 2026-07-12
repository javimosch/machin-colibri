// Frontier test: is colibri's prefill FFN bottleneck the un-blocked GEMM
// activation re-streaming? Compare, on the w1 shape (M output rows x K input,
// B positions), grouped-int8 GEMM done three ways, single-thread:
//   A. un-blocked: for each out row, for each position, dot (current colibri).
//   B. position-blocked micro-kernel: register-tile a small block of positions
//      against each weight row so the activation block stays in registers/L1.
// Reports GFLOP/s. If B >> A, a blocked GEMM micro-kernel is the lever.
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define M 8192      // output rows (hidden)
#define K 2048      // input dim
#define B 256       // positions (tile)
#define GS 64

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static int8_t *W;        // M x K int8 weights
static float   *WS;      // M x (K/GS) weight scales
static int8_t *X;        // B x K int8 activations
static float   *XS;      // B x (K/GS) activation scales
static float   *OUT;     // B x M

static inline int32_t gdot(const int8_t*a,const int8_t*b){ int32_t s=0; for(int k=0;k<GS;k++) s+=(int32_t)a[k]*(int32_t)b[k]; return s; }

// A: un-blocked (colibri current) — per out row, per position, per group.
static double runA(void){
    double t=now_s();
    for(int i=0;i<M;i++){
        const int8_t* wr=W+(int64_t)i*K; const float* wsc=WS+(int64_t)i*(K/GS);
        for(int b=0;b<B;b++){
            const int8_t* xr=X+(int64_t)b*K; const float* xsc=XS+(int64_t)b*(K/GS);
            float val=0;
            for(int g=0;g<K/GS;g++) val+=(float)gdot(xr+g*GS,wr+g*GS)*xsc[g]*wsc[g];
            OUT[(int64_t)b*M+i]=val;
        }
    }
    return now_s()-t;
}

// B: position-blocked — process NB positions against each weight row together,
// so the weight row (K bytes) is loaded once from cache and reused across NB
// positions while their partial sums live in registers. Activation blocks for
// NB positions stay hot.
#define NB 8
static double runB(void){
    double t=now_s();
    for(int i=0;i<M;i++){
        const int8_t* wr=W+(int64_t)i*K; const float* wsc=WS+(int64_t)i*(K/GS);
        for(int b0=0;b0<B;b0+=NB){
            float acc[NB]; for(int b=0;b<NB;b++) acc[b]=0;
            for(int g=0;g<K/GS;g++){
                const int8_t* wg=wr+g*GS; float ws=wsc[g];
                for(int b=0;b<NB;b++){
                    const int8_t* xg=X+(int64_t)(b0+b)*K+g*GS;
                    acc[b]+=(float)gdot(xg,wg)*XS[(int64_t)(b0+b)*(K/GS)+g]*ws;
                }
            }
            for(int b=0;b<NB;b++) OUT[(int64_t)(b0+b)*M+i]=acc[b];
        }
    }
    return now_s()-t;
}

int main(void){
    W=malloc((int64_t)M*K); WS=malloc((int64_t)M*(K/GS)*4);
    X=malloc((int64_t)B*K); XS=malloc((int64_t)B*(K/GS)*4);
    OUT=malloc((int64_t)B*M*4);
    for(int64_t i=0;i<(int64_t)M*K;i++) W[i]=(int8_t)(i*7);
    for(int64_t i=0;i<(int64_t)M*(K/GS);i++) WS[i]=0.01f;
    for(int64_t i=0;i<(int64_t)B*K;i++) X[i]=(int8_t)(i*3);
    for(int64_t i=0;i<(int64_t)B*(K/GS);i++) XS[i]=0.02f;
    double flop=2.0*M*K*B;
    for(int r=0;r<2;r++){
        double a=runA(); double b=runB();
        if(r){ printf("A unblocked : %.2f GF/s (%.2fs)\nB pos-blocked: %.2f GF/s (%.2fs)  -> %.1fx\n",
            flop/1e9/a, a, flop/1e9/b, b, a/b); }
    }
    return 0;
}
