#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M 8192
#define K 2048
#define GS 64
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
// dot_q8-shape: M rows of K int8, GEMV (1 activation)
static double q8(const int8_t*W,const float*WS,const int8_t*x,const float*xs,float*out){
    double t=now_s();
    for(int i=0;i<M;i++){const int8_t*wr=W+(int64_t)i*K;const float*ws=WS+(int64_t)i*(K/GS);float v=0;
        for(int g=0;g<K/GS;g++){int32_t a=0;const int8_t*wg=wr+g*GS,*xg=x+g*GS;for(int k=0;k<GS;k++)a+=(int32_t)xg[k]*wg[k];v+=(float)a*xs[g]*ws[g];}
        out[i]=v;} return now_s()-t;
}
// dot_q4-shape: M rows of K/2 packed int4, GEMV
static double q4(const uint8_t*W,const float*WS,const int8_t*x,const float*xs,float*out){
    double t=now_s();int half=GS/2;
    for(int i=0;i<M;i++){const uint8_t*wr=W+(int64_t)i*(K/2);const float*ws=WS+(int64_t)i*(K/GS);float v=0;
        for(int g=0;g<K/GS;g++){int32_t a=0;const uint8_t*wg=wr+g*half;const int8_t*xg=x+g*GS;
            for(int k=0;k<half;k++){uint8_t b=wg[k];a+=(int32_t)xg[k]*((int32_t)(b&15)-8);a+=(int32_t)xg[k+half]*((int32_t)(b>>4)-8);}
            v+=(float)a*xs[g]*ws[g];}
        out[i]=v;} return now_s()-t;
}
int main(void){
    int8_t*W8=malloc((int64_t)M*K);uint8_t*W4=malloc((int64_t)M*(K/2));float*WS=malloc((int64_t)M*(K/GS)*4);
    int8_t*x=malloc(K);float*xs=malloc((K/GS)*4);float*out=malloc(M*4);
    for(int64_t i=0;i<(int64_t)M*K;i++)W8[i]=(int8_t)i;
    for(int64_t i=0;i<(int64_t)M*(K/2);i++)W4[i]=(uint8_t)i;
    for(int64_t i=0;i<(int64_t)M*(K/GS);i++)WS[i]=0.01f;
    for(int i=0;i<K;i++)x[i]=(int8_t)i; for(int i=0;i<K/GS;i++)xs[i]=0.02f;
    double bytes8=(double)M*K, bytes4=(double)M*(K/2);
    for(int r=0;r<3;r++){double a=q8(W8,WS,x,xs,out),b=q4(W4,WS,x,xs,out);
        if(r==2)printf("q8: %.1f GB/s weights (%.3fs)\nq4: %.1f GB/s weights (%.3fs)  [q4 moves half the bytes]\n",bytes8/1e9/a,a,bytes4/1e9/b,b);}
    return 0;
}
