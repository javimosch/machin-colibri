#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <ctype.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <sys/select.h>
#ifndef __wasm__
#include <sys/wait.h>
#include <signal.h>
#endif

/* slices: a Go-style header over an unboxed backing array */
typedef struct { void* data; int64_t len; int64_t cap; } mfl_slice;

/* closures: a function pointer plus a heap environment of captured values */
typedef struct { void* fn; void* env; } mfl_closure;

/* ---- arena memory management ----
   Value buffers (strings, slice backings, closure environments) are allocated
   from a per-goroutine arena and reclaimed in bulk when the goroutine finishes.
   The main goroutine's arena lives for the whole program. This bounds the
   memory of a long-running concurrent server: each request handler runs in its
   own goroutine and frees everything it allocated on return. Subsystems that
   free explicitly (channels, maps, goroutine args) use raw malloc/free. */
typedef struct mfl_blk { struct mfl_blk* next; size_t size; } mfl_blk;
typedef struct { mfl_blk* head; } mfl_arena;
static mfl_arena mfl_main_arena = { NULL };
static _Thread_local mfl_arena* mfl_arena_cur = NULL;
static void* mfl_alloc(size_t sz) {
    if (!mfl_arena_cur) mfl_arena_cur = &mfl_main_arena;
    if (sz == 0) sz = 1;
    mfl_blk* b = malloc(sizeof(mfl_blk) + sz);
    b->size = sz; b->next = mfl_arena_cur->head; mfl_arena_cur->head = b;
    return (void*)(b + 1);
}
/* mfl_calloc is only ever used to box a local captured by a nested closure
   (see function() in codegen.go); that box must outlive the arena of whoever
   declares it, so it's plain calloc, not mfl_alloc (#314). */
static void* mfl_calloc(size_t n, size_t sz) { return calloc(n, sz); }
static void* mfl_realloc(void* old, size_t sz) {
    void* p = mfl_alloc(sz);
    if (old) { size_t o = ((mfl_blk*)old - 1)->size; memcpy(p, old, o < sz ? o : sz); }
    return p; /* old reclaimed with its arena */
}
/* substr strlen-cache: mfl_substr only needs strlen(s) to clamp the end offset.
   In hot loops (the lexer, parsers, scanners) the same source pointer is sliced
   thousands of times, so caching its length by pointer identity turns a per-call
   O(strlen) scan into O(1) while preserving exact clamping semantics. Two distinct
   *live* strings never share an address (the arena frees nothing mid-life), so
   pointer identity ⇒ same length — but a freed block's address can be reused by a
   later malloc, so mfl_arena_free invalidates the cache. */
static _Thread_local const char* mfl_strlen_cache_s = NULL;
static _Thread_local int64_t mfl_strlen_cache_n = 0;
static inline int64_t mfl_strlen_cached(const char* s) {
    if (s == mfl_strlen_cache_s) return mfl_strlen_cache_n;
    int64_t n = (int64_t)strlen(s);
    mfl_strlen_cache_s = s; mfl_strlen_cache_n = n;
    return n;
}
static void mfl_arena_free(mfl_arena* a) {
    mfl_blk* b = a->head;
    while (b) { mfl_blk* n = b->next; free(b); b = n; }
    a->head = NULL;
    mfl_strlen_cache_s = NULL; /* freed addresses may be reused — drop stale length */
}

/* --safe runtime checks (used only when the program is built with --safe) */
static void mfl_panic(const char* msg) { fputs("panic: ", stderr); fputs(msg, stderr); fputc('\n', stderr); exit(1); }
static int64_t mfl_bounds(int64_t i, int64_t n) {
    if (i < 0 || i >= n) { char b[80]; snprintf(b, 80, "index out of range [%lld] with length %lld", (long long)i, (long long)n); mfl_panic(b); }
    return i;
}
static int64_t mfl_idiv(int64_t a, int64_t b) { if (b == 0) mfl_panic("integer divide by zero"); return a / b; }
static int64_t mfl_imod(int64_t a, int64_t b) { if (b == 0) mfl_panic("integer modulo by zero"); return a % b; }
static int64_t mfl_iadd(int64_t a, int64_t b) { int64_t r; if (__builtin_add_overflow(a, b, &r)) mfl_panic("integer overflow (+)"); return r; }
static int64_t mfl_isub(int64_t a, int64_t b) { int64_t r; if (__builtin_sub_overflow(a, b, &r)) mfl_panic("integer overflow (-)"); return r; }
static int64_t mfl_imul(int64_t a, int64_t b) { int64_t r; if (__builtin_mul_overflow(a, b, &r)) mfl_panic("integer overflow (*)"); return r; }

static mfl_slice mfl_append(mfl_slice s, const void* elem, int64_t es) {
    if (s.len >= s.cap) {
        int64_t nc = s.cap ? s.cap * 2 : 4;
        s.data = mfl_realloc(s.data, nc * es); s.cap = nc;
    }
    memcpy((char*)s.data + s.len * es, elem, es);
    s.len++;
    return s;
}
static mfl_slice mfl_lit_i64(int64_t n, ...) {
    mfl_slice s = { n ? mfl_alloc(n * 8) : NULL, n, n };
    va_list ap; va_start(ap, n);
    for (int64_t i = 0; i < n; i++) ((int64_t*)s.data)[i] = va_arg(ap, int64_t);
    va_end(ap); return s;
}
static mfl_slice mfl_lit_f64(int64_t n, ...) {
    mfl_slice s = { n ? mfl_alloc(n * 8) : NULL, n, n };
    va_list ap; va_start(ap, n);
    for (int64_t i = 0; i < n; i++) ((double*)s.data)[i] = va_arg(ap, double);
    va_end(ap); return s;
}
static mfl_slice mfl_lit_str(int64_t n, ...) {
    mfl_slice s = { n ? mfl_alloc(n * sizeof(char*)) : NULL, n, n };
    va_list ap; va_start(ap, n);
    for (int64_t i = 0; i < n; i++) ((char**)s.data)[i] = va_arg(ap, char*);
    va_end(ap); return s;
}
static int64_t mfl_exit(int64_t code) { exit((int)code); return 0; }
static int64_t mfl_flush(void) { fflush(stdout); return 0; }
static void mfl_sleep(int64_t ms) {
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

/* channels: a mutex + condvar FIFO. An element's heap data lives in the sending
   goroutine's arena, which is reclaimed when that goroutine finishes — so the
   channel must copy it somewhere stable on send and into the receiver's arena on
   receive. Two marshaling modes (chosen by codegen from the element type):
     - string mode (stroff[]): byte offsets of every string (char*) reachable by
       value in the element; send deep-copies them, receive adopts them.
     - json mode (ser/des): for elements containing a slice or map, the whole
       value is serialized to JSON on send and parsed back on receive — a general
       deep copy that handles arbitrary nesting (slices, maps, structs).
   Scalar elements need neither and are a plain memcpy. */
static char* mfl_dup_arena(const char* s, size_t n); /* defined later in the runtime */
typedef struct mfl_cnode { struct mfl_cnode* next; void* data; } mfl_cnode;
typedef struct {
    pthread_mutex_t mu; pthread_cond_t cnd;
    mfl_cnode *head, *tail; int64_t es; int closed;
    int nstr; int* stroff;
    char* (*ser)(const void*);          /* json mode: element -> arena JSON */
    void (*des)(const char*, void*);    /* json mode: JSON -> *out (arena) */
} mfl_chan;
static mfl_chan* mfl_make_chan(int64_t es, char* (*ser)(const void*), void (*des)(const char*, void*), int nstr, ...) {
    mfl_chan* c = malloc(sizeof(mfl_chan));
    pthread_mutex_init(&c->mu, NULL); pthread_cond_init(&c->cnd, NULL);
    c->head = c->tail = NULL; c->es = es; c->closed = 0;
    c->ser = ser; c->des = des;
    c->nstr = nstr; c->stroff = NULL;
    if (nstr > 0) {
        c->stroff = (int*)malloc((size_t)nstr * sizeof(int));
        va_list ap; va_start(ap, nstr);
        for (int i = 0; i < nstr; i++) c->stroff[i] = va_arg(ap, int);
        va_end(ap);
    }
    return c;
}
/* freeze: replace each string field (found at the given byte offsets within
   elem) with a stable malloc'd copy -- the value is being handed off away
   from the current arena (to a channel, or across a go-statement's arena
   boundary; #310). Shared by mfl_chan_freeze and mfl_go's argument passing. */
static void mfl_freeze_strs(int nstr, int* stroff, void* elem) {
    for (int i = 0; i < nstr; i++) {
        char** p = (char**)((char*)elem + stroff[i]);
        if (*p) { size_t n = strlen(*p); char* d = (char*)malloc(n + 1); memcpy(d, *p, n + 1); *p = d; }
    }
}
/* thaw: move each frozen (malloc'd) string into the CURRENT arena, freeing the
   malloc'd copy. After this the value's strings live exactly as long as
   whichever goroutine's arena is current when this runs. */
static void mfl_thaw_strs(int nstr, int* stroff, void* elem) {
    for (int i = 0; i < nstr; i++) {
        char** p = (char**)((char*)elem + stroff[i]);
        if (*p) { char* a = mfl_dup_arena(*p, strlen(*p)); free(*p); *p = a; }
    }
}
static void mfl_chan_freeze(mfl_chan* c, void* elem) { mfl_freeze_strs(c->nstr, c->stroff, elem); }
static void mfl_chan_thaw(mfl_chan* c, void* elem) { mfl_thaw_strs(c->nstr, c->stroff, elem); }
/* close a channel: receivers drain the buffer then get "not ok". Wakes every
   blocked receiver so range/recv stop instead of hanging forever. */
static void mfl_chan_close(mfl_chan* c) {
    pthread_mutex_lock(&c->mu);
    c->closed = 1;
    pthread_cond_broadcast(&c->cnd);
    pthread_mutex_unlock(&c->mu);
}
static void mfl_chan_send(mfl_chan* c, const void* v) {
    mfl_cnode* n = malloc(sizeof(mfl_cnode));
    if (c->ser) {
        char* j = c->ser(v);                 /* arena JSON of the whole value */
        size_t L = strlen(j);
        n->data = malloc(L + 1); memcpy(n->data, j, L + 1);
    } else {
        n->data = malloc(c->es); memcpy(n->data, v, c->es);
        mfl_chan_freeze(c, n->data);
    }
    n->next = NULL;
    pthread_mutex_lock(&c->mu);
    if (c->tail) c->tail->next = n; else c->head = n;
    c->tail = n;
    pthread_cond_signal(&c->cnd);
    pthread_mutex_unlock(&c->mu);
}
/* deliver node n's payload into out (receiver arena), then free the node. */
static void mfl_chan_deliver(mfl_chan* c, mfl_cnode* n, void* out) {
    if (c->des) {
        c->des((const char*)n->data, out);
    } else {
        memcpy(out, n->data, c->es);
        mfl_chan_thaw(c, out);
    }
    free(n->data); free(n);
}
/* blocking receive with ok: 1 and fills out if a value arrived; 0 if the channel
   is closed and drained (out left untouched). The primitive behind range-over-
   channel and the comma-ok receive. */
static int mfl_chan_recv2(mfl_chan* c, void* out) {
    pthread_mutex_lock(&c->mu);
    while (!c->head && !c->closed) pthread_cond_wait(&c->cnd, &c->mu);
    mfl_cnode* n = c->head;
    if (!n) { pthread_mutex_unlock(&c->mu); return 0; }
    c->head = n->next;
    if (!c->head) c->tail = NULL;
    pthread_mutex_unlock(&c->mu);
    mfl_chan_deliver(c, n, out);
    return 1;
}
static void mfl_chan_recv(mfl_chan* c, void* out) {
    if (!mfl_chan_recv2(c, out)) memset(out, 0, c->es);
}
/* non-blocking receive for select: returns 1 if the case is ready — either a
   value arrived (*ok = 1, out filled) or the channel is closed and drained
   (*ok = 0, out zeroed). Returns 0 if not ready (open and empty). */
static int mfl_chan_tryrecv2(mfl_chan* c, void* out, int* ok) {
    pthread_mutex_lock(&c->mu);
    mfl_cnode* n = c->head;
    if (n) { c->head = n->next; if (!c->head) c->tail = NULL; }
    int closed = c->closed;
    pthread_mutex_unlock(&c->mu);
    if (n) { mfl_chan_deliver(c, n, out); *ok = 1; return 1; }
    if (closed) { memset(out, 0, c->es); *ok = 0; return 1; }
    return 0;
}

/* maps: a chained hash table keyed by int64 or string, fixed-size values */
typedef struct mfl_ment { struct mfl_ment* next; int64_t ik; char* sk; void* val; } mfl_ment;
typedef struct { mfl_ment** buckets; int64_t nb, count, vs; int sk; } mfl_map;
static uint64_t mfl_hash_i(int64_t k) { uint64_t x=(uint64_t)k; x^=x>>33; x*=0xff51afd7ed558ccdULL; x^=x>>33; return x; }
static uint64_t mfl_hash_s(const char* s) { uint64_t h=1469598103934665603ULL; while(*s){ h^=(unsigned char)*s++; h*=1099511628211ULL; } return h; }
static mfl_map* mfl_make_map(int keyIsStr, int64_t vs) {
    mfl_map* m = malloc(sizeof(mfl_map));
    m->nb = 16; m->count = 0; m->sk = keyIsStr; m->vs = vs;
    m->buckets = calloc(m->nb, sizeof(mfl_ment*));
    return m;
}
static mfl_ment** mfl_map_at(mfl_map* m, int64_t ik, const char* sk) {
    uint64_t h = m->sk ? mfl_hash_s(sk) : mfl_hash_i(ik);
    mfl_ment** pp = &m->buckets[h & (m->nb - 1)];
    while (*pp) { mfl_ment* e=*pp; if (m->sk ? strcmp(e->sk,sk)==0 : e->ik==ik) return pp; pp=&e->next; }
    return pp;
}
/* Double the bucket array and rehash when the load factor hits 1. Without this
   the table stays at 16 buckets forever, so N inserts are O(N^2) (every insert
   walks a chain of length N/16) — 25s for a 128k-entry map. Amortized O(1). */
static void mfl_map_grow(mfl_map* m) {
    int64_t nn = m->nb * 2;
    mfl_ment** nb2 = calloc(nn, sizeof(mfl_ment*));
    for (int64_t b = 0; b < m->nb; b++) {
        mfl_ment* e = m->buckets[b];
        while (e) {
            mfl_ment* nx = e->next;
            uint64_t h = m->sk ? mfl_hash_s(e->sk) : mfl_hash_i(e->ik);
            int64_t idx = (int64_t)(h & (uint64_t)(nn - 1));
            e->next = nb2[idx]; nb2[idx] = e;
            e = nx;
        }
    }
    free(m->buckets); m->buckets = nb2; m->nb = nn;
}
static void mfl_map_set(mfl_map* m, int64_t ik, const char* sk, const void* val) {
    mfl_ment** pp = mfl_map_at(m, ik, sk);
    if (*pp) { memcpy((*pp)->val, val, m->vs); return; }
    mfl_ment* e = malloc(sizeof(mfl_ment)); e->next=NULL; e->ik=ik; e->sk=NULL;
    if (m->sk) { e->sk = malloc(strlen(sk)+1); strcpy(e->sk, sk); }
    e->val = malloc(m->vs); memcpy(e->val, val, m->vs);
    *pp = e; m->count++;
    if (m->count > m->nb) mfl_map_grow(m);
}
static void mfl_map_get(mfl_map* m, int64_t ik, const char* sk, void* out) {
    mfl_ment** pp = mfl_map_at(m, ik, sk);
    if (*pp) memcpy(out, (*pp)->val, m->vs); else memset(out, 0, m->vs);
}
static int mfl_map_has(mfl_map* m, int64_t ik, const char* sk) { return *mfl_map_at(m, ik, sk) != NULL; }
static void mfl_map_del(mfl_map* m, int64_t ik, const char* sk) {
    mfl_ment** pp = mfl_map_at(m, ik, sk);
    if (*pp) { mfl_ment* e=*pp; *pp=e->next; free(e->sk); free(e->val); free(e); m->count--; }
}
static int64_t mfl_map_len(mfl_map* m) { return m->count; }
static mfl_slice mfl_map_keys(mfl_map* m) {
    int64_t es = m->sk ? (int64_t)sizeof(char*) : (int64_t)sizeof(int64_t);
    mfl_slice s = { m->count ? mfl_alloc(m->count*es) : NULL, m->count, m->count };
    int64_t idx = 0;
    for (int64_t b = 0; b < m->nb; b++)
        for (mfl_ment* e = m->buckets[b]; e; e = e->next) {
            if (m->sk) ((char**)s.data)[idx] = e->sk; else ((int64_t*)s.data)[idx] = e->ik;
            idx++;
        }
    return s;
}

// A string's zero value is "" — but an auto-zeroed string slot (an omitted struct
// literal field, a grown slice element, a default map value) is a NULL char*. So
// the string ops treat NULL as "" rather than dereferencing it.
static const char* mfl_s(const char* s) { return s ? s : ""; }
static int mfl_strcmp(const char* a, const char* b) { return strcmp(mfl_s(a), mfl_s(b)); }
static char* mfl_cat(const char* a, const char* b) {
    a = mfl_s(a); b = mfl_s(b);
    size_t la = strlen(a), lb = strlen(b);
    char* r = mfl_alloc(la + lb + 1);
    memcpy(r, a, la); memcpy(r + la, b, lb); r[la + lb] = 0;
    return r;
}
static char* mfl_str_i(int64_t v) { char* b = mfl_alloc(24); snprintf(b, 24, "%lld", (long long)v); return b; }
static char* mfl_str_d(double v)  { char* b = mfl_alloc(32); snprintf(b, 32, "%g", v); return b; }
/* reinterpret a double's IEEE-754 bits as an int64 and back — the byte-level access
   needed to (de)serialize 64-bit floats (e.g. BSON doubles). */
static int64_t mfl_f64_bits(double d) { int64_t i; memcpy(&i, &d, 8); return i; }
static double mfl_f64_from_bits(int64_t i) { double d; memcpy(&d, &i, 8); return d; }
static char* mfl_str_b(int64_t v) { return v ? "true" : "false"; }
static char* mfl_dup(const char* s) { size_t n = strlen(s); char* r = mfl_alloc(n+1); memcpy(r, s, n+1); return r; }
/* raw heap memory: a pointer is an int (intptr_t round-trip), as with the ptr
   FFI type. alloc() is zeroed (calloc) so building C structs is safe. For
   filling C buffers (vertex arrays) and structs to hand to a C API. */
static int64_t mfl_raw_alloc(int64_t n) { return (int64_t)(intptr_t)calloc(1, (size_t)(n > 0 ? n : 0)); }
static void mfl_raw_free(int64_t p) { free((void*)(intptr_t)p); }
/* mmap a file read-only into memory -> (pointer-as-int, size-in-bytes), or (0,0)
   on error. Zero-copy access to a large on-disk buffer (a model checkpoint, a
   memory-mapped asset) via peek_*; pages fault in lazily. The mapping lives
   until the process exits. Multi-assign only: p, n := mmap_file(path). */
typedef struct { int64_t ptr; int64_t len; } mfl_mmap_result;
static mfl_mmap_result mfl_mmap_file(const char* path) {
    mfl_mmap_result R; R.ptr = 0; R.len = 0;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return R;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return R; }
    void* p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return R;
    R.ptr = (int64_t)(intptr_t)p; R.len = (int64_t)st.st_size;
    return R;
}
/* read a NUL-terminated string from a raw pointer into an MFL (arena) string — the
   host->wasm direction: the JS host writes UTF-8 + a NUL into wasm memory at a
   pointer the program alloc'd, then passes it here. */
static char* mfl_ptr_str(int64_t p) { return p ? mfl_dup((const char*)(intptr_t)p) : mfl_dup(""); }
static void mfl_poke_f32(int64_t p, int64_t o, double v) { *(float*)((char*)(intptr_t)p + o) = (float)v; }
static void mfl_poke_i32(int64_t p, int64_t o, int64_t v) { *(int32_t*)((char*)(intptr_t)p + o) = (int32_t)v; }
static void mfl_poke_u8(int64_t p, int64_t o, int64_t v) { *(uint8_t*)((char*)(intptr_t)p + o) = (uint8_t)v; }
static void mfl_poke_u16(int64_t p, int64_t o, int64_t v) { *(uint16_t*)((char*)(intptr_t)p + o) = (uint16_t)v; }
static void mfl_poke_ptr(int64_t p, int64_t o, int64_t v) { *(void**)((char*)(intptr_t)p + o) = (void*)(intptr_t)v; }
static double mfl_peek_f32(int64_t p, int64_t o) { return (double)*(float*)((char*)(intptr_t)p + o); }
static int64_t mfl_peek_i32(int64_t p, int64_t o) { return (int64_t)*(int32_t*)((char*)(intptr_t)p + o); }
static int64_t mfl_peek_i8(int64_t p, int64_t o) { return (int64_t)*(int8_t*)((char*)(intptr_t)p + o); }
static int64_t mfl_peek_u8(int64_t p, int64_t o) { return (int64_t)*(uint8_t*)((char*)(intptr_t)p + o); }
/* signed-byte dot product with a 32-bit accumulator (the vector-friendly width:
 * cc autovectorizes this where an int64 reduction stays half-speed). Exact while
 * |sum| < 2^31 - always true for i8*i8 up to n ~ 133k - the quantized-matmul
 * group kernel of the AI domain, as sha256 is to the crypto domain. */
static int64_t mfl_dot_i8(int64_t a, int64_t b, int64_t n) {
    const int8_t* x = (const int8_t*)(intptr_t)a;
    const int8_t* w = (const int8_t*)(intptr_t)b;
    int32_t acc = 0;
    for (int64_t k = 0; k < n; k++) acc += (int32_t)x[k] * (int32_t)w[k];
    return (int64_t)acc;
}
/* grouped, dual-scaled int8 dot: sum over length-gs groups of
 * (int32 group dot) * xscale[g] * wscale[g]. The whole quantized-matmul inner
 * product in ONE call -- one int32 reduction per group (autovectorizes to
 * vpmaddwd/vpdpbusd) with the two per-group fp32 scales applied group-at-a-time.
 * Replaces an MFL loop that called dot_i8 + two peek_f32 per group, whose
 * per-group call overhead capped throughput. xq/wq are int8 buffers (n bytes);
 * xs/ws are fp32 group-scale buffers (n/gs floats). n must be a multiple of gs. */
/* float32 dot product of two raw f32 buffers (a[k]*b[k], k<n) with an fp32
 * accumulator -- the vectorizable inner product for attention scores (q.k) and
 * any dense float kernel, where an MFL loop of peek_f32*peek_f32 is scalar and
 * call-bound. */
static double mfl_dot_f32(int64_t a, int64_t b, int64_t n) {
    const float* x = (const float*)(intptr_t)a;
    const float* y = (const float*)(intptr_t)b;
    float acc = 0.0f;
    for (int64_t k = 0; k < n; k++) acc += x[k] * y[k];
    return (double)acc;
}
/* AXPY: y[k] += s * x[k] for k<n, over raw f32 buffers. The attention value
 * accumulation (weighted sum of V rows) and any scaled-add; vectorizes where an
 * MFL peek/poke loop cannot. */
static void mfl_axpy_f32(int64_t y, double s, int64_t x, int64_t n) {
    float* yy = (float*)(intptr_t)y;
    const float* xx = (const float*)(intptr_t)x;
    float sf = (float)s;
    for (int64_t k = 0; k < n; k++) yy[k] += sf * xx[k];
}
static double mfl_dot_q8(int64_t xq, int64_t xs, int64_t wq, int64_t ws, int64_t n, int64_t gs) {
    const int8_t* x = (const int8_t*)(intptr_t)xq;
    const float* xsc = (const float*)(intptr_t)xs;
    const int8_t* w = (const int8_t*)(intptr_t)wq;
    const float* wsc = (const float*)(intptr_t)ws;
    double val = 0.0;
    int64_t ng = gs > 0 ? n / gs : 0;
    for (int64_t g = 0; g < ng; g++) {
        const int8_t* xg = x + g * gs;
        const int8_t* wg = w + g * gs;
        int32_t acc = 0;
        for (int64_t k = 0; k < gs; k++) acc += (int32_t)xg[k] * (int32_t)wg[k];
        val += (double)acc * (double)xsc[g] * (double)wsc[g];
    }
    return val;
}
/* grouped, dual-scaled int4 dot: like dot_q8 but the weights are split-nibble
 * int4 (a group of gs weights packed into gs/2 bytes; byte k holds w[k] in the
 * low nibble and w[k+gs/2] in the high nibble, each stored as value+8 in 0..15).
 * Activations stay int8. Halves the weight bytes moved -- the memory-bound-decode
 * win. n multiple of gs; wq has n/2 packed bytes. */
static double mfl_dot_q4(int64_t xq, int64_t xs, int64_t wq, int64_t ws, int64_t n, int64_t gs) {
    const int8_t* x = (const int8_t*)(intptr_t)xq;
    const float* xsc = (const float*)(intptr_t)xs;
    const uint8_t* w = (const uint8_t*)(intptr_t)wq;
    const float* wsc = (const float*)(intptr_t)ws;
    double val = 0.0;
    int64_t ng = gs > 0 ? n / gs : 0;
    int64_t half = gs / 2;
    for (int64_t g = 0; g < ng; g++) {
        const int8_t* xg = x + g * gs;
        const uint8_t* wg = w + g * half;
        int32_t acc = 0;
        for (int64_t k = 0; k < half; k++) {
            uint8_t b = wg[k];
            acc += (int32_t)xg[k]        * ((int32_t)(b & 15) - 8);
            acc += (int32_t)xg[k + half] * ((int32_t)(b >> 4) - 8);
        }
        val += (double)acc * (double)xsc[g] * (double)wsc[g];
    }
    return val;
}
/* base64 (standard alphabet, padded) over text. */
static char* mfl_base64_encode(const char* s) {
    static const char t[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t n = strlen(s), j = 0, i = 0;
    char* out = (char*)mfl_alloc(4 * ((n + 2) / 3) + 1);
    for (; i + 3 <= n; i += 3) {
        uint32_t v = ((uint32_t)(unsigned char)s[i] << 16) | ((uint32_t)(unsigned char)s[i+1] << 8) | (unsigned char)s[i+2];
        out[j++] = t[(v >> 18) & 63]; out[j++] = t[(v >> 12) & 63];
        out[j++] = t[(v >> 6) & 63];  out[j++] = t[v & 63];
    }
    if (n - i == 1) {
        uint32_t v = (uint32_t)(unsigned char)s[i] << 16;
        out[j++] = t[(v >> 18) & 63]; out[j++] = t[(v >> 12) & 63];
        out[j++] = '='; out[j++] = '=';
    } else if (n - i == 2) {
        uint32_t v = ((uint32_t)(unsigned char)s[i] << 16) | ((uint32_t)(unsigned char)s[i+1] << 8);
        out[j++] = t[(v >> 18) & 63]; out[j++] = t[(v >> 12) & 63];
        out[j++] = t[(v >> 6) & 63];  out[j++] = '=';
    }
    out[j] = 0;
    return out;
}
/* lenient base64 decode: accepts standard and url-safe ('-' '_'), ignores
   padding/whitespace, so it also decodes JWT segments. "" for empty input. */
static int mfl_b64val(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}
static char* mfl_base64_decode(const char* s) {
    size_t n = strlen(s), j = 0;
    char* out = (char*)mfl_alloc(n + 1);
    int buf = 0, bits = 0;
    for (size_t i = 0; i < n; i++) {
        int v = mfl_b64val((unsigned char)s[i]);
        if (v < 0) continue;
        buf = (buf << 6) | v; bits += 6;
        if (bits >= 8) { bits -= 8; out[j++] = (char)((buf >> bits) & 0xFF); }
    }
    out[j] = 0;
    return out;
}
/* URL percent-encoding (RFC 3986). url_encode keeps the unreserved set
   A-Za-z0-9-._~ and %XX-encodes everything else (space -> %20). url_decode
   reverses it and is lenient: it also maps '+' to space (form style) and passes
   a malformed % through unchanged. */
static int mfl_hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static char* mfl_url_encode(const char* s) {
    static const char hex[] = "0123456789ABCDEF";
    size_t n = strlen(s), j = 0;
    char* out = (char*)mfl_alloc(n * 3 + 1);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
            c == '-' || c == '_' || c == '.' || c == '~') {
            out[j++] = (char)c;
        } else {
            out[j++] = '%';
            out[j++] = hex[c >> 4];
            out[j++] = hex[c & 15];
        }
    }
    out[j] = 0;
    return out;
}
static char* mfl_url_decode(const char* s) {
    size_t n = strlen(s), j = 0;
    char* out = (char*)mfl_alloc(n + 1);
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (c == '+') {
            out[j++] = ' ';
        } else if (c == '%' && i + 2 < n) {
            int hi = mfl_hexval((unsigned char)s[i+1]);
            int lo = mfl_hexval((unsigned char)s[i+2]);
            if (hi >= 0 && lo >= 0) { out[j++] = (char)((hi << 4) | lo); i += 2; }
            else { out[j++] = c; }
        } else {
            out[j++] = c;
        }
    }
    out[j] = 0;
    return out;
}
/* bytes: a NUL-safe binary buffer (pointer + length), the type strings can't be.
   Values are immutable (builtins return fresh arena buffers), so passing one by
   value just shares the backing — same discipline as strings. */
typedef struct { uint8_t* data; int64_t len; } mfl_bytes;
static mfl_bytes mfl_bytes_from_str(const char* s) {
    int64_t n = (int64_t)strlen(s);
    mfl_bytes b; b.len = n; b.data = (uint8_t*)mfl_alloc(n ? n : 1);
    memcpy(b.data, s, (size_t)n);
    return b;
}
static char* mfl_bytes_str(mfl_bytes b) {   /* NUL-terminated; truncates at an embedded 0 */
    char* out = (char*)mfl_alloc(b.len + 1);
    memcpy(out, b.data, (size_t)b.len);
    out[b.len] = 0;
    return out;
}
static char* mfl_bytes_hex(mfl_bytes b) {
    static const char hx[] = "0123456789abcdef";
    char* out = (char*)mfl_alloc(b.len * 2 + 1);
    for (int64_t i = 0; i < b.len; i++) { out[i*2] = hx[b.data[i] >> 4]; out[i*2+1] = hx[b.data[i] & 15]; }
    out[b.len*2] = 0;
    return out;
}
static mfl_bytes mfl_bytes_unhex(const char* s) {   /* skips non-hex chars (spaces, colons) */
    int64_t n = (int64_t)strlen(s);
    mfl_bytes b; b.len = 0; b.data = (uint8_t*)mfl_alloc(n / 2 + 1);
    int hi = -1;
    for (int64_t i = 0; i < n; i++) {
        int v = mfl_hexval((unsigned char)s[i]);
        if (v < 0) continue;
        if (hi < 0) hi = v;
        else { b.data[b.len++] = (uint8_t)((hi << 4) | v); hi = -1; }
    }
    return b;
}
static int64_t mfl_byte_at(mfl_bytes b, int64_t i) { return (i < 0 || i >= b.len) ? -1 : (int64_t)b.data[i]; }
static mfl_bytes mfl_bytes_sub(mfl_bytes b, int64_t start, int64_t end) {
    if (start < 0) start = 0;
    if (end > b.len) end = b.len;
    if (end < start) end = start;
    int64_t n = end - start;
    mfl_bytes r; r.len = n; r.data = (uint8_t*)mfl_alloc(n ? n : 1);
    memcpy(r.data, b.data + start, (size_t)n);
    return r;
}
/* find needle in haystack at or after start, NUL-safe; -1 if absent. For binary
   protocols (e.g. a multipart/form-data boundary inside an upload body). */
static int64_t mfl_bytes_index(mfl_bytes h, mfl_bytes nd, int64_t from) {
    if (from < 0) from = 0;
    if (nd.len == 0) return from <= h.len ? from : -1;
    for (int64_t i = from; i + nd.len <= h.len; i++) {
        if (memcmp(h.data + i, nd.data, (size_t)nd.len) == 0) return i;
    }
    return -1;
}
static mfl_bytes mfl_bytes_concat(mfl_bytes a, mfl_bytes b) {
    mfl_bytes r; r.len = a.len + b.len; r.data = (uint8_t*)mfl_alloc(r.len ? r.len : 1);
    memcpy(r.data, a.data, (size_t)a.len);
    memcpy(r.data + a.len, b.data, (size_t)b.len);
    return r;
}
/* binary-safe base64: encode raw bytes (incl. NUL), decode to raw bytes. The
   string forms (mfl_base64_encode/decode) stop at a NUL; these carry an explicit
   length, for binary protocols / crypto (e.g. SCRAM salts and proofs). */
static char* mfl_base64_encode_bytes(mfl_bytes b) {
    static const char t[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t n = (size_t)b.len, j = 0, i = 0; const unsigned char* s = b.data;
    char* out = (char*)mfl_alloc(4 * ((n + 2) / 3) + 1);
    for (; i + 3 <= n; i += 3) {
        uint32_t v = ((uint32_t)s[i] << 16) | ((uint32_t)s[i+1] << 8) | s[i+2];
        out[j++] = t[(v >> 18) & 63]; out[j++] = t[(v >> 12) & 63];
        out[j++] = t[(v >> 6) & 63];  out[j++] = t[v & 63];
    }
    if (n - i == 1) {
        uint32_t v = (uint32_t)s[i] << 16;
        out[j++] = t[(v >> 18) & 63]; out[j++] = t[(v >> 12) & 63];
        out[j++] = '='; out[j++] = '=';
    } else if (n - i == 2) {
        uint32_t v = ((uint32_t)s[i] << 16) | ((uint32_t)s[i+1] << 8);
        out[j++] = t[(v >> 18) & 63]; out[j++] = t[(v >> 12) & 63];
        out[j++] = t[(v >> 6) & 63];  out[j++] = '=';
    }
    out[j] = 0;
    return out;
}
static mfl_bytes mfl_base64_decode_bytes(const char* s) {
    size_t n = strlen(s), j = 0;
    mfl_bytes b; b.data = (uint8_t*)mfl_alloc(n ? n : 1);
    int buf = 0, bits = 0;
    for (size_t i = 0; i < n; i++) {
        int v = mfl_b64val((unsigned char)s[i]);
        if (v < 0) continue;
        buf = (buf << 6) | v; bits += 6;
        if (bits >= 8) { bits -= 8; b.data[j++] = (uint8_t)((buf >> bits) & 0xFF); }
    }
    b.len = (int64_t)j;
    return b;
}
/* SHA-256 + HMAC-SHA256 (pure C, no dependency). Operate on NUL-terminated text
   and return a lowercase hex digest. */
static uint32_t mfl_ror32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static void mfl_sha256_raw(const unsigned char* msg, size_t len, unsigned char out[32]) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t newlen = ((len + 8) / 64 + 1) * 64;
    unsigned char* m = (unsigned char*)calloc(newlen, 1);
    memcpy(m, msg, len);
    m[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) m[newlen - 1 - i] = (unsigned char)(bits >> (8 * i));
    for (size_t off = 0; off < newlen; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)m[off+i*4] << 24) | ((uint32_t)m[off+i*4+1] << 16) | ((uint32_t)m[off+i*4+2] << 8) | m[off+i*4+3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = mfl_ror32(w[i-15],7) ^ mfl_ror32(w[i-15],18) ^ (w[i-15] >> 3);
            uint32_t s1 = mfl_ror32(w[i-2],17) ^ mfl_ror32(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = mfl_ror32(e,6) ^ mfl_ror32(e,11) ^ mfl_ror32(e,25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = mfl_ror32(a,2) ^ mfl_ror32(a,13) ^ mfl_ror32(a,22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    free(m);
    for (int i = 0; i < 8; i++) { out[i*4]=(unsigned char)(h[i]>>24); out[i*4+1]=(unsigned char)(h[i]>>16); out[i*4+2]=(unsigned char)(h[i]>>8); out[i*4+3]=(unsigned char)h[i]; }
}
static char* mfl_hex32(const unsigned char d[32]) {
    static const char* h = "0123456789abcdef";
    char* s = (char*)mfl_alloc(65);
    for (int i = 0; i < 32; i++) { s[i*2] = h[d[i] >> 4]; s[i*2+1] = h[d[i] & 15]; }
    s[64] = 0;
    return s;
}
static char* mfl_sha256(const char* s) {
    unsigned char d[32];
    mfl_sha256_raw((const unsigned char*)s, strlen(s), d);
    return mfl_hex32(d);
}
static char* mfl_hmac_sha256(const char* key, const char* msg) {
    unsigned char k[64];
    size_t klen = strlen(key);
    if (klen > 64) {
        unsigned char kh[32];
        mfl_sha256_raw((const unsigned char*)key, klen, kh);
        memcpy(k, kh, 32); memset(k + 32, 0, 32);
    } else {
        memcpy(k, key, klen); memset(k + klen, 0, 64 - klen);
    }
    unsigned char ipad[64], opad[64];
    for (int i = 0; i < 64; i++) { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5c; }
    size_t mlen = strlen(msg);
    unsigned char* ibuf = (unsigned char*)malloc(64 + mlen);
    memcpy(ibuf, ipad, 64); memcpy(ibuf + 64, msg, mlen);
    unsigned char inner[32];
    mfl_sha256_raw(ibuf, 64 + mlen, inner);
    free(ibuf);
    unsigned char obuf[96];
    memcpy(obuf, opad, 64); memcpy(obuf + 64, inner, 32);
    unsigned char d[32];
    mfl_sha256_raw(obuf, 96, d);
    return mfl_hex32(d);
}
static char* mfl_json_str(const char* s) { /* quote + escape a string */
    if (!s) s = "";
    size_t n = strlen(s), j = 0;
    char* b = mfl_alloc(n*2 + 3);
    b[j++] = '"';
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (c=='"' || c=='\\') { b[j++]='\\'; b[j++]=c; }
        else if (c=='\n') { b[j++]='\\'; b[j++]='n'; }
        else if (c=='\t') { b[j++]='\\'; b[j++]='t'; }
        else if (c=='\r') { b[j++]='\\'; b[j++]='r'; }
        else b[j++]=c;
    }
    b[j++]='"'; b[j]=0;
    return b;
}
static char* mfl_http_body(const char* s) { /* bytes after the blank line of an HTTP message */
    const char* b = strstr(s, "\r\n\r\n");
    return mfl_dup(b ? b + 4 : "");
}

/* JSON parsing: a cursor (const char**) walked by recursive-descent helpers */
static void mfl_js_ws(const char** p) { while (**p==' '||**p=='\t'||**p=='\n'||**p=='\r') (*p)++; }
static int64_t mfl_js_int(const char** p) { mfl_js_ws(p); char* e; long long v = strtoll(*p, &e, 10); *p = e; return v; }
static double mfl_js_float(const char** p) { mfl_js_ws(p); char* e; double v = strtod(*p, &e); *p = e; return v; }
static int mfl_js_bool(const char** p) { mfl_js_ws(p); if (**p=='t') { *p += 4; return 1; } if (**p=='f') { *p += 5; return 0; } return 0; }
/* mfl_hex4: the 4 hex digits at p as an int, or -1 if any of them aren't hex
   (a malformed \u escape -- the caller falls back to treating it literally). */
static int mfl_hex4(const char* p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i]; int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return -1;
        v = v * 16 + d;
    }
    return v;
}
/* mfl_utf8_encode: UTF-8 encode one Unicode code point into out, returning the
   byte count written (1-4). */
static size_t mfl_utf8_encode(char* out, int cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F)); out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}
static char* mfl_js_str(const char** p) {
    mfl_js_ws(p);
    if (**p != '"') return mfl_dup("");
    (*p)++;
    char* out = mfl_alloc(strlen(*p) + 1); size_t j = 0;
    while (**p && **p != '"') {
        char c = **p;
        if (c == '\\') {
            (*p)++; char e = **p;
            if (e == 'u') {
                int cp = mfl_hex4(*p + 1);
                if (cp < 0) { out[j++] = 'u'; (*p)++; }
                else {
                    *p += 5; /* past "u" + 4 hex digits */
                    if (cp >= 0xD800 && cp <= 0xDBFF && (*p)[0] == '\\' && (*p)[1] == 'u') {
                        int lo = mfl_hex4(*p + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            *p += 6;
                        }
                    }
                    j += mfl_utf8_encode(out + j, cp);
                }
            }
            else if (e=='n') { out[j++]='\n'; (*p)++; }
            else if (e=='t') { out[j++]='\t'; (*p)++; }
            else if (e=='r') { out[j++]='\r'; (*p)++; }
            else { out[j++] = e; (*p)++; }
        } else { out[j++] = c; (*p)++; }
    }
    if (**p == '"') (*p)++;
    out[j] = 0; return out;
}
static void mfl_js_skip(const char** p) { /* skip one JSON value, including extra object fields */
    mfl_js_ws(p);
    char c = **p;
    if (c == '"') { mfl_js_str(p); return; }
    if (c == '{') { (*p)++; mfl_js_ws(p); if (**p=='}') { (*p)++; return; }
        while (1) { mfl_js_str(p); mfl_js_ws(p); if (**p==':') (*p)++; mfl_js_skip(p); mfl_js_ws(p); if (**p==',') { (*p)++; continue; } break; }
        if (**p=='}') (*p)++; return; }
    if (c == '[') { (*p)++; mfl_js_ws(p); if (**p==']') { (*p)++; return; }
        while (1) { mfl_js_skip(p); mfl_js_ws(p); if (**p==',') { (*p)++; continue; } break; }
        if (**p==']') (*p)++; return; }
    while (**p && **p!=',' && **p!='}' && **p!=']') (*p)++;
}
static int mfl_js_more(const char** p) { mfl_js_ws(p); if (**p==',') { (*p)++; return 1; } return 0; }

/* string operations */
static char* mfl_substr(const char* s, int64_t i, int64_t j) {
    int64_t n = mfl_strlen_cached(s);
    if (i < 0) i = 0; if (j > n) j = n; if (i > j) i = j;
    int64_t len = j - i;
    char* r = mfl_alloc(len + 1); memcpy(r, s + i, len); r[len] = 0;
    return r;
}
static int64_t mfl_index(const char* s, const char* sub) { const char* f = strstr(s, sub); return f ? (int64_t)(f - s) : -1; }
static int mfl_contains(const char* s, const char* sub) { return strstr(s, sub) != NULL; }
static int mfl_has_prefix(const char* s, const char* p) { return strncmp(s, p, strlen(p)) == 0; }
static int mfl_has_suffix(const char* s, const char* p) { size_t ls = strlen(s), lp = strlen(p); return lp <= ls && strcmp(s + ls - lp, p) == 0; }
static char* mfl_charat(const char* s, int64_t i) { int64_t n = strlen(s); if (i < 0 || i >= n) return mfl_dup(""); char* r = mfl_alloc(2); r[0] = s[i]; r[1] = 0; return r; }
static char* mfl_to_upper(const char* s) { size_t n = strlen(s); char* r = mfl_alloc(n + 1); for (size_t i = 0; i < n; i++) r[i] = toupper((unsigned char)s[i]); r[n] = 0; return r; }
static char* mfl_to_lower(const char* s) { size_t n = strlen(s); char* r = mfl_alloc(n + 1); for (size_t i = 0; i < n; i++) r[i] = tolower((unsigned char)s[i]); r[n] = 0; return r; }
static char* mfl_trim(const char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    int64_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n-1])) n--;
    char* r = mfl_alloc(n + 1); memcpy(r, s, n); r[n] = 0; return r;
}
static char* mfl_replace(const char* s, const char* old, const char* neww) {
    size_t lo = strlen(old);
    if (lo == 0) return mfl_dup(s);
    size_t ln = strlen(neww), cnt = 0;
    const char* t = s; while ((t = strstr(t, old))) { cnt++; t += lo; }
    char* r = mfl_alloc(strlen(s) + cnt * (ln > lo ? ln - lo : 0) + 1);
    char* w = r; const char* p = s;
    while (1) { const char* f = strstr(p, old); if (!f) { strcpy(w, p); break; }
        memcpy(w, p, f - p); w += f - p; memcpy(w, neww, ln); w += ln; p = f + lo; }
    return r;
}
static mfl_slice mfl_split(const char* s, const char* sep) {
    mfl_slice out = {0};
    size_t ls = strlen(sep);
    if (ls == 0) { int64_t n = strlen(s);
        for (int64_t i = 0; i < n; i++) { char* c = mfl_alloc(2); c[0] = s[i]; c[1] = 0; out = mfl_append(out, &c, sizeof(char*)); }
        return out; }
    const char* p = s;
    while (1) { const char* f = strstr(p, sep);
        if (!f) { char* piece = mfl_dup(p); out = mfl_append(out, &piece, sizeof(char*)); break; }
        size_t len = f - p; char* piece = mfl_alloc(len + 1); memcpy(piece, p, len); piece[len] = 0;
        out = mfl_append(out, &piece, sizeof(char*)); p = f + ls; }
    return out;
}
static char* mfl_join(mfl_slice xs, const char* sep) {
    if (xs.len == 0) return mfl_dup("");
    char** parts = (char**)xs.data;
    size_t ls = strlen(sep), total = 0;
    for (int64_t i = 0; i < xs.len; i++) total += strlen(mfl_s(parts[i]));
    total += ls * (size_t)(xs.len - 1);
    char* r = mfl_alloc(total + 1);
    char* w = r;
    for (int64_t i = 0; i < xs.len; i++) {
        if (i > 0) { memcpy(w, sep, ls); w += ls; }
        const char* p = mfl_s(parts[i]);
        size_t lp = strlen(p);
        memcpy(w, p, lp); w += lp;
    }
    *w = 0;
    return r;
}

/* read one line from stdin (without the trailing newline); "" at EOF */
static char* mfl_input(void) {
    size_t cap = 128, len = 0;
    char* buf = mfl_alloc(cap);
    int c;
    while ((c = getchar()) != EOF && c != '\n') {
        if (len + 1 >= cap) { cap *= 2; buf = mfl_realloc(buf, cap); }
        buf[len++] = (char)c;
    }
    buf[len] = 0;
    return buf;
}
/* read all of stdin verbatim until EOF (no line splitting). Exact for text;
   an embedded NUL would truncate the string view (machin strings are C strings). */
static char* mfl_read_stdin(void) {
    size_t cap = 65536, len = 0;
    char* buf = (char*)malloc(cap);
    size_t n;
    while ((n = fread(buf + len, 1, cap - len - 1, stdin)) > 0) {
        len += n;
        if (len + 1 >= cap) { cap *= 2; buf = (char*)realloc(buf, cap); }
    }
    char* r = mfl_dup_arena(buf, len);
    free(buf);
    return r;
}

/* command-line arguments, environment, and wall-clock time */
static int mfl_argc = 0;
static char** mfl_argv = NULL;
static mfl_slice mfl_args(void) {
    mfl_slice s = { mfl_argc ? mfl_alloc(mfl_argc * sizeof(char*)) : NULL, mfl_argc, mfl_argc };
    for (int i = 0; i < mfl_argc; i++) ((char**)s.data)[i] = mfl_argv[i];
    return s;
}
static char* mfl_env(const char* k) { char* v = getenv(k); return v ? v : ""; }
static int64_t mfl_now(void) { return (int64_t)time(NULL); }
static int64_t mfl_now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000; }
/* decompose a Unix timestamp (local time) into
   [year, month(1-12), day(1-31), hour, minute, second, weekday(0=Sun), yearday(1-366)] */
static mfl_slice mfl_time_fields(int64_t unix) {
    time_t t = (time_t)unix;
    struct tm tmv;
    localtime_r(&t, &tmv);
    int64_t v[8] = { tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
        tmv.tm_hour, tmv.tm_min, tmv.tm_sec, tmv.tm_wday, tmv.tm_yday + 1 };
    mfl_slice s = { mfl_alloc(8 * sizeof(int64_t)), 8, 8 };
    memcpy(s.data, v, sizeof(v));
    return s;
}
/* format a Unix timestamp (local time) with a strftime(3) pattern:
   %Y %m %d %H %M %S %A %a %B %b %p %j %z %Z %F %T ... ("" if it overflows) */
static char* mfl_time_format(int64_t unix, const char* fmt) {
    time_t t = (time_t)unix;
    struct tm tmv;
    localtime_r(&t, &tmv);
    char buf[512];
    size_t n = strftime(buf, sizeof(buf), fmt, &tmv);
    char* out = mfl_alloc(n + 1);
    memcpy(out, buf, n);
    out[n] = 0;
    return out;
}
/* construct a Unix timestamp from calendar fields (local time, the inverse of
   time_fields): mktime normalizes out-of-range fields (e.g. day 32 rolls over)
   and resolves DST via tm_isdst=-1. */
static int64_t mfl_time_make(int64_t y, int64_t mo, int64_t d, int64_t h, int64_t mi, int64_t s) {
    struct tm tmv;
    memset(&tmv, 0, sizeof(tmv));
    tmv.tm_year = (int)(y - 1900);
    tmv.tm_mon = (int)(mo - 1);
    tmv.tm_mday = (int)d;
    tmv.tm_hour = (int)h;
    tmv.tm_min = (int)mi;
    tmv.tm_sec = (int)s;
    tmv.tm_isdst = -1;
    return (int64_t)mktime(&tmv);
}
/* like time_format but in UTC (gmtime): the form .ics / RFC-3339 timestamps want. */
static char* mfl_time_format_utc(int64_t unix, const char* fmt) {
    time_t t = (time_t)unix;
    struct tm tmv;
    gmtime_r(&t, &tmv);
    char buf[512];
    size_t n = strftime(buf, sizeof(buf), fmt, &tmv);
    char* out = mfl_alloc(n + 1);
    memcpy(out, buf, n);
    out[n] = 0;
    return out;
}
static int64_t mfl_parse_int(const char* s) { return (int64_t)strtoll(s, NULL, 10); }
static double mfl_parse_float(const char* s) { return strtod(s, NULL); }

/* file system: read/write whole files, list a directory, make a directory */

/* fopen(dir, "rb") SUCCEEDS on Linux (opening a directory read-only is legal
   at the syscall level), but ftell() on it returns LONG_MAX (not -1) rather
   than failing — so the usual "if (n < 0) n = 0" guard never catches it, and
   the caller ends up trying to alloc ~9.2 exabytes. list_dir()'s entries can
   be subdirectories, so read_file/read_file_bytes on one of those is a real,
   easy-to-hit path, not a contrived one — check with stat() first. */
static int mfl_is_dir(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}
static char* mfl_read_file(const char* path) {
    if (mfl_is_dir(path)) return mfl_dup("");
    FILE* f = fopen(path, "rb");
    if (!f) return mfl_dup("");
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < 0) n = 0;
    char* buf = mfl_alloc((size_t)n + 1);
    size_t r = fread(buf, 1, (size_t)n, f); buf[r] = 0;
    fclose(f); return buf;
}
static int64_t mfl_write_file(const char* path, const char* content) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    size_t len = strlen(content);
    size_t w = fwrite(content, 1, len, f);
    fclose(f); return (int64_t)w;
}
/* write raw bytes to a file, NUL-safe (length-driven) — for binary uploads/assets. */
static int64_t mfl_write_file_bytes(const char* path, mfl_bytes b) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    size_t w = b.len ? fwrite(b.data, 1, (size_t)b.len, f) : 0;
    fclose(f); return (int64_t)w;
}
/* delete a file (0 on success, -1 on error) — e.g. removing a stored upload. */
static int64_t mfl_remove_file(const char* path) { return remove(path) == 0 ? 0 : -1; }
/* read a file's raw bytes (NUL-safe, unlike read_file which returns a C string).
   Empty bytes if the file can't be opened. */
static mfl_bytes mfl_read_file_bytes(const char* path) {
    mfl_bytes b; b.len = 0; b.data = (uint8_t*)mfl_alloc(1);
    if (mfl_is_dir(path)) return b;
    FILE* f = fopen(path, "rb");
    if (!f) return b;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < 0) n = 0;
    b.data = (uint8_t*)mfl_alloc((size_t)n ? (size_t)n : 1);
    b.len = (int64_t)fread(b.data, 1, (size_t)n, f);
    fclose(f);
    return b;
}
static mfl_slice mfl_list_dir(const char* path) {
    mfl_slice out = {0};
    DIR* d = opendir(path);
    if (!d) return out;
    struct dirent* e;
    while ((e = readdir(d))) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
        char* name = mfl_dup(e->d_name);
        out = mfl_append(out, &name, sizeof(char*));
    }
    closedir(d); return out;
}
static int64_t mfl_mkdir(const char* path) {
    int r = mkdir(path, 0755);
    if (r < 0 && errno == EEXIST) return 0;
    return r;
}
#ifndef __wasm__
/* run a shell command, return its exit code (-1 if it could not be launched). For
   process orchestration — e.g. spawning a detached daemon with a trailing "&". */
static int64_t mfl_system(const char* cmd) {
    int r = system(cmd);
    if (r == -1) return -1;
    return (int64_t)WEXITSTATUS(r);
}
/* run a shell command and capture its output: returns (exit_code, stdout, stderr).
   The command runs via /bin/sh in a subshell with stdout/stderr redirected to temp
   files (so there is no pipe-buffer deadlock), then both are read back. Captured
   text is NUL-terminated — a command producing binary output should redirect it to
   a file itself (e.g. mongodump --archive piped to gzip > out.gz). */
typedef struct { int64_t code; char* out; char* err; } mfl_exec_result;
static mfl_exec_result mfl_exec(const char* cmd) {
    mfl_exec_result R; R.code = -1; R.out = mfl_dup(""); R.err = mfl_dup("");
    char op[] = "/tmp/mfl-exec-XXXXXX", ep[] = "/tmp/mfl-exec-XXXXXX";
    int fo = mkstemp(op); if (fo < 0) return R; close(fo);
    int fe = mkstemp(ep); if (fe < 0) { unlink(op); return R; } close(fe);
    size_t n = strlen(cmd) + strlen(op) + strlen(ep) + 16;
    char* full = (char*)malloc(n);
    if (full) {
        snprintf(full, n, "( %s ) >%s 2>%s", cmd, op, ep);
        int r = system(full);
        R.code = (r == -1) ? -1 : (int64_t)WEXITSTATUS(r);
        R.out = mfl_read_file(op);
        R.err = mfl_read_file(ep);
        free(full);
    }
    unlink(op); unlink(ep);
    return R;
}
#endif

/* copy n bytes into a fresh NUL-terminated arena string */
static char* mfl_dup_arena(const char* s, size_t n) {
    char* r = (char*)mfl_alloc(n + 1);
    if (n) memcpy(r, s, n);
    r[n] = 0;
    return r;
}

/* ---- JSON path query (json_get) ----
   A non-allocating scanner: it walks the document following a jq-style path
   (.key, [index], chained) and returns the located value's raw JSON text. No
   tree is built — values not on the path are skipped, respecting nesting and
   string escapes (unlike naive substring search). */
typedef struct { char* value; char* err; } mfl_json_result;

static const char* mfl_jq_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}
static const char* mfl_jq_str(const char* p) { /* p at opening quote -> past closing */
    if (*p != '"') return NULL;
    p++;
    while (*p) {
        if (*p == '\\') { p++; if (!*p) return NULL; p++; continue; }
        if (*p == '"') return p + 1;
        p++;
    }
    return NULL;
}
static const char* mfl_jq_val(const char* p) { /* skip one value -> just past it */
    p = mfl_jq_ws(p);
    if (*p == '"') return mfl_jq_str(p);
    if (*p == '{' || *p == '[') {
        char open = *p, close = open == '{' ? '}' : ']';
        p = mfl_jq_ws(p + 1);
        if (*p == close) return p + 1;
        for (;;) {
            if (open == '{') {
                p = mfl_jq_ws(p);
                p = mfl_jq_str(p); if (!p) return NULL;
                p = mfl_jq_ws(p);
                if (*p != ':') return NULL;
                p++;
            }
            p = mfl_jq_val(p); if (!p) return NULL;
            p = mfl_jq_ws(p);
            if (*p == ',') { p++; continue; }
            if (*p == close) return p + 1;
            return NULL;
        }
    }
    if (*p == ',' || *p == '}' || *p == ']' || *p == 0) return NULL;
    while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
    return p;
}
static const char* mfl_jq_member(const char* p, const char* key, size_t keylen) {
    p = mfl_jq_ws(p);
    if (*p != '{') return NULL;
    p = mfl_jq_ws(p + 1);
    if (*p == '}') return NULL;
    for (;;) {
        p = mfl_jq_ws(p);
        if (*p != '"') return NULL;
        const char* ks = p + 1;
        const char* ke = mfl_jq_str(p); if (!ke) return NULL;
        size_t klen = (size_t)((ke - 1) - ks);
        p = mfl_jq_ws(ke);
        if (*p != ':') return NULL;
        const char* vs = mfl_jq_ws(p + 1);
        if (klen == keylen && memcmp(ks, key, keylen) == 0) return vs;
        const char* ve = mfl_jq_val(vs); if (!ve) return NULL;
        p = mfl_jq_ws(ve);
        if (*p == ',') { p++; continue; }
        return NULL;
    }
}
static const char* mfl_jq_elem(const char* p, long n) {
    p = mfl_jq_ws(p);
    if (*p != '[') return NULL;
    p = mfl_jq_ws(p + 1);
    if (*p == ']') return NULL;
    long i = 0;
    for (;;) {
        const char* vs = mfl_jq_ws(p);
        if (i == n) return vs;
        const char* ve = mfl_jq_val(vs); if (!ve) return NULL;
        p = mfl_jq_ws(ve);
        if (*p == ',') { p++; i++; continue; }
        return NULL;
    }
}
static mfl_json_result mfl_json_get(const char* json, const char* path) {
    mfl_json_result R;
    R.value = mfl_dup_arena("", 0);
    R.err = mfl_dup_arena("", 0);
    const char* cur = mfl_jq_ws(json);
    const char* p = path;
    while (*p) {
        if (*p == '.') {
            p++;
            const char* ks = p;
            while (*p && *p != '.' && *p != '[') p++;
            size_t klen = (size_t)(p - ks);
            if (klen == 0) continue;
            cur = mfl_jq_member(cur, ks, klen);
            if (!cur) { R.err = mfl_dup_arena("notfound", 8); return R; }
        } else if (*p == '[') {
            p++;
            char* endp;
            long idx = strtol(p, &endp, 10);
            if (endp == p || *endp != ']') { R.err = mfl_dup_arena("path", 4); return R; }
            p = endp + 1;
            cur = mfl_jq_elem(cur, idx);
            if (!cur) { R.err = mfl_dup_arena("notfound", 8); return R; }
        } else {
            R.err = mfl_dup_arena("path", 4);
            return R;
        }
    }
    const char* end = mfl_jq_val(cur);
    if (!end) { R.err = mfl_dup_arena("parse", 5); return R; }
    R.value = mfl_dup_arena(cur, (size_t)(end - cur));
    return R;
}

/* networking: the low-level shape of Go's net package */
static int64_t mfl_listen(int64_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in a; memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr*)&a, sizeof(a)) < 0) { perror("bind"); exit(1); }
    if (listen(fd, 64) < 0) { perror("listen"); exit(1); }
    return fd;
}
static int64_t mfl_accept(int64_t fd) { return accept((int)fd, NULL, NULL); }
/* peer_addr: the remote IP of a connected socket (getpeername), "" on error — the real
   client IP when not behind a proxy (behind one, prefer X-Forwarded-For). */
static const char* mfl_peer_addr(int64_t fd) {
    struct sockaddr_storage ss; socklen_t sl = sizeof(ss);
    if (getpeername((int)fd, (struct sockaddr*)&ss, &sl) != 0) return "";
    char host[64] = {0};
    if (getnameinfo((struct sockaddr*)&ss, sl, host, sizeof(host), NULL, 0, NI_NUMERICHOST) != 0) return "";
    char* r = (char*)mfl_alloc(strlen(host) + 1); strcpy(r, host); return r;
}
/* socket_timeout: cap blocking recv/send on a socket to ms milliseconds (0 = none), so a
   slow client can't park a connection forever. Returns 0 on success, -1 on error. */
static int64_t mfl_socket_timeout(int64_t fd, int64_t ms) {
    struct timeval tv; tv.tv_sec = ms / 1000; tv.tv_usec = (ms % 1000) * 1000;
    int a = setsockopt((int)fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    int b = setsockopt((int)fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    return (a == 0 && b == 0) ? 0 : -1;
}
/* dial: connect a TCP socket to host:port, returning an fd (-1 on failure).
   The fd is used with the same read/write/close as an accepted connection. */
static int64_t mfl_dial(const char* host, int64_t port) {
    struct addrinfo hints, *res, *rp;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    char ps[16]; snprintf(ps, sizeof(ps), "%lld", (long long)port);
    if (getaddrinfo(host, ps, &hints, &res) != 0) return -1;
    int fd = -1;
    for (rp = res; rp; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;
        close(fd); fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}
static char* mfl_read(int64_t fd) {
    char* buf = mfl_alloc(65536);
    ssize_t n = read((int)fd, buf, 65535);
    if (n < 0) n = 0;
    buf[n] = 0;
    return buf;
}
// mfl_read_bytes is the NUL-safe socket read: returns the raw bytes of one chunk
// (empty at EOF / on error). For binary wire protocols where read() (a C string)
// would truncate at the first 0 byte.
static mfl_bytes mfl_read_bytes(int64_t fd) {
    mfl_bytes b; b.data = (uint8_t*)mfl_alloc(65536);
    ssize_t n = read((int)fd, b.data, 65536);
    if (n < 0) n = 0;
    b.len = (int64_t)n;
    return b;
}
static int64_t mfl_write(int64_t fd, const char* s) { return (int64_t)write((int)fd, s, strlen(s)); }
/* write the exact bytes of a buffer to an fd (NUL-safe, for binary responses). */
static int64_t mfl_write_bytes(int64_t fd, mfl_bytes b) {
    size_t off = 0;
    while (off < (size_t)b.len) {
        ssize_t w = write((int)fd, b.data + off, (size_t)b.len - off);
        if (w <= 0) break;
        off += (size_t)w;
    }
    return (int64_t)off;
}
static void mfl_close(int64_t fd) { close((int)fd); }

/* terminal raw mode + non-blocking single-key read (for TUIs and games).
   raw_mode(1) puts the tty in cbreak + no-echo with VMIN=0/VTIME=0 so reads
   never block; raw_mode(0) restores the saved settings. */
static struct termios mfl_tty_saved;
static int mfl_tty_raw = 0;
static int64_t mfl_raw_mode(int64_t on) {
    if (on) {
        if (mfl_tty_raw) return 0;
        struct termios t;
        if (tcgetattr(STDIN_FILENO, &t) != 0) return -1;
        mfl_tty_saved = t;
        t.c_lflag &= ~(ICANON | ECHO);
        t.c_cc[VMIN] = 0;
        t.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &t) != 0) return -1;
        mfl_tty_raw = 1;
    } else {
        if (!mfl_tty_raw) return 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &mfl_tty_saved);
        mfl_tty_raw = 0;
    }
    return 0;
}
/* non-blocking read of one key; a 1-char string, or "" if nothing is waiting.
   In raw mode VMIN=0 already makes read() return immediately; otherwise poll
   with select() so we never block. */
static char* mfl_read_key(void) {
    char* buf = mfl_alloc(2);
    buf[0] = 0; buf[1] = 0;
    unsigned char c = 0;
    if (mfl_tty_raw) {
        if (read(STDIN_FILENO, &c, 1) == 1) buf[0] = (char)c;
    } else {
        fd_set fds; FD_ZERO(&fds); FD_SET(STDIN_FILENO, &fds);
        struct timeval tv; tv.tv_sec = 0; tv.tv_usec = 0;
        if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0) {
            if (read(STDIN_FILENO, &c, 1) == 1) buf[0] = (char)c;
        }
    }
    return buf;
}

#include <math.h>
static double mfl_math_sin(double x){return sin(x);}
static double mfl_math_cos(double x){return cos(x);}
static double mfl_math_tan(double x){return tan(x);}
static double mfl_math_asin(double x){return asin(x);}
static double mfl_math_acos(double x){return acos(x);}
static double mfl_math_atan(double x){return atan(x);}
static double mfl_math_exp(double x){return exp(x);}
static double mfl_math_log(double x){return log(x);}
static double mfl_math_log2(double x){return log2(x);}
static double mfl_math_log10(double x){return log10(x);}
static double mfl_math_sqrt(double x){return sqrt(x);}
static double mfl_math_cbrt(double x){return cbrt(x);}
static double mfl_math_floor(double x){return floor(x);}
static double mfl_math_ceil(double x){return ceil(x);}
static double mfl_math_round(double x){return round(x);}
static double mfl_math_trunc(double x){return trunc(x);}
static double mfl_math_fabs(double x){return fabs(x);}
static double mfl_math_pow(double a,double b){return pow(a,b);}
static double mfl_math_atan2(double a,double b){return atan2(a,b);}
static double mfl_math_fmod(double a,double b){return fmod(a,b);}
static double mfl_math_hypot(double a,double b){return hypot(a,b);}
static double mfl_math_pi(void){return M_PI;}

typedef struct { int64_t r0; int64_t r1; } mfl_cp_at_0_ret;
static int64_t mfl_g_dim;
static int64_t mfl_g_hidden_dim;
static int64_t mfl_g_n_layers;
static int64_t mfl_g_n_heads;
static int64_t mfl_g_n_kv_heads;
static int64_t mfl_g_vocab_size;
static int64_t mfl_g_seq_len;
static int64_t mfl_g_head_size;
static int64_t mfl_g_kv_dim;
static int64_t mfl_g_gs;
static int64_t mfl_g_shared_cls;
static int64_t mfl_g_quant4;
static int64_t mfl_g_wbuf;
static int64_t mfl_g_w_rms_att;
static int64_t mfl_g_w_rms_ffn;
static int64_t mfl_g_w_rms_final;
static int64_t mfl_g_w_tok;
static int64_t mfl_g_wq_b;
static int64_t mfl_g_wk_b;
static int64_t mfl_g_wv_b;
static int64_t mfl_g_wo_b;
static int64_t mfl_g_w1_b;
static int64_t mfl_g_w2_b;
static int64_t mfl_g_w3_b;
static int64_t mfl_g_wcls_b;
static int64_t mfl_g_s_x;
static int64_t mfl_g_s_xb;
static int64_t mfl_g_s_xb2;
static int64_t mfl_g_s_hb;
static int64_t mfl_g_s_hb2;
static int64_t mfl_g_s_q;
static int64_t mfl_g_s_att;
static int64_t mfl_g_s_logits;
static int64_t mfl_g_s_key_cache;
static int64_t mfl_g_s_val_cache;
static int64_t mfl_g_kv_off;
static int64_t mfl_g_slotstride;
static int64_t mfl_g_cb_slots;
static int64_t mfl_g_cb_active;
static int64_t mfl_g_cb_pos;
static int64_t mfl_g_cb_curtok;
static int64_t mfl_g_cb_kt;
static int64_t mfl_g_cb_vt;
static int64_t mfl_g_cb_next;
static int64_t mfl_g_s_xq;
static int64_t mfl_g_s_xq_s;
static int64_t mfl_g_s_hq;
static int64_t mfl_g_s_hq_s;
static mfl_slice mfl_g_vocab;
static mfl_map* mfl_g_vocab_idx;
static int64_t mfl_g_tok_scores;
static int64_t mfl_g_pb_tile;
static int64_t mfl_g_pb_enabled;
static int64_t mfl_g_prof_qkv;
static int64_t mfl_g_prof_attn;
static int64_t mfl_g_prof_o;
static int64_t mfl_g_prof_ffn;
static int64_t mfl_g_pb_B;
static int64_t mfl_g_pb_tstart;
static int64_t mfl_g_pb_x;
static int64_t mfl_g_pb_xb;
static int64_t mfl_g_pb_xb2;
static int64_t mfl_g_pb_q;
static int64_t mfl_g_pb_hb;
static int64_t mfl_g_pb_hb2;
static int64_t mfl_g_pb_xq;
static int64_t mfl_g_pb_xqs;
static int64_t mfl_g_pb_hq;
static int64_t mfl_g_pb_hqs;
static int64_t mfl_g_n_workers;
static mfl_chan* mfl_g_jobs;
static mfl_chan* mfl_g_done;
static int64_t mfl_g_pending;
static int64_t mfl_g_rng_state;
static int64_t mfl_g_cache_buf;
static int64_t mfl_g_cache_len;
static int64_t mfl_g_spec_enabled;
static int64_t mfl_g_sparse_probe;
static int64_t mfl_g_exit_layer;
static int64_t mfl_g_sp_below;
static int64_t mfl_g_sp_total;
static double mfl_g_sp_massk;
static int64_t mfl_g_sp_massn;
static int64_t mfl_g_spec_k;
static int64_t mfl_g_spec_maxb;
static int64_t mfl_g_spec_logits;
static int64_t mfl_g_spec_argmax;
static int64_t mfl_g_spec_tbuf;
static int64_t mfl_g_spec_steps;
static int64_t mfl_g_spec_tokens;
static int64_t mfl_g_gen_prefill_reused;
static int64_t mfl_g_rope_freqs;
static double mfl_g_rope_theta;
static int64_t mfl_g_l3_scaling;
static double mfl_g_l3_factor;
static double mfl_g_l3_low;
static double mfl_g_l3_high;
static int64_t mfl_g_l3_orig;
static int64_t mfl_g_tok_l3;
static mfl_slice mfl_g_vocab_l3;
static mfl_map* mfl_g_l3_idx;

void mfl_main(void);
void mfl_load_model_0(char* v_path);
int64_t mfl_qblock_0(int64_t v_size);
int64_t mfl_wsz_0(int64_t v_size);
double mfl_rope_freq_0(int64_t v_hd);
void mfl_alloc_state_0(void);
void mfl_alloc_prefill_0(void);
void mfl_alloc_spec_0(void);
void mfl_load_tokenizer_auto_0(void);
void mfl_load_tokenizer_l3_0(char* v_path);
int64_t mfl_read_i32_0(mfl_bytes v_b, int64_t v_off);
void mfl_load_tokenizer_0(char* v_path);
int64_t mfl_bytes_to_buf_0(mfl_bytes v_b);
void mfl_worker_0(mfl_chan* v_jc, mfl_chan* v_dc);
void mfl_do_job_0(int64_t v_j, int64_t v_sc, int64_t v_att);
void mfl_mmj_0(int64_t v_xout, int64_t v_xq, int64_t v_xs, int64_t v_tb, int64_t v_tsize, int64_t v_n, int64_t v_lo, int64_t v_hi, int64_t v_sc);
void mfl_matmul_q4_0(int64_t v_xout, int64_t v_xq, int64_t v_xs, int64_t v_wq, int64_t v_ws, int64_t v_n, int64_t v_lo, int64_t v_hi, int64_t v_sc);
void mfl_matmul_q_0(int64_t v_xout, int64_t v_xq, int64_t v_xs, int64_t v_wq, int64_t v_ws, int64_t v_n, int64_t v_lo, int64_t v_hi);
void mfl_attn_head_0(int64_t v_h, int64_t v_l, int64_t v_pos);
void mfl_softmax_0(int64_t v_x, int64_t v_size);
void mfl_matmul_q_batch_0(int64_t v_ob, int64_t v_ostride, int64_t v_xqb, int64_t v_xsb, int64_t v_wq, int64_t v_ws, int64_t v_n, int64_t v_lo, int64_t v_hi, int64_t v_B);
void mfl_attn_head_at_0(int64_t v_h, int64_t v_l, int64_t v_pos, int64_t v_qbase, int64_t v_obase, int64_t v_att);
void mfl_rmsnorm_0(int64_t v_dst, int64_t v_x, int64_t v_w, int64_t v_size);
void mfl_quantize_act_0(int64_t v_xq, int64_t v_xs, int64_t v_x, int64_t v_n);
void mfl_rope_at_0(int64_t v_l, int64_t v_pos, int64_t v_qbase);
void mfl_attn_head_cb_0(int64_t v_jj, int64_t v_hh, int64_t v_l, int64_t v_att);
void mfl_copy_f32_0(int64_t v_dst, int64_t v_src, int64_t v_n);
void mfl_rope_qk_0(int64_t v_pos, int64_t v_qbase, int64_t v_kdst);
mfl_slice mfl_encode_text_0(char* v_text);
mfl_slice mfl_l3_encode_0(char* v_text);
mfl_slice mfl_l3_pretok_0(char* v_text);
int64_t mfl_l3_contraction_len_0(mfl_bytes v_b, int64_t v_i, int64_t v_n);
mfl_cp_at_0_ret mfl_cp_at_0(mfl_bytes v_b, int64_t v_i);
int mfl_is_letter_0(int64_t v_cp);
int mfl_is_digit_cp_0(int64_t v_cp);
int mfl_is_space_cp_0(int64_t v_cp);
mfl_slice mfl_l3_bpe_piece_0(char* v_piece);
mfl_slice mfl_bpe_encode_0(char* v_text);
mfl_slice mfl_bpe_encode_opts_0(char* v_text, int64_t v_add_bos, int64_t v_add_prefix);
void mfl_cb_prefill_0(int64_t v_sl, mfl_slice v_toks, int64_t v_np);
void mfl_forward_0(int64_t v_token, int64_t v_pos);
void mfl_embed_token_0(int64_t v_token);
void mfl_dispatch_rows_0(int64_t v_t, int64_t v_l, int64_t v_pos, int64_t v_d);
void mfl_barrier_0(void);
void mfl_rope_0(int64_t v_l, int64_t v_pos);
int64_t mfl_argmax_logits_0(void);
void mfl_cb_step_0(int64_t v_M);
void mfl_embed_to_0(int64_t v_token, int64_t v_dst);
char* mfl_piece_text_0(int64_t v_prev, int64_t v_token);
char* mfl_decode_piece_0(int64_t v_prev, int64_t v_token);

struct mfl_go_0 { char _; mfl_chan* a0; mfl_chan* a1; };
static void* mfl_go_run_0(void* p) { mfl_arena _a = {0}; mfl_arena_cur = &_a; struct mfl_go_0* s = (struct mfl_go_0*)p;
    mfl_worker_0(s->a0, s->a1); free(s); mfl_arena_free(&_a); return NULL; }
void mfl_main(void) {
    mfl_slice v_a = {0};
    char* v_model = "";
    int64_t v_N = 0;
    int64_t v_steps = 0;
    char* v_prompt = "";
    int64_t v_nw = 0;
    int64_t v_w = 0;
    mfl_slice v_toks = {0};
    int64_t v_np = 0;
    int64_t v_s = 0;
    mfl_slice v_out0 = {0};
    int64_t v_allmatch = 0;
    int64_t v_t0 = 0;
    int64_t v_i = 0;
    int64_t v_tok0 = 0;
    int64_t v_t1 = 0;
    int64_t v_total = 0;
    char* v_txt = "";
    int64_t v_prev = 0;
    int64_t v_k = 0;
    char* v_amv = "";
    v_a = mfl_args();
    v_model = "models/llama32-1b-q80.bin";
    v_N = 4;
    v_steps = 32;
    v_prompt = "The capital of France is";
    if (({ int64_t _sq0_0 = ((v_a).len); int64_t _sq0_1 = 1; (_sq0_0 > _sq0_1); })) {
        v_model = ((char**)(v_a).data)[1];
    }
    if (({ int64_t _sq1_0 = ((v_a).len); int64_t _sq1_1 = 2; (_sq1_0 > _sq1_1); })) {
        v_N = mfl_parse_int(((char**)(v_a).data)[2]);
    }
    if (({ int64_t _sq2_0 = ((v_a).len); int64_t _sq2_1 = 3; (_sq2_0 > _sq2_1); })) {
        v_steps = mfl_parse_int(((char**)(v_a).data)[3]);
    }
    if (({ int64_t _sq3_0 = ((v_a).len); int64_t _sq3_1 = 4; (_sq3_0 > _sq3_1); })) {
        v_prompt = ((char**)(v_a).data)[4];
    }
    v_nw = mfl_parse_int(mfl_env("COLIBRI_THREADS"));
    if ((v_nw > 0)) {
        mfl_g_n_workers = v_nw;
    } else {
        mfl_g_n_workers = 6;
    }
    mfl_load_model_0(v_model);
    mfl_alloc_state_0();
    mfl_load_tokenizer_auto_0();
    v_w = 0;
    while ((v_w < mfl_g_n_workers)) {
        {
        struct mfl_go_0* s = malloc(sizeof(*s));
        s->a0 = (mfl_g_jobs);
        s->a1 = (mfl_g_done);
        pthread_t t; pthread_create(&t, NULL, mfl_go_run_0, s); pthread_detach(t);
    }
        v_w = (v_w + 1);
    }
    v_toks = mfl_encode_text_0(v_prompt);
    v_np = ((v_toks).len);
    v_s = 0;
    while ((v_s < v_N)) {
        mfl_cb_prefill_0(v_s, v_toks, v_np);
        v_s = (v_s + 1);
    }
    v_s = 0;
    while ((v_s < v_N)) {
        mfl_poke_i32(mfl_g_cb_active, (v_s * 4), v_s);
        v_s = (v_s + 1);
    }
    v_out0 = (mfl_slice){0};
    v_out0 = ({ mfl_slice _sq4_0 = v_out0; int64_t _sq4_1 = mfl_peek_i32(mfl_g_cb_curtok, 0); mfl_append(_sq4_0, &((int64_t[1]){_sq4_1})[0], sizeof(int64_t)); });
    v_allmatch = 1;
    v_t0 = mfl_now_ms();
    v_i = 0;
    while ((v_i < v_steps)) {
        mfl_cb_step_0(v_N);
        v_tok0 = mfl_peek_i32(mfl_g_cb_curtok, 0);
        v_out0 = mfl_append(v_out0, &((int64_t[1]){v_tok0})[0], sizeof(int64_t));
        v_s = 1;
        while ((v_s < v_N)) {
            if (({ int64_t _sq5_0 = mfl_peek_i32(mfl_g_cb_curtok, (v_s * 4)); int64_t _sq5_1 = v_tok0; (_sq5_0 != _sq5_1); })) {
                v_allmatch = 0;
            }
            v_s = (v_s + 1);
        }
        v_i = (v_i + 1);
    }
    v_t1 = mfl_now_ms();
    v_total = (v_N * (v_steps + 1));
    v_txt = "";
    v_prev = ((int64_t*)(v_toks).data)[(v_np - 1)];
    v_k = 0;
    while (({ int64_t _sq6_0 = v_k; int64_t _sq6_1 = ((v_out0).len); (_sq6_0 < _sq6_1); })) {
        v_txt = ({ char* _sq7_0 = v_txt; char* _sq7_1 = mfl_piece_text_0(v_prev, ((int64_t*)(v_out0).data)[v_k]); mfl_cat(_sq7_0, _sq7_1); });
        v_prev = ((int64_t*)(v_out0).data)[v_k];
        v_k = (v_k + 1);
    }
    { const char* _s = (mfl_cat("slot0: ", v_txt)); fputs(_s ? _s : "", stdout); }
    fputs("\n", stdout);
    v_amv = "YES";
    if ((v_allmatch == 0)) {
        v_amv = "NO";
    }
    { const char* _s = (({ char* _sq20_0 = ({ char* _sq19_0 = ({ char* _sq17_0 = ({ char* _sq16_0 = ({ char* _sq15_0 = ({ char* _sq14_0 = ({ char* _sq13_0 = ({ char* _sq12_0 = ({ char* _sq11_0 = ({ char* _sq10_0 = ({ char* _sq9_0 = ({ char* _sq8_0 = "N="; char* _sq8_1 = mfl_str_i(v_N); mfl_cat(_sq8_0, _sq8_1); }); char* _sq9_1 = " steps="; mfl_cat(_sq9_0, _sq9_1); }); char* _sq10_1 = mfl_str_i(v_steps); mfl_cat(_sq10_0, _sq10_1); }); char* _sq11_1 = " all-slots-identical="; mfl_cat(_sq11_0, _sq11_1); }); char* _sq12_1 = v_amv; mfl_cat(_sq12_0, _sq12_1); }); char* _sq13_1 = " | total="; mfl_cat(_sq13_0, _sq13_1); }); char* _sq14_1 = mfl_str_i(v_total); mfl_cat(_sq14_0, _sq14_1); }); char* _sq15_1 = " tok  time="; mfl_cat(_sq15_0, _sq15_1); }); char* _sq16_1 = mfl_str_i((v_t1 - v_t0)); mfl_cat(_sq16_0, _sq16_1); }); char* _sq17_1 = "ms  aggregate="; mfl_cat(_sq17_0, _sq17_1); }); char* _sq19_1 = mfl_str_d(({ double _sq18_0 = ((double)((v_total * 1000))); double _sq18_1 = ((double)((v_t1 - v_t0))); (_sq18_0 / _sq18_1); })); mfl_cat(_sq19_0, _sq19_1); }); char* _sq20_1 = " tok/s"; mfl_cat(_sq20_0, _sq20_1); })); fputs(_s ? _s : "", stdout); }
    fputs("\n", stdout);
}

void mfl_load_model_0(char* v_path) {
    int64_t v_mb = 0;
    int64_t v_m = 0;
    int64_t v_v4 = 0;
    int64_t v_o = 0;
    int64_t v_hd = 0;
    {
        mfl_mmap_result _t21 = mfl_mmap_file(v_path);
        v_mb = _t21.ptr;
    }
    mfl_g_wbuf = v_mb;
    if ((mfl_g_wbuf == 0)) {
        { const char* _s = (mfl_cat("cannot mmap model: ", v_path)); fputs(_s ? _s : "", stdout); }
        fputs("\n", stdout);
        mfl_exit(1);
    }
    v_m = mfl_peek_i32(mfl_g_wbuf, 0);
    v_v4 = 0;
    if ((v_m == 1634415666)) {
        mfl_g_quant4 = 0;
    } else {
        if ((v_m == 1634415667)) {
            mfl_g_quant4 = 1;
        } else {
            if ((v_m == 1634415668)) {
                mfl_g_quant4 = 0;
                v_v4 = 1;
            } else {
                if ((v_m == 1634415669)) {
                    mfl_g_quant4 = 1;
                    v_v4 = 1;
                } else {
                    { const char* _s = ("bad magic (want ak42/ak43/ak44/ak45)"); fputs(_s ? _s : "", stdout); }
                    fputs("\n", stdout);
                    mfl_exit(1);
                }
            }
        }
    }
    mfl_g_dim = mfl_peek_i32(mfl_g_wbuf, 8);
    mfl_g_hidden_dim = mfl_peek_i32(mfl_g_wbuf, 12);
    mfl_g_n_layers = mfl_peek_i32(mfl_g_wbuf, 16);
    mfl_g_n_heads = mfl_peek_i32(mfl_g_wbuf, 20);
    mfl_g_n_kv_heads = mfl_peek_i32(mfl_g_wbuf, 24);
    mfl_g_vocab_size = mfl_peek_i32(mfl_g_wbuf, 28);
    mfl_g_seq_len = mfl_peek_i32(mfl_g_wbuf, 32);
    mfl_g_shared_cls = mfl_peek_u8(mfl_g_wbuf, 36);
    mfl_g_gs = mfl_peek_i32(mfl_g_wbuf, 37);
    if ((v_v4 == 1)) {
        mfl_g_rope_theta = mfl_peek_f32(mfl_g_wbuf, 41);
        mfl_g_l3_scaling = mfl_peek_u8(mfl_g_wbuf, 45);
        mfl_g_l3_factor = mfl_peek_f32(mfl_g_wbuf, 46);
        mfl_g_l3_low = mfl_peek_f32(mfl_g_wbuf, 50);
        mfl_g_l3_high = mfl_peek_f32(mfl_g_wbuf, 54);
        mfl_g_l3_orig = mfl_peek_i32(mfl_g_wbuf, 58);
    }
    mfl_g_head_size = (mfl_g_dim / mfl_g_n_heads);
    mfl_g_kv_dim = ((mfl_g_dim * mfl_g_n_kv_heads) / mfl_g_n_heads);
    v_o = 256;
    mfl_g_w_rms_att = v_o;
    v_o = (v_o + ((mfl_g_n_layers * mfl_g_dim) * 4));
    mfl_g_w_rms_ffn = v_o;
    v_o = (v_o + ((mfl_g_n_layers * mfl_g_dim) * 4));
    mfl_g_w_rms_final = v_o;
    v_o = (v_o + (mfl_g_dim * 4));
    mfl_g_w_tok = v_o;
    v_o = ({ int64_t _sq22_0 = v_o; int64_t _sq22_1 = mfl_qblock_0((mfl_g_vocab_size * mfl_g_dim)); (_sq22_0 + _sq22_1); });
    mfl_g_wq_b = v_o;
    v_o = ({ int64_t _sq24_0 = v_o; int64_t _sq24_1 = ({ int64_t _sq23_0 = mfl_g_n_layers; int64_t _sq23_1 = mfl_qblock_0((mfl_g_dim * mfl_g_dim)); (_sq23_0 * _sq23_1); }); (_sq24_0 + _sq24_1); });
    mfl_g_wk_b = v_o;
    v_o = ({ int64_t _sq26_0 = v_o; int64_t _sq26_1 = ({ int64_t _sq25_0 = mfl_g_n_layers; int64_t _sq25_1 = mfl_qblock_0((mfl_g_dim * mfl_g_kv_dim)); (_sq25_0 * _sq25_1); }); (_sq26_0 + _sq26_1); });
    mfl_g_wv_b = v_o;
    v_o = ({ int64_t _sq28_0 = v_o; int64_t _sq28_1 = ({ int64_t _sq27_0 = mfl_g_n_layers; int64_t _sq27_1 = mfl_qblock_0((mfl_g_dim * mfl_g_kv_dim)); (_sq27_0 * _sq27_1); }); (_sq28_0 + _sq28_1); });
    mfl_g_wo_b = v_o;
    v_o = ({ int64_t _sq30_0 = v_o; int64_t _sq30_1 = ({ int64_t _sq29_0 = mfl_g_n_layers; int64_t _sq29_1 = mfl_qblock_0((mfl_g_dim * mfl_g_dim)); (_sq29_0 * _sq29_1); }); (_sq30_0 + _sq30_1); });
    mfl_g_w1_b = v_o;
    v_o = ({ int64_t _sq32_0 = v_o; int64_t _sq32_1 = ({ int64_t _sq31_0 = mfl_g_n_layers; int64_t _sq31_1 = mfl_qblock_0((mfl_g_dim * mfl_g_hidden_dim)); (_sq31_0 * _sq31_1); }); (_sq32_0 + _sq32_1); });
    mfl_g_w2_b = v_o;
    v_o = ({ int64_t _sq34_0 = v_o; int64_t _sq34_1 = ({ int64_t _sq33_0 = mfl_g_n_layers; int64_t _sq33_1 = mfl_qblock_0((mfl_g_hidden_dim * mfl_g_dim)); (_sq33_0 * _sq33_1); }); (_sq34_0 + _sq34_1); });
    mfl_g_w3_b = v_o;
    v_o = ({ int64_t _sq36_0 = v_o; int64_t _sq36_1 = ({ int64_t _sq35_0 = mfl_g_n_layers; int64_t _sq35_1 = mfl_qblock_0((mfl_g_dim * mfl_g_hidden_dim)); (_sq35_0 * _sq35_1); }); (_sq36_0 + _sq36_1); });
    if ((mfl_g_shared_cls == 1)) {
        mfl_g_wcls_b = mfl_g_w_tok;
    } else {
        mfl_g_wcls_b = v_o;
    }
    mfl_g_rope_freqs = mfl_raw_alloc(((mfl_g_head_size / 2) * 4));
    v_hd = 0;
    while ((v_hd < mfl_g_head_size)) {
        ({ int64_t _sq37_0 = mfl_g_rope_freqs; int64_t _sq37_1 = ((v_hd / 2) * 4); double _sq37_2 = mfl_rope_freq_0(v_hd); mfl_poke_f32(_sq37_0, _sq37_1, _sq37_2); });
        v_hd = (v_hd + 2);
    }
}

int64_t mfl_qblock_0(int64_t v_size) {
    int64_t v_n = 0;
    v_n = ({ int64_t _sq38_0 = mfl_wsz_0(v_size); int64_t _sq38_1 = (4 * (v_size / mfl_g_gs)); (_sq38_0 + _sq38_1); });
    return v_n;
    return v_n;
}

int64_t mfl_wsz_0(int64_t v_size) {
    int64_t v_n = 0;
    v_n = v_size;
    if ((mfl_g_quant4 == 1)) {
        v_n = (v_size / 2);
    }
    return v_n;
    return v_n;
}

double mfl_rope_freq_0(int64_t v_hd) {
    double v_freq = 0.0;
    double v_wavelen = 0.0;
    double v_lowwl = 0.0;
    double v_highwl = 0.0;
    double v_smooth = 0.0;
    v_freq = ({ double _sq41_0 = 1.0; double _sq41_1 = ({ double _sq40_0 = mfl_g_rope_theta; double _sq40_1 = ({ double _sq39_0 = ((double)(v_hd)); double _sq39_1 = ((double)(mfl_g_head_size)); (_sq39_0 / _sq39_1); }); mfl_math_pow((double)(_sq40_0), (double)(_sq40_1)); }); (_sq41_0 / _sq41_1); });
    if ((mfl_g_l3_scaling == 1)) {
        v_wavelen = ({ double _sq43_0 = ({ double _sq42_0 = 2.0; double _sq42_1 = mfl_math_pi(); (_sq42_0 * _sq42_1); }); double _sq43_1 = v_freq; (_sq43_0 / _sq43_1); });
        v_lowwl = ({ double _sq44_0 = ((double)(mfl_g_l3_orig)); double _sq44_1 = mfl_g_l3_low; (_sq44_0 / _sq44_1); });
        v_highwl = ({ double _sq45_0 = ((double)(mfl_g_l3_orig)); double _sq45_1 = mfl_g_l3_high; (_sq45_0 / _sq45_1); });
        if ((v_wavelen > v_lowwl)) {
            v_freq = (v_freq / mfl_g_l3_factor);
        } else {
            if ((v_wavelen > v_highwl)) {
                v_smooth = ({ double _sq48_0 = ({ double _sq47_0 = ({ double _sq46_0 = ((double)(mfl_g_l3_orig)); double _sq46_1 = v_wavelen; (_sq46_0 / _sq46_1); }); double _sq47_1 = mfl_g_l3_low; (_sq47_0 - _sq47_1); }); double _sq48_1 = (mfl_g_l3_high - mfl_g_l3_low); (_sq48_0 / _sq48_1); });
                v_freq = (((1.0 - v_smooth) * (v_freq / mfl_g_l3_factor)) + (v_smooth * v_freq));
            }
        }
    }
    return v_freq;
    return v_freq;
}

void mfl_alloc_state_0(void) {
    mfl_g_s_x = mfl_raw_alloc((mfl_g_dim * 4));
    mfl_g_s_xb = mfl_raw_alloc((mfl_g_dim * 4));
    mfl_g_s_xb2 = mfl_raw_alloc((mfl_g_dim * 4));
    mfl_g_s_hb = mfl_raw_alloc((mfl_g_hidden_dim * 4));
    mfl_g_s_hb2 = mfl_raw_alloc((mfl_g_hidden_dim * 4));
    mfl_g_s_q = mfl_raw_alloc((mfl_g_dim * 4));
    mfl_g_s_att = mfl_raw_alloc(((mfl_g_n_heads * mfl_g_seq_len) * 4));
    mfl_g_s_logits = mfl_raw_alloc((mfl_g_vocab_size * 4));
    mfl_g_slotstride = ((mfl_g_n_layers * mfl_g_seq_len) * mfl_g_kv_dim);
    mfl_g_s_key_cache = mfl_raw_alloc(((mfl_g_cb_slots * mfl_g_slotstride) * 4));
    mfl_g_s_val_cache = mfl_raw_alloc(((mfl_g_cb_slots * mfl_g_slotstride) * 4));
    mfl_g_s_xq = mfl_raw_alloc(mfl_g_dim);
    mfl_g_s_xq_s = mfl_raw_alloc(((mfl_g_dim / mfl_g_gs) * 4));
    mfl_g_s_hq = mfl_raw_alloc(mfl_g_hidden_dim);
    mfl_g_s_hq_s = mfl_raw_alloc(((mfl_g_hidden_dim / mfl_g_gs) * 4));
    mfl_g_cache_buf = mfl_raw_alloc((mfl_g_seq_len * 4));
    mfl_alloc_prefill_0();
    mfl_alloc_spec_0();
    mfl_g_cb_active = mfl_raw_alloc((mfl_g_cb_slots * 4));
    mfl_g_cb_pos = mfl_raw_alloc((mfl_g_cb_slots * 4));
    mfl_g_cb_curtok = mfl_raw_alloc((mfl_g_cb_slots * 4));
    mfl_g_cb_kt = mfl_raw_alloc(((mfl_g_cb_slots * mfl_g_kv_dim) * 4));
    mfl_g_cb_vt = mfl_raw_alloc(((mfl_g_cb_slots * mfl_g_kv_dim) * 4));
    mfl_g_cb_next = mfl_raw_alloc((mfl_g_cb_slots * 4));
}

void mfl_alloc_prefill_0(void) {
    int64_t v_t = 0;
    v_t = mfl_g_pb_tile;
    mfl_g_pb_x = mfl_raw_alloc(((v_t * mfl_g_dim) * 4));
    mfl_g_pb_xb = mfl_raw_alloc(((v_t * mfl_g_dim) * 4));
    mfl_g_pb_xb2 = mfl_raw_alloc(((v_t * mfl_g_dim) * 4));
    mfl_g_pb_q = mfl_raw_alloc(((v_t * mfl_g_dim) * 4));
    mfl_g_pb_hb = mfl_raw_alloc(((v_t * mfl_g_hidden_dim) * 4));
    mfl_g_pb_hb2 = mfl_raw_alloc(((v_t * mfl_g_hidden_dim) * 4));
    mfl_g_pb_xq = mfl_raw_alloc((v_t * mfl_g_dim));
    mfl_g_pb_xqs = mfl_raw_alloc(((v_t * (mfl_g_dim / mfl_g_gs)) * 4));
    mfl_g_pb_hq = mfl_raw_alloc((v_t * mfl_g_hidden_dim));
    mfl_g_pb_hqs = mfl_raw_alloc(((v_t * (mfl_g_hidden_dim / mfl_g_gs)) * 4));
}

void mfl_alloc_spec_0(void) {
    mfl_g_spec_logits = mfl_raw_alloc(((mfl_g_spec_maxb * mfl_g_vocab_size) * 4));
    mfl_g_spec_argmax = mfl_raw_alloc((mfl_g_spec_maxb * 4));
    mfl_g_spec_tbuf = mfl_raw_alloc((mfl_g_spec_maxb * 4));
}

void mfl_load_tokenizer_auto_0(void) {
    if ((mfl_g_vocab_size > 100000)) {
        mfl_g_tok_l3 = 1;
        mfl_load_tokenizer_l3_0("models/tokenizer-l3.bin");
        return;
    }
    mfl_load_tokenizer_0("models/tokenizer.bin");
}

void mfl_load_tokenizer_l3_0(char* v_path) {
    mfl_bytes v_b = {0};
    int64_t v_n = 0;
    int64_t v_off = 0;
    int64_t v_i = 0;
    mfl_slice v_v = {0};
    int64_t v_l = 0;
    mfl_bytes v_tb = {0};
    v_b = mfl_read_file_bytes(v_path);
    if (({ int64_t _sq49_0 = ((v_b).len); int64_t _sq49_1 = 0; (_sq49_0 == _sq49_1); })) {
        { const char* _s = (mfl_cat("cannot read tokenizer: ", v_path)); fputs(_s ? _s : "", stdout); }
        fputs("\n", stdout);
        mfl_exit(1);
    }
    v_n = mfl_read_i32_0(v_b, 0);
    v_off = 4;
    v_i = 0;
    v_v = (mfl_slice){0};
    while ((v_i < v_n)) {
        v_l = mfl_read_i32_0(v_b, v_off);
        v_off = (v_off + 4);
        v_tb = mfl_bytes_sub(v_b, v_off, (v_off + v_l));
        v_v = ({ mfl_slice _sq50_0 = v_v; char* _sq50_1 = mfl_bytes_str(v_tb); mfl_append(_sq50_0, &((char*[1]){_sq50_1})[0], sizeof(char*)); });
        if ((v_i < 128000)) {
            mfl_map_set(mfl_g_l3_idx, 0, mfl_bytes_hex(v_tb), &((int64_t[1]){v_i})[0]);
        }
        v_off = (v_off + v_l);
        v_i = (v_i + 1);
    }
    mfl_g_vocab_l3 = v_v;
}

int64_t mfl_read_i32_0(mfl_bytes v_b, int64_t v_off) {
    int64_t v_v = 0;
    v_v = ({ int64_t _sq56_0 = ({ int64_t _sq54_0 = ({ int64_t _sq52_0 = mfl_byte_at(v_b, v_off); int64_t _sq52_1 = ({ int64_t _sq51_0 = mfl_byte_at(v_b, (v_off + 1)); int64_t _sq51_1 = 8; (_sq51_0 << _sq51_1); }); (_sq52_0 | _sq52_1); }); int64_t _sq54_1 = ({ int64_t _sq53_0 = mfl_byte_at(v_b, (v_off + 2)); int64_t _sq53_1 = 16; (_sq53_0 << _sq53_1); }); (_sq54_0 | _sq54_1); }); int64_t _sq56_1 = ({ int64_t _sq55_0 = mfl_byte_at(v_b, (v_off + 3)); int64_t _sq55_1 = 24; (_sq55_0 << _sq55_1); }); (_sq56_0 | _sq56_1); });
    if ((v_v >= 2147483648)) {
        v_v = (v_v - 4294967296);
    }
    return v_v;
    return v_v;
}

void mfl_load_tokenizer_0(char* v_path) {
    mfl_bytes v_b = {0};
    int64_t v_tbuf = 0;
    int64_t v_off = 0;
    int64_t v_i = 0;
    int64_t v_n = 0;
    char* v_piece = "";
    v_b = mfl_read_file_bytes(v_path);
    if (({ int64_t _sq57_0 = ((v_b).len); int64_t _sq57_1 = 0; (_sq57_0 == _sq57_1); })) {
        { const char* _s = (mfl_cat("cannot read tokenizer: ", v_path)); fputs(_s ? _s : "", stdout); }
        fputs("\n", stdout);
        mfl_exit(1);
    }
    v_tbuf = mfl_bytes_to_buf_0(v_b);
    mfl_g_tok_scores = mfl_raw_alloc((mfl_g_vocab_size * 4));
    v_off = 4;
    v_i = 0;
    while ((v_i < mfl_g_vocab_size)) {
        ({ int64_t _sq58_0 = mfl_g_tok_scores; int64_t _sq58_1 = (v_i * 4); double _sq58_2 = mfl_peek_f32(v_tbuf, v_off); mfl_poke_f32(_sq58_0, _sq58_1, _sq58_2); });
        v_off = (v_off + 4);
        v_n = mfl_read_i32_0(v_b, v_off);
        v_off = (v_off + 4);
        v_piece = mfl_bytes_str(mfl_bytes_sub(v_b, v_off, (v_off + v_n)));
        mfl_g_vocab = mfl_append(mfl_g_vocab, &((char*[1]){v_piece})[0], sizeof(char*));
        mfl_map_set(mfl_g_vocab_idx, 0, v_piece, &((int64_t[1]){v_i})[0]);
        v_off = (v_off + v_n);
        v_i = (v_i + 1);
    }
    mfl_raw_free(v_tbuf);
}

int64_t mfl_bytes_to_buf_0(mfl_bytes v_b) {
    int64_t v_p = 0;
    int64_t v_n = 0;
    int64_t v_i = 0;
    int64_t v_w = 0;
    v_n = ((v_b).len);
    v_p = mfl_raw_alloc((v_n + 4));
    v_i = 0;
    while (((v_i + 4) <= v_n)) {
        v_w = ({ int64_t _sq64_0 = ({ int64_t _sq62_0 = ({ int64_t _sq60_0 = mfl_byte_at(v_b, v_i); int64_t _sq60_1 = ({ int64_t _sq59_0 = mfl_byte_at(v_b, (v_i + 1)); int64_t _sq59_1 = 8; (_sq59_0 << _sq59_1); }); (_sq60_0 | _sq60_1); }); int64_t _sq62_1 = ({ int64_t _sq61_0 = mfl_byte_at(v_b, (v_i + 2)); int64_t _sq61_1 = 16; (_sq61_0 << _sq61_1); }); (_sq62_0 | _sq62_1); }); int64_t _sq64_1 = ({ int64_t _sq63_0 = mfl_byte_at(v_b, (v_i + 3)); int64_t _sq63_1 = 24; (_sq63_0 << _sq63_1); }); (_sq64_0 | _sq64_1); });
        mfl_poke_i32(v_p, v_i, v_w);
        v_i = (v_i + 4);
    }
    while ((v_i < v_n)) {
        ({ int64_t _sq65_0 = v_p; int64_t _sq65_1 = v_i; int64_t _sq65_2 = mfl_byte_at(v_b, v_i); mfl_poke_u8(_sq65_0, _sq65_1, _sq65_2); });
        v_i = (v_i + 1);
    }
    return v_p;
    return v_p;
}

void mfl_worker_0(mfl_chan* v_jc, mfl_chan* v_dc) {
    int64_t v_sc = 0;
    int64_t v_att = 0;
    int64_t v_j = 0;
    v_sc = mfl_raw_alloc(mfl_g_hidden_dim);
    v_att = mfl_raw_alloc((mfl_g_seq_len * 4));
    {
        mfl_chan* _ch0 = v_jc;
        while (mfl_chan_recv2(_ch0, &v_j)) {
            mfl_do_job_0(v_j, v_sc, v_att);
            mfl_chan_send(v_dc, &((int64_t[1]){1})[0]);
        }
    }
}

void mfl_do_job_0(int64_t v_j, int64_t v_sc, int64_t v_att) {
    int64_t v_t = 0;
    int64_t v_l = 0;
    int64_t v_pos = 0;
    int64_t v_lo = 0;
    int64_t v_hi = 0;
    int64_t v_kptr = 0;
    int64_t v_vptr = 0;
    int64_t v_h = 0;
    int64_t v_klo = 0;
    int64_t v_khi = 0;
    int64_t v_qb = 0;
    int64_t v_kb = 0;
    int64_t v_vb = 0;
    int64_t v_ob = 0;
    int64_t v_wb = 0;
    int64_t v_u = 0;
    int64_t v_b = 0;
    int64_t v_i = 0;
    double v_v = 0.0;
    int64_t v_jj = 0;
    int64_t v_hh = 0;
    int64_t v_base = 0;
    int64_t v_best = 0;
    double v_bv = 0.0;
    int64_t v_ii = 0;
    int64_t v_s = 0;
    int64_t v_koff = 0;
    int64_t v_kdst = 0;
    v_t = (v_j & 31);
    v_l = ((v_j >> 5) & 255);
    v_pos = ((v_j >> 13) & 4095);
    v_lo = ((v_j >> 25) & 524287);
    v_hi = ((v_j >> 44) & 524287);
    if ((v_t == 0)) {
        ({ int64_t _sq68_0 = mfl_g_s_q; int64_t _sq68_1 = mfl_g_s_xq; int64_t _sq68_2 = mfl_g_s_xq_s; int64_t _sq68_3 = ({ int64_t _sq67_0 = mfl_g_wq_b; int64_t _sq67_1 = ({ int64_t _sq66_0 = v_l; int64_t _sq66_1 = mfl_qblock_0((mfl_g_dim * mfl_g_dim)); (_sq66_0 * _sq66_1); }); (_sq67_0 + _sq67_1); }); int64_t _sq68_4 = (mfl_g_dim * mfl_g_dim); int64_t _sq68_5 = mfl_g_dim; int64_t _sq68_6 = v_lo; int64_t _sq68_7 = v_hi; int64_t _sq68_8 = v_sc; mfl_mmj_0(_sq68_0, _sq68_1, _sq68_2, _sq68_3, _sq68_4, _sq68_5, _sq68_6, _sq68_7, _sq68_8); });
    }
    if ((v_t == 1)) {
        v_kptr = (mfl_g_s_key_cache + (((mfl_g_kv_off + ((v_l * mfl_g_seq_len) * mfl_g_kv_dim)) + (v_pos * mfl_g_kv_dim)) * 4));
        ({ int64_t _sq71_0 = v_kptr; int64_t _sq71_1 = mfl_g_s_xq; int64_t _sq71_2 = mfl_g_s_xq_s; int64_t _sq71_3 = ({ int64_t _sq70_0 = mfl_g_wk_b; int64_t _sq70_1 = ({ int64_t _sq69_0 = v_l; int64_t _sq69_1 = mfl_qblock_0((mfl_g_dim * mfl_g_kv_dim)); (_sq69_0 * _sq69_1); }); (_sq70_0 + _sq70_1); }); int64_t _sq71_4 = (mfl_g_dim * mfl_g_kv_dim); int64_t _sq71_5 = mfl_g_dim; int64_t _sq71_6 = v_lo; int64_t _sq71_7 = v_hi; int64_t _sq71_8 = v_sc; mfl_mmj_0(_sq71_0, _sq71_1, _sq71_2, _sq71_3, _sq71_4, _sq71_5, _sq71_6, _sq71_7, _sq71_8); });
    }
    if ((v_t == 2)) {
        v_vptr = (mfl_g_s_val_cache + (((mfl_g_kv_off + ((v_l * mfl_g_seq_len) * mfl_g_kv_dim)) + (v_pos * mfl_g_kv_dim)) * 4));
        ({ int64_t _sq74_0 = v_vptr; int64_t _sq74_1 = mfl_g_s_xq; int64_t _sq74_2 = mfl_g_s_xq_s; int64_t _sq74_3 = ({ int64_t _sq73_0 = mfl_g_wv_b; int64_t _sq73_1 = ({ int64_t _sq72_0 = v_l; int64_t _sq72_1 = mfl_qblock_0((mfl_g_dim * mfl_g_kv_dim)); (_sq72_0 * _sq72_1); }); (_sq73_0 + _sq73_1); }); int64_t _sq74_4 = (mfl_g_dim * mfl_g_kv_dim); int64_t _sq74_5 = mfl_g_dim; int64_t _sq74_6 = v_lo; int64_t _sq74_7 = v_hi; int64_t _sq74_8 = v_sc; mfl_mmj_0(_sq74_0, _sq74_1, _sq74_2, _sq74_3, _sq74_4, _sq74_5, _sq74_6, _sq74_7, _sq74_8); });
    }
    if ((v_t == 3)) {
        ({ int64_t _sq77_0 = mfl_g_s_xb2; int64_t _sq77_1 = mfl_g_s_xq; int64_t _sq77_2 = mfl_g_s_xq_s; int64_t _sq77_3 = ({ int64_t _sq76_0 = mfl_g_wo_b; int64_t _sq76_1 = ({ int64_t _sq75_0 = v_l; int64_t _sq75_1 = mfl_qblock_0((mfl_g_dim * mfl_g_dim)); (_sq75_0 * _sq75_1); }); (_sq76_0 + _sq76_1); }); int64_t _sq77_4 = (mfl_g_dim * mfl_g_dim); int64_t _sq77_5 = mfl_g_dim; int64_t _sq77_6 = v_lo; int64_t _sq77_7 = v_hi; int64_t _sq77_8 = v_sc; mfl_mmj_0(_sq77_0, _sq77_1, _sq77_2, _sq77_3, _sq77_4, _sq77_5, _sq77_6, _sq77_7, _sq77_8); });
    }
    if ((v_t == 4)) {
        ({ int64_t _sq80_0 = mfl_g_s_hb; int64_t _sq80_1 = mfl_g_s_xq; int64_t _sq80_2 = mfl_g_s_xq_s; int64_t _sq80_3 = ({ int64_t _sq79_0 = mfl_g_w1_b; int64_t _sq79_1 = ({ int64_t _sq78_0 = v_l; int64_t _sq78_1 = mfl_qblock_0((mfl_g_dim * mfl_g_hidden_dim)); (_sq78_0 * _sq78_1); }); (_sq79_0 + _sq79_1); }); int64_t _sq80_4 = (mfl_g_dim * mfl_g_hidden_dim); int64_t _sq80_5 = mfl_g_dim; int64_t _sq80_6 = v_lo; int64_t _sq80_7 = v_hi; int64_t _sq80_8 = v_sc; mfl_mmj_0(_sq80_0, _sq80_1, _sq80_2, _sq80_3, _sq80_4, _sq80_5, _sq80_6, _sq80_7, _sq80_8); });
    }
    if ((v_t == 5)) {
        ({ int64_t _sq83_0 = mfl_g_s_hb2; int64_t _sq83_1 = mfl_g_s_xq; int64_t _sq83_2 = mfl_g_s_xq_s; int64_t _sq83_3 = ({ int64_t _sq82_0 = mfl_g_w3_b; int64_t _sq82_1 = ({ int64_t _sq81_0 = v_l; int64_t _sq81_1 = mfl_qblock_0((mfl_g_dim * mfl_g_hidden_dim)); (_sq81_0 * _sq81_1); }); (_sq82_0 + _sq82_1); }); int64_t _sq83_4 = (mfl_g_dim * mfl_g_hidden_dim); int64_t _sq83_5 = mfl_g_dim; int64_t _sq83_6 = v_lo; int64_t _sq83_7 = v_hi; int64_t _sq83_8 = v_sc; mfl_mmj_0(_sq83_0, _sq83_1, _sq83_2, _sq83_3, _sq83_4, _sq83_5, _sq83_6, _sq83_7, _sq83_8); });
    }
    if ((v_t == 6)) {
        ({ int64_t _sq86_0 = mfl_g_s_xb; int64_t _sq86_1 = mfl_g_s_hq; int64_t _sq86_2 = mfl_g_s_hq_s; int64_t _sq86_3 = ({ int64_t _sq85_0 = mfl_g_w2_b; int64_t _sq85_1 = ({ int64_t _sq84_0 = v_l; int64_t _sq84_1 = mfl_qblock_0((mfl_g_hidden_dim * mfl_g_dim)); (_sq84_0 * _sq84_1); }); (_sq85_0 + _sq85_1); }); int64_t _sq86_4 = (mfl_g_hidden_dim * mfl_g_dim); int64_t _sq86_5 = mfl_g_hidden_dim; int64_t _sq86_6 = v_lo; int64_t _sq86_7 = v_hi; int64_t _sq86_8 = v_sc; mfl_mmj_0(_sq86_0, _sq86_1, _sq86_2, _sq86_3, _sq86_4, _sq86_5, _sq86_6, _sq86_7, _sq86_8); });
    }
    if ((v_t == 7)) {
        mfl_mmj_0(mfl_g_s_logits, mfl_g_s_xq, mfl_g_s_xq_s, mfl_g_wcls_b, (mfl_g_dim * mfl_g_vocab_size), mfl_g_dim, v_lo, v_hi, v_sc);
    }
    if ((v_t == 8)) {
        v_h = v_lo;
        while ((v_h < v_hi)) {
            mfl_attn_head_0(v_h, v_l, v_pos);
            v_h = (v_h + 1);
        }
    }
    if ((v_t == 9)) {
        ({ int64_t _sq89_0 = mfl_g_s_q; int64_t _sq89_1 = mfl_g_s_xq; int64_t _sq89_2 = mfl_g_s_xq_s; int64_t _sq89_3 = ({ int64_t _sq88_0 = mfl_g_wq_b; int64_t _sq88_1 = ({ int64_t _sq87_0 = v_l; int64_t _sq87_1 = mfl_qblock_0((mfl_g_dim * mfl_g_dim)); (_sq87_0 * _sq87_1); }); (_sq88_0 + _sq88_1); }); int64_t _sq89_4 = (mfl_g_dim * mfl_g_dim); int64_t _sq89_5 = mfl_g_dim; int64_t _sq89_6 = v_lo; int64_t _sq89_7 = v_hi; int64_t _sq89_8 = v_sc; mfl_mmj_0(_sq89_0, _sq89_1, _sq89_2, _sq89_3, _sq89_4, _sq89_5, _sq89_6, _sq89_7, _sq89_8); });
        v_klo = ((v_lo * mfl_g_kv_dim) / mfl_g_dim);
        v_khi = ((v_hi * mfl_g_kv_dim) / mfl_g_dim);
        if ((v_khi > v_klo)) {
            v_kptr = (mfl_g_s_key_cache + (((mfl_g_kv_off + ((v_l * mfl_g_seq_len) * mfl_g_kv_dim)) + (v_pos * mfl_g_kv_dim)) * 4));
            v_vptr = (mfl_g_s_val_cache + (((mfl_g_kv_off + ((v_l * mfl_g_seq_len) * mfl_g_kv_dim)) + (v_pos * mfl_g_kv_dim)) * 4));
            ({ int64_t _sq92_0 = v_kptr; int64_t _sq92_1 = mfl_g_s_xq; int64_t _sq92_2 = mfl_g_s_xq_s; int64_t _sq92_3 = ({ int64_t _sq91_0 = mfl_g_wk_b; int64_t _sq91_1 = ({ int64_t _sq90_0 = v_l; int64_t _sq90_1 = mfl_qblock_0((mfl_g_dim * mfl_g_kv_dim)); (_sq90_0 * _sq90_1); }); (_sq91_0 + _sq91_1); }); int64_t _sq92_4 = (mfl_g_dim * mfl_g_kv_dim); int64_t _sq92_5 = mfl_g_dim; int64_t _sq92_6 = v_klo; int64_t _sq92_7 = v_khi; int64_t _sq92_8 = v_sc; mfl_mmj_0(_sq92_0, _sq92_1, _sq92_2, _sq92_3, _sq92_4, _sq92_5, _sq92_6, _sq92_7, _sq92_8); });
            ({ int64_t _sq95_0 = v_vptr; int64_t _sq95_1 = mfl_g_s_xq; int64_t _sq95_2 = mfl_g_s_xq_s; int64_t _sq95_3 = ({ int64_t _sq94_0 = mfl_g_wv_b; int64_t _sq94_1 = ({ int64_t _sq93_0 = v_l; int64_t _sq93_1 = mfl_qblock_0((mfl_g_dim * mfl_g_kv_dim)); (_sq93_0 * _sq93_1); }); (_sq94_0 + _sq94_1); }); int64_t _sq95_4 = (mfl_g_dim * mfl_g_kv_dim); int64_t _sq95_5 = mfl_g_dim; int64_t _sq95_6 = v_klo; int64_t _sq95_7 = v_khi; int64_t _sq95_8 = v_sc; mfl_mmj_0(_sq95_0, _sq95_1, _sq95_2, _sq95_3, _sq95_4, _sq95_5, _sq95_6, _sq95_7, _sq95_8); });
        }
    }
    if ((v_t == 10)) {
        ({ int64_t _sq98_0 = mfl_g_s_hb; int64_t _sq98_1 = mfl_g_s_xq; int64_t _sq98_2 = mfl_g_s_xq_s; int64_t _sq98_3 = ({ int64_t _sq97_0 = mfl_g_w1_b; int64_t _sq97_1 = ({ int64_t _sq96_0 = v_l; int64_t _sq96_1 = mfl_qblock_0((mfl_g_dim * mfl_g_hidden_dim)); (_sq96_0 * _sq96_1); }); (_sq97_0 + _sq97_1); }); int64_t _sq98_4 = (mfl_g_dim * mfl_g_hidden_dim); int64_t _sq98_5 = mfl_g_dim; int64_t _sq98_6 = v_lo; int64_t _sq98_7 = v_hi; int64_t _sq98_8 = v_sc; mfl_mmj_0(_sq98_0, _sq98_1, _sq98_2, _sq98_3, _sq98_4, _sq98_5, _sq98_6, _sq98_7, _sq98_8); });
        ({ int64_t _sq101_0 = mfl_g_s_hb2; int64_t _sq101_1 = mfl_g_s_xq; int64_t _sq101_2 = mfl_g_s_xq_s; int64_t _sq101_3 = ({ int64_t _sq100_0 = mfl_g_w3_b; int64_t _sq100_1 = ({ int64_t _sq99_0 = v_l; int64_t _sq99_1 = mfl_qblock_0((mfl_g_dim * mfl_g_hidden_dim)); (_sq99_0 * _sq99_1); }); (_sq100_0 + _sq100_1); }); int64_t _sq101_4 = (mfl_g_dim * mfl_g_hidden_dim); int64_t _sq101_5 = mfl_g_dim; int64_t _sq101_6 = v_lo; int64_t _sq101_7 = v_hi; int64_t _sq101_8 = v_sc; mfl_mmj_0(_sq101_0, _sq101_1, _sq101_2, _sq101_3, _sq101_4, _sq101_5, _sq101_6, _sq101_7, _sq101_8); });
    }
    if ((v_t == 11)) {
        v_qb = ({ int64_t _sq103_0 = mfl_g_wq_b; int64_t _sq103_1 = ({ int64_t _sq102_0 = v_l; int64_t _sq102_1 = mfl_qblock_0((mfl_g_dim * mfl_g_dim)); (_sq102_0 * _sq102_1); }); (_sq103_0 + _sq103_1); });
        mfl_matmul_q_batch_0(mfl_g_pb_q, mfl_g_dim, mfl_g_pb_xq, mfl_g_pb_xqs, v_qb, (v_qb + (mfl_g_dim * mfl_g_dim)), mfl_g_dim, v_lo, v_hi, mfl_g_pb_B);
    }
    if ((v_t == 12)) {
        v_kb = ({ int64_t _sq105_0 = mfl_g_wk_b; int64_t _sq105_1 = ({ int64_t _sq104_0 = v_l; int64_t _sq104_1 = mfl_qblock_0((mfl_g_dim * mfl_g_kv_dim)); (_sq104_0 * _sq104_1); }); (_sq105_0 + _sq105_1); });
        mfl_matmul_q_batch_0((mfl_g_s_key_cache + ((((v_l * mfl_g_seq_len) * mfl_g_kv_dim) + (mfl_g_pb_tstart * mfl_g_kv_dim)) * 4)), mfl_g_kv_dim, mfl_g_pb_xq, mfl_g_pb_xqs, v_kb, (v_kb + (mfl_g_dim * mfl_g_kv_dim)), mfl_g_dim, v_lo, v_hi, mfl_g_pb_B);
    }
    if ((v_t == 13)) {
        v_vb = ({ int64_t _sq107_0 = mfl_g_wv_b; int64_t _sq107_1 = ({ int64_t _sq106_0 = v_l; int64_t _sq106_1 = mfl_qblock_0((mfl_g_dim * mfl_g_kv_dim)); (_sq106_0 * _sq106_1); }); (_sq107_0 + _sq107_1); });
        mfl_matmul_q_batch_0((mfl_g_s_val_cache + ((((v_l * mfl_g_seq_len) * mfl_g_kv_dim) + (mfl_g_pb_tstart * mfl_g_kv_dim)) * 4)), mfl_g_kv_dim, mfl_g_pb_xq, mfl_g_pb_xqs, v_vb, (v_vb + (mfl_g_dim * mfl_g_kv_dim)), mfl_g_dim, v_lo, v_hi, mfl_g_pb_B);
    }
    if ((v_t == 14)) {
        v_ob = ({ int64_t _sq109_0 = mfl_g_wo_b; int64_t _sq109_1 = ({ int64_t _sq108_0 = v_l; int64_t _sq108_1 = mfl_qblock_0((mfl_g_dim * mfl_g_dim)); (_sq108_0 * _sq108_1); }); (_sq109_0 + _sq109_1); });
        mfl_matmul_q_batch_0(mfl_g_pb_xb2, mfl_g_dim, mfl_g_pb_xq, mfl_g_pb_xqs, v_ob, (v_ob + (mfl_g_dim * mfl_g_dim)), mfl_g_dim, v_lo, v_hi, mfl_g_pb_B);
    }
    if ((v_t == 15)) {
        v_wb = ({ int64_t _sq111_0 = mfl_g_w1_b; int64_t _sq111_1 = ({ int64_t _sq110_0 = v_l; int64_t _sq110_1 = mfl_qblock_0((mfl_g_dim * mfl_g_hidden_dim)); (_sq110_0 * _sq110_1); }); (_sq111_0 + _sq111_1); });
        mfl_matmul_q_batch_0(mfl_g_pb_hb, mfl_g_hidden_dim, mfl_g_pb_xq, mfl_g_pb_xqs, v_wb, (v_wb + (mfl_g_dim * mfl_g_hidden_dim)), mfl_g_dim, v_lo, v_hi, mfl_g_pb_B);
    }
    if ((v_t == 16)) {
        v_wb = ({ int64_t _sq113_0 = mfl_g_w3_b; int64_t _sq113_1 = ({ int64_t _sq112_0 = v_l; int64_t _sq112_1 = mfl_qblock_0((mfl_g_dim * mfl_g_hidden_dim)); (_sq112_0 * _sq112_1); }); (_sq113_0 + _sq113_1); });
        mfl_matmul_q_batch_0(mfl_g_pb_hb2, mfl_g_hidden_dim, mfl_g_pb_xq, mfl_g_pb_xqs, v_wb, (v_wb + (mfl_g_dim * mfl_g_hidden_dim)), mfl_g_dim, v_lo, v_hi, mfl_g_pb_B);
    }
    if ((v_t == 17)) {
        v_wb = ({ int64_t _sq115_0 = mfl_g_w2_b; int64_t _sq115_1 = ({ int64_t _sq114_0 = v_l; int64_t _sq114_1 = mfl_qblock_0((mfl_g_hidden_dim * mfl_g_dim)); (_sq114_0 * _sq114_1); }); (_sq115_0 + _sq115_1); });
        mfl_matmul_q_batch_0(mfl_g_pb_xb, mfl_g_dim, mfl_g_pb_hq, mfl_g_pb_hqs, v_wb, (v_wb + (mfl_g_hidden_dim * mfl_g_dim)), mfl_g_hidden_dim, v_lo, v_hi, mfl_g_pb_B);
    }
    if ((v_t == 18)) {
        v_u = v_lo;
        while ((v_u < v_hi)) {
            v_b = (v_u / mfl_g_n_heads);
            v_h = (v_u % mfl_g_n_heads);
            mfl_attn_head_at_0(v_h, v_l, (mfl_g_pb_tstart + v_b), (mfl_g_pb_q + ((v_b * mfl_g_dim) * 4)), (mfl_g_pb_xb + ((v_b * mfl_g_dim) * 4)), v_att);
            v_u = (v_u + 1);
        }
    }
    if ((v_t == 19)) {
        v_b = v_lo;
        while ((v_b < v_hi)) {
            mfl_rmsnorm_0((mfl_g_pb_xb + ((v_b * mfl_g_dim) * 4)), (mfl_g_pb_x + ((v_b * mfl_g_dim) * 4)), ((mfl_g_wbuf + mfl_g_w_rms_att) + ((v_l * mfl_g_dim) * 4)), mfl_g_dim);
            mfl_quantize_act_0((mfl_g_pb_xq + (v_b * mfl_g_dim)), (mfl_g_pb_xqs + ((v_b * (mfl_g_dim / mfl_g_gs)) * 4)), (mfl_g_pb_xb + ((v_b * mfl_g_dim) * 4)), mfl_g_dim);
            v_b = (v_b + 1);
        }
    }
    if ((v_t == 20)) {
        v_b = v_lo;
        while ((v_b < v_hi)) {
            mfl_rmsnorm_0((mfl_g_pb_xb + ((v_b * mfl_g_dim) * 4)), (mfl_g_pb_x + ((v_b * mfl_g_dim) * 4)), ((mfl_g_wbuf + mfl_g_w_rms_ffn) + ((v_l * mfl_g_dim) * 4)), mfl_g_dim);
            mfl_quantize_act_0((mfl_g_pb_xq + (v_b * mfl_g_dim)), (mfl_g_pb_xqs + ((v_b * (mfl_g_dim / mfl_g_gs)) * 4)), (mfl_g_pb_xb + ((v_b * mfl_g_dim) * 4)), mfl_g_dim);
            v_b = (v_b + 1);
        }
    }
    if ((v_t == 21)) {
        v_b = v_lo;
        while ((v_b < v_hi)) {
            mfl_quantize_act_0((mfl_g_pb_xq + (v_b * mfl_g_dim)), (mfl_g_pb_xqs + ((v_b * (mfl_g_dim / mfl_g_gs)) * 4)), (mfl_g_pb_xb + ((v_b * mfl_g_dim) * 4)), mfl_g_dim);
            v_b = (v_b + 1);
        }
    }
    if ((v_t == 22)) {
        v_b = v_lo;
        while ((v_b < v_hi)) {
            v_i = 0;
            while ((v_i < mfl_g_dim)) {
                ({ int64_t _sq117_0 = (mfl_g_pb_x + ((v_b * mfl_g_dim) * 4)); int64_t _sq117_1 = (v_i * 4); double _sq117_2 = ({ double _sq116_0 = mfl_peek_f32((mfl_g_pb_x + ((v_b * mfl_g_dim) * 4)), (v_i * 4)); double _sq116_1 = mfl_peek_f32((mfl_g_pb_xb2 + ((v_b * mfl_g_dim) * 4)), (v_i * 4)); (_sq116_0 + _sq116_1); }); mfl_poke_f32(_sq117_0, _sq117_1, _sq117_2); });
                v_i = (v_i + 1);
            }
            v_b = (v_b + 1);
        }
    }
    if ((v_t == 23)) {
        v_b = v_lo;
        while ((v_b < v_hi)) {
            v_i = 0;
            while ((v_i < mfl_g_hidden_dim)) {
                v_v = mfl_peek_f32((mfl_g_pb_hb + ((v_b * mfl_g_hidden_dim) * 4)), (v_i * 4));
                v_v = ({ double _sq120_0 = v_v; double _sq120_1 = ({ double _sq119_0 = 1.0; double _sq119_1 = ({ double _sq118_0 = 1.0; double _sq118_1 = mfl_math_exp((double)((0.0 - v_v))); (_sq118_0 + _sq118_1); }); (_sq119_0 / _sq119_1); }); (_sq120_0 * _sq120_1); });
                ({ int64_t _sq122_0 = (mfl_g_pb_hb + ((v_b * mfl_g_hidden_dim) * 4)); int64_t _sq122_1 = (v_i * 4); double _sq122_2 = ({ double _sq121_0 = v_v; double _sq121_1 = mfl_peek_f32((mfl_g_pb_hb2 + ((v_b * mfl_g_hidden_dim) * 4)), (v_i * 4)); (_sq121_0 * _sq121_1); }); mfl_poke_f32(_sq122_0, _sq122_1, _sq122_2); });
                v_i = (v_i + 1);
            }
            mfl_quantize_act_0((mfl_g_pb_hq + (v_b * mfl_g_hidden_dim)), (mfl_g_pb_hqs + ((v_b * (mfl_g_hidden_dim / mfl_g_gs)) * 4)), (mfl_g_pb_hb + ((v_b * mfl_g_hidden_dim) * 4)), mfl_g_hidden_dim);
            v_b = (v_b + 1);
        }
    }
    if ((v_t == 24)) {
        v_b = v_lo;
        while ((v_b < v_hi)) {
            v_i = 0;
            while ((v_i < mfl_g_dim)) {
                ({ int64_t _sq124_0 = (mfl_g_pb_x + ((v_b * mfl_g_dim) * 4)); int64_t _sq124_1 = (v_i * 4); double _sq124_2 = ({ double _sq123_0 = mfl_peek_f32((mfl_g_pb_x + ((v_b * mfl_g_dim) * 4)), (v_i * 4)); double _sq123_1 = mfl_peek_f32((mfl_g_pb_xb + ((v_b * mfl_g_dim) * 4)), (v_i * 4)); (_sq123_0 + _sq123_1); }); mfl_poke_f32(_sq124_0, _sq124_1, _sq124_2); });
                v_i = (v_i + 1);
            }
            v_b = (v_b + 1);
        }
    }
    if ((v_t == 25)) {
        v_b = v_lo;
        while ((v_b < v_hi)) {
            mfl_rope_at_0(v_l, (mfl_g_pb_tstart + v_b), (mfl_g_pb_q + ((v_b * mfl_g_dim) * 4)));
            v_b = (v_b + 1);
        }
    }
    if ((v_t == 26)) {
        v_b = v_lo;
        while ((v_b < v_hi)) {
            mfl_rmsnorm_0((mfl_g_pb_xb + ((v_b * mfl_g_dim) * 4)), (mfl_g_pb_x + ((v_b * mfl_g_dim) * 4)), (mfl_g_wbuf + mfl_g_w_rms_final), mfl_g_dim);
            mfl_quantize_act_0((mfl_g_pb_xq + (v_b * mfl_g_dim)), (mfl_g_pb_xqs + ((v_b * (mfl_g_dim / mfl_g_gs)) * 4)), (mfl_g_pb_xb + ((v_b * mfl_g_dim) * 4)), mfl_g_dim);
            v_b = (v_b + 1);
        }
    }
    if ((v_t == 27)) {
        mfl_matmul_q_batch_0(mfl_g_spec_logits, mfl_g_vocab_size, mfl_g_pb_xq, mfl_g_pb_xqs, mfl_g_wcls_b, (mfl_g_wcls_b + (mfl_g_dim * mfl_g_vocab_size)), mfl_g_dim, v_lo, v_hi, mfl_g_pb_B);
    }
    if ((v_t == 28)) {
        v_kb = ({ int64_t _sq126_0 = mfl_g_wk_b; int64_t _sq126_1 = ({ int64_t _sq125_0 = v_l; int64_t _sq125_1 = mfl_qblock_0((mfl_g_dim * mfl_g_kv_dim)); (_sq125_0 * _sq125_1); }); (_sq126_0 + _sq126_1); });
        mfl_matmul_q_batch_0(mfl_g_cb_kt, mfl_g_kv_dim, mfl_g_pb_xq, mfl_g_pb_xqs, v_kb, (v_kb + (mfl_g_dim * mfl_g_kv_dim)), mfl_g_dim, v_lo, v_hi, mfl_g_pb_B);
    }
    if ((v_t == 29)) {
        v_vb = ({ int64_t _sq128_0 = mfl_g_wv_b; int64_t _sq128_1 = ({ int64_t _sq127_0 = v_l; int64_t _sq127_1 = mfl_qblock_0((mfl_g_dim * mfl_g_kv_dim)); (_sq127_0 * _sq127_1); }); (_sq128_0 + _sq128_1); });
        mfl_matmul_q_batch_0(mfl_g_cb_vt, mfl_g_kv_dim, mfl_g_pb_xq, mfl_g_pb_xqs, v_vb, (v_vb + (mfl_g_dim * mfl_g_kv_dim)), mfl_g_dim, v_lo, v_hi, mfl_g_pb_B);
    }
    if ((v_t == 30)) {
        v_u = v_lo;
        while ((v_u < v_hi)) {
            v_jj = (v_u / mfl_g_n_heads);
            v_hh = (v_u % mfl_g_n_heads);
            mfl_attn_head_cb_0(v_jj, v_hh, v_l, v_att);
            v_u = (v_u + 1);
        }
    }
    if ((v_t == 31)) {
        v_jj = v_lo;
        while ((v_jj < v_hi)) {
            v_base = (mfl_g_spec_logits + ((v_jj * mfl_g_vocab_size) * 4));
            v_best = 0;
            v_bv = mfl_peek_f32(v_base, 0);
            v_ii = 1;
            while ((v_ii < mfl_g_vocab_size)) {
                v_v = mfl_peek_f32(v_base, (v_ii * 4));
                if ((v_v > v_bv)) {
                    v_bv = v_v;
                    v_best = v_ii;
                }
                v_ii = (v_ii + 1);
            }
            mfl_poke_i32(mfl_g_cb_next, (v_jj * 4), v_best);
            v_jj = (v_jj + 1);
        }
    }
    if ((v_t == 32)) {
        v_jj = v_lo;
        while ((v_jj < v_hi)) {
            v_s = mfl_peek_i32(mfl_g_cb_active, (v_jj * 4));
            v_pos = mfl_peek_i32(mfl_g_cb_pos, (v_s * 4));
            v_koff = ((((v_s * mfl_g_slotstride) + ((v_l * mfl_g_seq_len) * mfl_g_kv_dim)) + (v_pos * mfl_g_kv_dim)) * 4);
            v_kdst = (mfl_g_s_key_cache + v_koff);
            mfl_copy_f32_0(v_kdst, (mfl_g_cb_kt + ((v_jj * mfl_g_kv_dim) * 4)), mfl_g_kv_dim);
            mfl_copy_f32_0((mfl_g_s_val_cache + v_koff), (mfl_g_cb_vt + ((v_jj * mfl_g_kv_dim) * 4)), mfl_g_kv_dim);
            mfl_rope_qk_0(v_pos, (mfl_g_pb_q + ((v_jj * mfl_g_dim) * 4)), v_kdst);
            v_jj = (v_jj + 1);
        }
    }
}

void mfl_mmj_0(int64_t v_xout, int64_t v_xq, int64_t v_xs, int64_t v_tb, int64_t v_tsize, int64_t v_n, int64_t v_lo, int64_t v_hi, int64_t v_sc) {
    if ((mfl_g_quant4 == 1)) {
        ({ int64_t _sq130_0 = v_xout; int64_t _sq130_1 = v_xq; int64_t _sq130_2 = v_xs; int64_t _sq130_3 = v_tb; int64_t _sq130_4 = ({ int64_t _sq129_0 = v_tb; int64_t _sq129_1 = mfl_wsz_0(v_tsize); (_sq129_0 + _sq129_1); }); int64_t _sq130_5 = v_n; int64_t _sq130_6 = v_lo; int64_t _sq130_7 = v_hi; int64_t _sq130_8 = v_sc; mfl_matmul_q4_0(_sq130_0, _sq130_1, _sq130_2, _sq130_3, _sq130_4, _sq130_5, _sq130_6, _sq130_7, _sq130_8); });
    } else {
        ({ int64_t _sq132_0 = v_xout; int64_t _sq132_1 = v_xq; int64_t _sq132_2 = v_xs; int64_t _sq132_3 = v_tb; int64_t _sq132_4 = ({ int64_t _sq131_0 = v_tb; int64_t _sq131_1 = mfl_wsz_0(v_tsize); (_sq131_0 + _sq131_1); }); int64_t _sq132_5 = v_n; int64_t _sq132_6 = v_lo; int64_t _sq132_7 = v_hi; mfl_matmul_q_0(_sq132_0, _sq132_1, _sq132_2, _sq132_3, _sq132_4, _sq132_5, _sq132_6, _sq132_7); });
    }
}

void mfl_matmul_q4_0(int64_t v_xout, int64_t v_xq, int64_t v_xs, int64_t v_wq, int64_t v_ws, int64_t v_n, int64_t v_lo, int64_t v_hi, int64_t v_sc) {
    int64_t v_i = 0;
    int64_t v_in = 0;
    v_i = v_lo;
    while ((v_i < v_hi)) {
        v_in = (v_i * v_n);
        ({ int64_t _sq133_0 = v_xout; int64_t _sq133_1 = (v_i * 4); double _sq133_2 = mfl_dot_q4(v_xq, v_xs, ((mfl_g_wbuf + v_wq) + (v_i * (v_n / 2))), ((mfl_g_wbuf + v_ws) + ((v_in / mfl_g_gs) * 4)), v_n, mfl_g_gs); mfl_poke_f32(_sq133_0, _sq133_1, _sq133_2); });
        v_i = (v_i + 1);
    }
}

void mfl_matmul_q_0(int64_t v_xout, int64_t v_xq, int64_t v_xs, int64_t v_wq, int64_t v_ws, int64_t v_n, int64_t v_lo, int64_t v_hi) {
    int64_t v_i = 0;
    int64_t v_in = 0;
    v_i = v_lo;
    while ((v_i < v_hi)) {
        v_in = (v_i * v_n);
        ({ int64_t _sq134_0 = v_xout; int64_t _sq134_1 = (v_i * 4); double _sq134_2 = mfl_dot_q8(v_xq, v_xs, ((mfl_g_wbuf + v_wq) + v_in), ((mfl_g_wbuf + v_ws) + ((v_in / mfl_g_gs) * 4)), v_n, mfl_g_gs); mfl_poke_f32(_sq134_0, _sq134_1, _sq134_2); });
        v_i = (v_i + 1);
    }
}

void mfl_attn_head_0(int64_t v_h, int64_t v_l, int64_t v_pos) {
    int64_t v_loff = 0;
    int64_t v_kv_mul = 0;
    int64_t v_qh = 0;
    int64_t v_atth = 0;
    int64_t v_t = 0;
    int64_t v_kt = 0;
    double v_score = 0.0;
    int64_t v_j = 0;
    int64_t v_xbh = 0;
    int64_t v_vt = 0;
    double v_a = 0.0;
    v_loff = ((v_l * mfl_g_seq_len) * mfl_g_kv_dim);
    v_kv_mul = (mfl_g_n_heads / mfl_g_n_kv_heads);
    v_qh = (mfl_g_s_q + ((v_h * mfl_g_head_size) * 4));
    v_atth = (mfl_g_s_att + ((v_h * mfl_g_seq_len) * 4));
    v_t = 0;
    while ((v_t <= v_pos)) {
        v_kt = (mfl_g_s_key_cache + ((((mfl_g_kv_off + v_loff) + (v_t * mfl_g_kv_dim)) + ((v_h / v_kv_mul) * mfl_g_head_size)) * 4));
        v_score = 0.0;
        v_j = 0;
        while ((v_j < mfl_g_head_size)) {
            v_score = ({ double _sq136_0 = v_score; double _sq136_1 = ({ double _sq135_0 = mfl_peek_f32(v_qh, (v_j * 4)); double _sq135_1 = mfl_peek_f32(v_kt, (v_j * 4)); (_sq135_0 * _sq135_1); }); (_sq136_0 + _sq136_1); });
            v_j = (v_j + 1);
        }
        ({ int64_t _sq138_0 = v_atth; int64_t _sq138_1 = (v_t * 4); double _sq138_2 = ({ double _sq137_0 = v_score; double _sq137_1 = mfl_math_sqrt((double)(((double)(mfl_g_head_size)))); (_sq137_0 / _sq137_1); }); mfl_poke_f32(_sq138_0, _sq138_1, _sq138_2); });
        v_t = (v_t + 1);
    }
    mfl_softmax_0(v_atth, (v_pos + 1));
    v_xbh = (mfl_g_s_xb + ((v_h * mfl_g_head_size) * 4));
    v_j = 0;
    while ((v_j < mfl_g_head_size)) {
        mfl_poke_f32(v_xbh, (v_j * 4), 0.0);
        v_j = (v_j + 1);
    }
    v_t = 0;
    while ((v_t <= v_pos)) {
        v_vt = (mfl_g_s_val_cache + ((((mfl_g_kv_off + v_loff) + (v_t * mfl_g_kv_dim)) + ((v_h / v_kv_mul) * mfl_g_head_size)) * 4));
        v_a = mfl_peek_f32(v_atth, (v_t * 4));
        v_j = 0;
        while ((v_j < mfl_g_head_size)) {
            ({ int64_t _sq141_0 = v_xbh; int64_t _sq141_1 = (v_j * 4); double _sq141_2 = ({ double _sq140_0 = mfl_peek_f32(v_xbh, (v_j * 4)); double _sq140_1 = ({ double _sq139_0 = v_a; double _sq139_1 = mfl_peek_f32(v_vt, (v_j * 4)); (_sq139_0 * _sq139_1); }); (_sq140_0 + _sq140_1); }); mfl_poke_f32(_sq141_0, _sq141_1, _sq141_2); });
            v_j = (v_j + 1);
        }
        v_t = (v_t + 1);
    }
}

void mfl_softmax_0(int64_t v_x, int64_t v_size) {
    double v_mx = 0.0;
    int64_t v_i = 0;
    double v_v = 0.0;
    double v_sum = 0.0;
    double v_e = 0.0;
    v_mx = mfl_peek_f32(v_x, 0);
    v_i = 1;
    while ((v_i < v_size)) {
        v_v = mfl_peek_f32(v_x, (v_i * 4));
        if ((v_v > v_mx)) {
            v_mx = v_v;
        }
        v_i = (v_i + 1);
    }
    v_sum = 0.0;
    v_i = 0;
    while ((v_i < v_size)) {
        v_e = mfl_math_exp((double)(({ double _sq142_0 = mfl_peek_f32(v_x, (v_i * 4)); double _sq142_1 = v_mx; (_sq142_0 - _sq142_1); })));
        mfl_poke_f32(v_x, (v_i * 4), v_e);
        v_sum = (v_sum + v_e);
        v_i = (v_i + 1);
    }
    v_i = 0;
    while ((v_i < v_size)) {
        ({ int64_t _sq144_0 = v_x; int64_t _sq144_1 = (v_i * 4); double _sq144_2 = ({ double _sq143_0 = mfl_peek_f32(v_x, (v_i * 4)); double _sq143_1 = v_sum; (_sq143_0 / _sq143_1); }); mfl_poke_f32(_sq144_0, _sq144_1, _sq144_2); });
        v_i = (v_i + 1);
    }
}

void mfl_matmul_q_batch_0(int64_t v_ob, int64_t v_ostride, int64_t v_xqb, int64_t v_xsb, int64_t v_wq, int64_t v_ws, int64_t v_n, int64_t v_lo, int64_t v_hi, int64_t v_B) {
    int64_t v_nsc = 0;
    int64_t v_i = 0;
    int64_t v_in = 0;
    int64_t v_wrow = 0;
    int64_t v_wsp = 0;
    int64_t v_b = 0;
    v_nsc = ((v_n / mfl_g_gs) * 4);
    v_i = v_lo;
    while ((v_i < v_hi)) {
        v_in = (v_i * v_n);
        v_wrow = ((mfl_g_wbuf + v_wq) + v_in);
        v_wsp = ((mfl_g_wbuf + v_ws) + ((v_in / mfl_g_gs) * 4));
        v_b = 0;
        while ((v_b < v_B)) {
            ({ int64_t _sq145_0 = v_ob; int64_t _sq145_1 = (((v_b * v_ostride) + v_i) * 4); double _sq145_2 = mfl_dot_q8((v_xqb + (v_b * v_n)), (v_xsb + (v_b * v_nsc)), v_wrow, v_wsp, v_n, mfl_g_gs); mfl_poke_f32(_sq145_0, _sq145_1, _sq145_2); });
            v_b = (v_b + 1);
        }
        v_i = (v_i + 1);
    }
}

void mfl_attn_head_at_0(int64_t v_h, int64_t v_l, int64_t v_pos, int64_t v_qbase, int64_t v_obase, int64_t v_att) {
    int64_t v_loff = 0;
    int64_t v_kv_mul = 0;
    int64_t v_qh = 0;
    int64_t v_t = 0;
    int64_t v_kt = 0;
    int64_t v_xbh = 0;
    int64_t v_j = 0;
    int64_t v_vt = 0;
    v_loff = ((v_l * mfl_g_seq_len) * mfl_g_kv_dim);
    v_kv_mul = (mfl_g_n_heads / mfl_g_n_kv_heads);
    v_qh = (v_qbase + ((v_h * mfl_g_head_size) * 4));
    v_t = 0;
    while ((v_t <= v_pos)) {
        v_kt = (mfl_g_s_key_cache + ((((mfl_g_kv_off + v_loff) + (v_t * mfl_g_kv_dim)) + ((v_h / v_kv_mul) * mfl_g_head_size)) * 4));
        ({ int64_t _sq147_0 = v_att; int64_t _sq147_1 = (v_t * 4); double _sq147_2 = ({ double _sq146_0 = mfl_dot_f32(v_qh, v_kt, mfl_g_head_size); double _sq146_1 = mfl_math_sqrt((double)(((double)(mfl_g_head_size)))); (_sq146_0 / _sq146_1); }); mfl_poke_f32(_sq147_0, _sq147_1, _sq147_2); });
        v_t = (v_t + 1);
    }
    mfl_softmax_0(v_att, (v_pos + 1));
    v_xbh = (v_obase + ((v_h * mfl_g_head_size) * 4));
    v_j = 0;
    while ((v_j < mfl_g_head_size)) {
        mfl_poke_f32(v_xbh, (v_j * 4), 0.0);
        v_j = (v_j + 1);
    }
    v_t = 0;
    while ((v_t <= v_pos)) {
        v_vt = (mfl_g_s_val_cache + ((((mfl_g_kv_off + v_loff) + (v_t * mfl_g_kv_dim)) + ((v_h / v_kv_mul) * mfl_g_head_size)) * 4));
        ({ int64_t _sq148_0 = v_xbh; double _sq148_1 = mfl_peek_f32(v_att, (v_t * 4)); int64_t _sq148_2 = v_vt; int64_t _sq148_3 = mfl_g_head_size; mfl_axpy_f32(_sq148_0, _sq148_1, _sq148_2, _sq148_3); });
        v_t = (v_t + 1);
    }
}

void mfl_rmsnorm_0(int64_t v_dst, int64_t v_x, int64_t v_w, int64_t v_size) {
    double v_ss = 0.0;
    int64_t v_j = 0;
    double v_v = 0.0;
    v_ss = 0.0;
    v_j = 0;
    while ((v_j < v_size)) {
        v_v = mfl_peek_f32(v_x, (v_j * 4));
        v_ss = (v_ss + (v_v * v_v));
        v_j = (v_j + 1);
    }
    v_ss = ({ double _sq149_0 = v_ss; double _sq149_1 = ((double)(v_size)); (_sq149_0 / _sq149_1); });
    v_ss = ({ double _sq150_0 = 1.0; double _sq150_1 = mfl_math_sqrt((double)((v_ss + 1e-05))); (_sq150_0 / _sq150_1); });
    v_j = 0;
    while ((v_j < v_size)) {
        ({ int64_t _sq153_0 = v_dst; int64_t _sq153_1 = (v_j * 4); double _sq153_2 = ({ double _sq152_0 = ({ double _sq151_0 = mfl_peek_f32(v_w, (v_j * 4)); double _sq151_1 = v_ss; (_sq151_0 * _sq151_1); }); double _sq152_1 = mfl_peek_f32(v_x, (v_j * 4)); (_sq152_0 * _sq152_1); }); mfl_poke_f32(_sq153_0, _sq153_1, _sq153_2); });
        v_j = (v_j + 1);
    }
}

void mfl_quantize_act_0(int64_t v_xq, int64_t v_xs, int64_t v_x, int64_t v_n) {
    int64_t v_ng = 0;
    int64_t v_g = 0;
    int64_t v_base = 0;
    double v_wmax = 0.0;
    int64_t v_i = 0;
    double v_v = 0.0;
    double v_scale = 0.0;
    int64_t v_q = 0;
    v_ng = (v_n / mfl_g_gs);
    v_g = 0;
    while ((v_g < v_ng)) {
        v_base = (v_g * mfl_g_gs);
        v_wmax = 0.0;
        v_i = 0;
        while ((v_i < mfl_g_gs)) {
            v_v = mfl_math_fabs((double)(mfl_peek_f32(v_x, ((v_base + v_i) * 4))));
            if ((v_v > v_wmax)) {
                v_wmax = v_v;
            }
            v_i = (v_i + 1);
        }
        v_scale = (v_wmax / 127.0);
        mfl_poke_f32(v_xs, (v_g * 4), v_scale);
        v_i = 0;
        while ((v_i < mfl_g_gs)) {
            v_q = 0;
            if ((v_scale > 0.0)) {
                v_q = ((int64_t)(mfl_math_round((double)(({ double _sq154_0 = mfl_peek_f32(v_x, ((v_base + v_i) * 4)); double _sq154_1 = v_scale; (_sq154_0 / _sq154_1); })))));
            }
            mfl_poke_u8(v_xq, (v_base + v_i), (v_q & 255));
            v_i = (v_i + 1);
        }
        v_g = (v_g + 1);
    }
}

void mfl_rope_at_0(int64_t v_l, int64_t v_pos, int64_t v_qbase) {
    int64_t v_kptr = 0;
    int64_t v_i = 0;
    int64_t v_hd = 0;
    double v_freq = 0.0;
    double v_val = 0.0;
    double v_fcr = 0.0;
    double v_fci = 0.0;
    double v_q0 = 0.0;
    double v_q1 = 0.0;
    double v_k0 = 0.0;
    double v_k1 = 0.0;
    v_kptr = (mfl_g_s_key_cache + (((mfl_g_kv_off + ((v_l * mfl_g_seq_len) * mfl_g_kv_dim)) + (v_pos * mfl_g_kv_dim)) * 4));
    v_i = 0;
    while ((v_i < mfl_g_dim)) {
        v_hd = (v_i % mfl_g_head_size);
        v_freq = mfl_peek_f32(mfl_g_rope_freqs, ((v_hd / 2) * 4));
        v_val = ({ double _sq155_0 = ((double)(v_pos)); double _sq155_1 = v_freq; (_sq155_0 * _sq155_1); });
        v_fcr = mfl_math_cos((double)(v_val));
        v_fci = mfl_math_sin((double)(v_val));
        v_q0 = mfl_peek_f32(v_qbase, (v_i * 4));
        v_q1 = mfl_peek_f32(v_qbase, ((v_i * 4) + 4));
        mfl_poke_f32(v_qbase, (v_i * 4), ((v_q0 * v_fcr) - (v_q1 * v_fci)));
        mfl_poke_f32(v_qbase, ((v_i * 4) + 4), ((v_q0 * v_fci) + (v_q1 * v_fcr)));
        if ((v_i < mfl_g_kv_dim)) {
            v_k0 = mfl_peek_f32(v_kptr, (v_i * 4));
            v_k1 = mfl_peek_f32(v_kptr, ((v_i * 4) + 4));
            mfl_poke_f32(v_kptr, (v_i * 4), ((v_k0 * v_fcr) - (v_k1 * v_fci)));
            mfl_poke_f32(v_kptr, ((v_i * 4) + 4), ((v_k0 * v_fci) + (v_k1 * v_fcr)));
        }
        v_i = (v_i + 2);
    }
}

void mfl_attn_head_cb_0(int64_t v_jj, int64_t v_hh, int64_t v_l, int64_t v_att) {
    int64_t v_s = 0;
    int64_t v_pos = 0;
    int64_t v_loff = 0;
    int64_t v_kv_mul = 0;
    int64_t v_qh = 0;
    int64_t v_t = 0;
    int64_t v_kt = 0;
    int64_t v_xbh = 0;
    int64_t v_j = 0;
    int64_t v_vt = 0;
    v_s = mfl_peek_i32(mfl_g_cb_active, (v_jj * 4));
    v_pos = mfl_peek_i32(mfl_g_cb_pos, (v_s * 4));
    v_loff = ((v_s * mfl_g_slotstride) + ((v_l * mfl_g_seq_len) * mfl_g_kv_dim));
    v_kv_mul = (mfl_g_n_heads / mfl_g_n_kv_heads);
    v_qh = ((mfl_g_pb_q + ((v_jj * mfl_g_dim) * 4)) + ((v_hh * mfl_g_head_size) * 4));
    v_t = 0;
    while ((v_t <= v_pos)) {
        v_kt = (mfl_g_s_key_cache + (((v_loff + (v_t * mfl_g_kv_dim)) + ((v_hh / v_kv_mul) * mfl_g_head_size)) * 4));
        ({ int64_t _sq157_0 = v_att; int64_t _sq157_1 = (v_t * 4); double _sq157_2 = ({ double _sq156_0 = mfl_dot_f32(v_qh, v_kt, mfl_g_head_size); double _sq156_1 = mfl_math_sqrt((double)(((double)(mfl_g_head_size)))); (_sq156_0 / _sq156_1); }); mfl_poke_f32(_sq157_0, _sq157_1, _sq157_2); });
        v_t = (v_t + 1);
    }
    mfl_softmax_0(v_att, (v_pos + 1));
    v_xbh = ((mfl_g_pb_xb + ((v_jj * mfl_g_dim) * 4)) + ((v_hh * mfl_g_head_size) * 4));
    v_j = 0;
    while ((v_j < mfl_g_head_size)) {
        mfl_poke_f32(v_xbh, (v_j * 4), 0.0);
        v_j = (v_j + 1);
    }
    v_t = 0;
    while ((v_t <= v_pos)) {
        v_vt = (mfl_g_s_val_cache + (((v_loff + (v_t * mfl_g_kv_dim)) + ((v_hh / v_kv_mul) * mfl_g_head_size)) * 4));
        ({ int64_t _sq158_0 = v_xbh; double _sq158_1 = mfl_peek_f32(v_att, (v_t * 4)); int64_t _sq158_2 = v_vt; int64_t _sq158_3 = mfl_g_head_size; mfl_axpy_f32(_sq158_0, _sq158_1, _sq158_2, _sq158_3); });
        v_t = (v_t + 1);
    }
}

void mfl_copy_f32_0(int64_t v_dst, int64_t v_src, int64_t v_n) {
    int64_t v_i = 0;
    v_i = 0;
    while ((v_i < v_n)) {
        ({ int64_t _sq159_0 = v_dst; int64_t _sq159_1 = (v_i * 4); double _sq159_2 = mfl_peek_f32(v_src, (v_i * 4)); mfl_poke_f32(_sq159_0, _sq159_1, _sq159_2); });
        v_i = (v_i + 1);
    }
}

void mfl_rope_qk_0(int64_t v_pos, int64_t v_qbase, int64_t v_kdst) {
    int64_t v_i = 0;
    int64_t v_hd = 0;
    double v_freq = 0.0;
    double v_val = 0.0;
    double v_fcr = 0.0;
    double v_fci = 0.0;
    double v_q0 = 0.0;
    double v_q1 = 0.0;
    double v_k0 = 0.0;
    double v_k1 = 0.0;
    v_i = 0;
    while ((v_i < mfl_g_dim)) {
        v_hd = (v_i % mfl_g_head_size);
        v_freq = mfl_peek_f32(mfl_g_rope_freqs, ((v_hd / 2) * 4));
        v_val = ({ double _sq160_0 = ((double)(v_pos)); double _sq160_1 = v_freq; (_sq160_0 * _sq160_1); });
        v_fcr = mfl_math_cos((double)(v_val));
        v_fci = mfl_math_sin((double)(v_val));
        v_q0 = mfl_peek_f32(v_qbase, (v_i * 4));
        v_q1 = mfl_peek_f32(v_qbase, ((v_i * 4) + 4));
        mfl_poke_f32(v_qbase, (v_i * 4), ((v_q0 * v_fcr) - (v_q1 * v_fci)));
        mfl_poke_f32(v_qbase, ((v_i * 4) + 4), ((v_q0 * v_fci) + (v_q1 * v_fcr)));
        if ((v_i < mfl_g_kv_dim)) {
            v_k0 = mfl_peek_f32(v_kdst, (v_i * 4));
            v_k1 = mfl_peek_f32(v_kdst, ((v_i * 4) + 4));
            mfl_poke_f32(v_kdst, (v_i * 4), ((v_k0 * v_fcr) - (v_k1 * v_fci)));
            mfl_poke_f32(v_kdst, ((v_i * 4) + 4), ((v_k0 * v_fci) + (v_k1 * v_fcr)));
        }
        v_i = (v_i + 2);
    }
}

mfl_slice mfl_encode_text_0(char* v_text) {
    mfl_slice v_ids = {0};
    mfl_slice v_sub = {0};
    int64_t v_i = 0;
    if ((mfl_g_tok_l3 == 1)) {
        v_ids = (mfl_slice){0};
        v_ids = mfl_append(v_ids, &((int64_t[1]){128000})[0], sizeof(int64_t));
        v_sub = mfl_l3_encode_0(v_text);
        v_i = 0;
        while (({ int64_t _sq161_0 = v_i; int64_t _sq161_1 = ((v_sub).len); (_sq161_0 < _sq161_1); })) {
            v_ids = mfl_append(v_ids, &((int64_t[1]){((int64_t*)(v_sub).data)[v_i]})[0], sizeof(int64_t));
            v_i = (v_i + 1);
        }
        return v_ids;
    }
    v_ids = mfl_bpe_encode_0(v_text);
    return v_ids;
    return v_ids;
}

mfl_slice mfl_l3_encode_0(char* v_text) {
    mfl_slice v_ids = {0};
    mfl_slice v_parts = {0};
    int64_t v_i = 0;
    mfl_slice v_sub = {0};
    int64_t v_j = 0;
    v_ids = (mfl_slice){0};
    v_parts = mfl_l3_pretok_0(v_text);
    v_i = 0;
    while (({ int64_t _sq162_0 = v_i; int64_t _sq162_1 = ((v_parts).len); (_sq162_0 < _sq162_1); })) {
        v_sub = mfl_l3_bpe_piece_0(((char**)(v_parts).data)[v_i]);
        v_j = 0;
        while (({ int64_t _sq163_0 = v_j; int64_t _sq163_1 = ((v_sub).len); (_sq163_0 < _sq163_1); })) {
            v_ids = mfl_append(v_ids, &((int64_t[1]){((int64_t*)(v_sub).data)[v_j]})[0], sizeof(int64_t));
            v_j = (v_j + 1);
        }
        v_i = (v_i + 1);
    }
    return v_ids;
    return v_ids;
}

mfl_slice mfl_l3_pretok_0(char* v_text) {
    mfl_slice v_parts = {0};
    mfl_bytes v_b = {0};
    int64_t v_n = 0;
    int64_t v_i = 0;
    int64_t v_cl = 0;
    int64_t v_cp = 0;
    int64_t v_w = 0;
    int64_t v_j = 0;
    int64_t v_pfx = 0;
    int64_t v_c2 = 0;
    int64_t v_w2 = 0;
    int64_t v_k = 0;
    int64_t v_c3 = 0;
    int64_t v_w3 = 0;
    int64_t v_cnt = 0;
    int64_t v_has_nl = 0;
    int64_t v_last_nl_end = 0;
    int64_t v_trail = 0;
    int64_t v_m = 0;
    v_parts = (mfl_slice){0};
    v_b = mfl_bytes_from_str(v_text);
    v_n = ((v_b).len);
    v_i = 0;
    while ((v_i < v_n)) {
        v_cl = mfl_l3_contraction_len_0(v_b, v_i, v_n);
        if ((v_cl > 0)) {
            v_parts = ({ mfl_slice _sq164_0 = v_parts; char* _sq164_1 = mfl_bytes_str(mfl_bytes_sub(v_b, v_i, (v_i + v_cl))); mfl_append(_sq164_0, &((char*[1]){_sq164_1})[0], sizeof(char*)); });
            v_i = (v_i + v_cl);
            continue;
        }
        {
            mfl_cp_at_0_ret _t165 = mfl_cp_at_0(v_b, v_i);
            v_cp = _t165.r0;
            v_w = _t165.r1;
        }
        v_j = v_i;
        v_pfx = 0;
        if (({ int _sq170_0 = ({ int _sq169_0 = ({ int _sq168_0 = ({ int _sq166_0 = mfl_is_letter_0(v_cp); int _sq166_1 = 0; (_sq166_0 == _sq166_1); }); int _sq168_1 = ({ int _sq167_0 = mfl_is_digit_cp_0(v_cp); int _sq167_1 = 0; (_sq167_0 == _sq167_1); }); (_sq168_0 && _sq168_1); }); int _sq169_1 = (v_cp != 10); (_sq169_0 && _sq169_1); }); int _sq170_1 = (v_cp != 13); (_sq170_0 && _sq170_1); })) {
            v_j = (v_i + v_w);
            v_pfx = 1;
        }
        if ((v_j < v_n)) {
            {
                mfl_cp_at_0_ret _t171 = mfl_cp_at_0(v_b, v_j);
                v_c2 = _t171.r0;
                v_w2 = _t171.r1;
            }
            if (mfl_is_letter_0(v_c2)) {
                v_k = v_j;
                while ((v_k < v_n)) {
                    {
                        mfl_cp_at_0_ret _t172 = mfl_cp_at_0(v_b, v_k);
                        v_c3 = _t172.r0;
                        v_w3 = _t172.r1;
                    }
                    if (({ int _sq173_0 = mfl_is_letter_0(v_c3); int _sq173_1 = 0; (_sq173_0 == _sq173_1); })) {
                        break;
                    }
                    v_k = (v_k + v_w3);
                    v_w2 = v_w2;
                }
                v_parts = ({ mfl_slice _sq174_0 = v_parts; char* _sq174_1 = mfl_bytes_str(mfl_bytes_sub(v_b, v_i, v_k)); mfl_append(_sq174_0, &((char*[1]){_sq174_1})[0], sizeof(char*)); });
                v_i = v_k;
                continue;
            }
        }
        if ((v_pfx == 1)) {
            v_j = v_i;
        }
        if (mfl_is_digit_cp_0(v_cp)) {
            v_k = v_i;
            v_cnt = 0;
            while (((v_k < v_n) && (v_cnt < 3))) {
                {
                    mfl_cp_at_0_ret _t175 = mfl_cp_at_0(v_b, v_k);
                    v_c3 = _t175.r0;
                    v_w3 = _t175.r1;
                }
                if (({ int _sq176_0 = mfl_is_digit_cp_0(v_c3); int _sq176_1 = 0; (_sq176_0 == _sq176_1); })) {
                    break;
                }
                v_k = (v_k + v_w3);
                v_cnt = (v_cnt + 1);
            }
            v_parts = ({ mfl_slice _sq177_0 = v_parts; char* _sq177_1 = mfl_bytes_str(mfl_bytes_sub(v_b, v_i, v_k)); mfl_append(_sq177_0, &((char*[1]){_sq177_1})[0], sizeof(char*)); });
            v_i = v_k;
            continue;
        }
        v_j = v_i;
        if ((v_cp == 32)) {
            v_j = (v_i + 1);
        }
        if ((v_j < v_n)) {
            {
                mfl_cp_at_0_ret _t178 = mfl_cp_at_0(v_b, v_j);
                v_c2 = _t178.r0;
                v_w2 = _t178.r1;
            }
            if (({ int _sq183_0 = ({ int _sq181_0 = ({ int _sq179_0 = mfl_is_space_cp_0(v_c2); int _sq179_1 = 0; (_sq179_0 == _sq179_1); }); int _sq181_1 = ({ int _sq180_0 = mfl_is_letter_0(v_c2); int _sq180_1 = 0; (_sq180_0 == _sq180_1); }); (_sq181_0 && _sq181_1); }); int _sq183_1 = ({ int _sq182_0 = mfl_is_digit_cp_0(v_c2); int _sq182_1 = 0; (_sq182_0 == _sq182_1); }); (_sq183_0 && _sq183_1); })) {
                v_k = v_j;
                while ((v_k < v_n)) {
                    {
                        mfl_cp_at_0_ret _t184 = mfl_cp_at_0(v_b, v_k);
                        v_c3 = _t184.r0;
                        v_w3 = _t184.r1;
                    }
                    if (({ int _sq186_0 = ({ int _sq185_0 = mfl_is_space_cp_0(v_c3); int _sq185_1 = mfl_is_letter_0(v_c3); (_sq185_0 || _sq185_1); }); int _sq186_1 = mfl_is_digit_cp_0(v_c3); (_sq186_0 || _sq186_1); })) {
                        break;
                    }
                    v_k = (v_k + v_w3);
                }
                while (({ int _sq190_0 = (v_k < v_n); int _sq190_1 = ({ int _sq189_0 = ({ int64_t _sq187_0 = mfl_byte_at(v_b, v_k); int64_t _sq187_1 = 10; (_sq187_0 == _sq187_1); }); int _sq189_1 = ({ int64_t _sq188_0 = mfl_byte_at(v_b, v_k); int64_t _sq188_1 = 13; (_sq188_0 == _sq188_1); }); (_sq189_0 || _sq189_1); }); (_sq190_0 && _sq190_1); })) {
                    v_k = (v_k + 1);
                }
                v_parts = ({ mfl_slice _sq191_0 = v_parts; char* _sq191_1 = mfl_bytes_str(mfl_bytes_sub(v_b, v_i, v_k)); mfl_append(_sq191_0, &((char*[1]){_sq191_1})[0], sizeof(char*)); });
                v_i = v_k;
                continue;
            }
        }
        if (mfl_is_space_cp_0(v_cp)) {
            v_k = v_i;
            v_has_nl = 0;
            v_last_nl_end = v_i;
            while ((v_k < v_n)) {
                {
                    mfl_cp_at_0_ret _t192 = mfl_cp_at_0(v_b, v_k);
                    v_c3 = _t192.r0;
                    v_w3 = _t192.r1;
                }
                if (({ int _sq193_0 = mfl_is_space_cp_0(v_c3); int _sq193_1 = 0; (_sq193_0 == _sq193_1); })) {
                    break;
                }
                v_k = (v_k + v_w3);
                if (((v_c3 == 10) || (v_c3 == 13))) {
                    v_has_nl = 1;
                    v_last_nl_end = v_k;
                }
            }
            if (((v_has_nl == 1) && (v_last_nl_end > v_i))) {
                v_trail = 1;
                v_m = v_last_nl_end;
                while ((v_m < v_k)) {
                    v_m = (v_m + 1);
                }
                v_parts = ({ mfl_slice _sq194_0 = v_parts; char* _sq194_1 = mfl_bytes_str(mfl_bytes_sub(v_b, v_i, v_last_nl_end)); mfl_append(_sq194_0, &((char*[1]){_sq194_1})[0], sizeof(char*)); });
                v_i = v_last_nl_end;
                v_trail = v_trail;
                continue;
            }
            if (((v_k < v_n) && ((v_k - v_i) > 1))) {
                v_parts = ({ mfl_slice _sq195_0 = v_parts; char* _sq195_1 = mfl_bytes_str(mfl_bytes_sub(v_b, v_i, (v_k - 1))); mfl_append(_sq195_0, &((char*[1]){_sq195_1})[0], sizeof(char*)); });
                v_i = (v_k - 1);
                continue;
            }
            if ((v_k == v_n)) {
                v_parts = ({ mfl_slice _sq196_0 = v_parts; char* _sq196_1 = mfl_bytes_str(mfl_bytes_sub(v_b, v_i, v_k)); mfl_append(_sq196_0, &((char*[1]){_sq196_1})[0], sizeof(char*)); });
                v_i = v_k;
                continue;
            }
            v_parts = ({ mfl_slice _sq197_0 = v_parts; char* _sq197_1 = mfl_bytes_str(mfl_bytes_sub(v_b, v_i, v_k)); mfl_append(_sq197_0, &((char*[1]){_sq197_1})[0], sizeof(char*)); });
            v_i = v_k;
            continue;
        }
        v_parts = ({ mfl_slice _sq198_0 = v_parts; char* _sq198_1 = mfl_bytes_str(mfl_bytes_sub(v_b, v_i, (v_i + v_w))); mfl_append(_sq198_0, &((char*[1]){_sq198_1})[0], sizeof(char*)); });
        v_i = (v_i + v_w);
    }
    return v_parts;
    return v_parts;
}

int64_t mfl_l3_contraction_len_0(mfl_bytes v_b, int64_t v_i, int64_t v_n) {
    int64_t v_l = 0;
    int64_t v_c1 = 0;
    int64_t v_c2 = 0;
    v_l = 0;
    if (({ int64_t _sq199_0 = mfl_byte_at(v_b, v_i); int64_t _sq199_1 = 39; (_sq199_0 != _sq199_1); })) {
        return v_l;
    }
    if (((v_i + 1) >= v_n)) {
        return v_l;
    }
    v_c1 = ({ int64_t _sq200_0 = mfl_byte_at(v_b, (v_i + 1)); int64_t _sq200_1 = 32; (_sq200_0 | _sq200_1); });
    if (((((v_c1 == 115) || (v_c1 == 116)) || (v_c1 == 109)) || (v_c1 == 100))) {
        v_l = 2;
        return v_l;
    }
    if (((v_i + 2) < v_n)) {
        v_c2 = ({ int64_t _sq201_0 = mfl_byte_at(v_b, (v_i + 2)); int64_t _sq201_1 = 32; (_sq201_0 | _sq201_1); });
        if (((((v_c1 == 114) && (v_c2 == 101)) || ((v_c1 == 118) && (v_c2 == 101))) || ((v_c1 == 108) && (v_c2 == 108)))) {
            v_l = 3;
            return v_l;
        }
    }
    return v_l;
    return v_l;
}

mfl_cp_at_0_ret mfl_cp_at_0(mfl_bytes v_b, int64_t v_i) {
    int64_t v_cp = 0;
    int64_t v_n = 0;
    int64_t v_c = 0;
    v_c = mfl_byte_at(v_b, v_i);
    v_cp = v_c;
    v_n = 1;
    if ((v_c < 128)) {
        return (mfl_cp_at_0_ret){ v_cp, v_n };
    }
    if (((v_c & 224) == 192)) {
        v_cp = ({ int64_t _sq203_0 = ((v_c & 31) << 6); int64_t _sq203_1 = ({ int64_t _sq202_0 = mfl_byte_at(v_b, (v_i + 1)); int64_t _sq202_1 = 63; (_sq202_0 & _sq202_1); }); (_sq203_0 | _sq203_1); });
        v_n = 2;
        return (mfl_cp_at_0_ret){ v_cp, v_n };
    }
    if (((v_c & 240) == 224)) {
        v_cp = ({ int64_t _sq208_0 = ({ int64_t _sq206_0 = ((v_c & 15) << 12); int64_t _sq206_1 = ({ int64_t _sq205_0 = ({ int64_t _sq204_0 = mfl_byte_at(v_b, (v_i + 1)); int64_t _sq204_1 = 63; (_sq204_0 & _sq204_1); }); int64_t _sq205_1 = 6; (_sq205_0 << _sq205_1); }); (_sq206_0 | _sq206_1); }); int64_t _sq208_1 = ({ int64_t _sq207_0 = mfl_byte_at(v_b, (v_i + 2)); int64_t _sq207_1 = 63; (_sq207_0 & _sq207_1); }); (_sq208_0 | _sq208_1); });
        v_n = 3;
        return (mfl_cp_at_0_ret){ v_cp, v_n };
    }
    v_cp = ({ int64_t _sq216_0 = ({ int64_t _sq214_0 = ({ int64_t _sq211_0 = ((v_c & 7) << 18); int64_t _sq211_1 = ({ int64_t _sq210_0 = ({ int64_t _sq209_0 = mfl_byte_at(v_b, (v_i + 1)); int64_t _sq209_1 = 63; (_sq209_0 & _sq209_1); }); int64_t _sq210_1 = 12; (_sq210_0 << _sq210_1); }); (_sq211_0 | _sq211_1); }); int64_t _sq214_1 = ({ int64_t _sq213_0 = ({ int64_t _sq212_0 = mfl_byte_at(v_b, (v_i + 2)); int64_t _sq212_1 = 63; (_sq212_0 & _sq212_1); }); int64_t _sq213_1 = 6; (_sq213_0 << _sq213_1); }); (_sq214_0 | _sq214_1); }); int64_t _sq216_1 = ({ int64_t _sq215_0 = mfl_byte_at(v_b, (v_i + 3)); int64_t _sq215_1 = 63; (_sq215_0 & _sq215_1); }); (_sq216_0 | _sq216_1); });
    v_n = 4;
    return (mfl_cp_at_0_ret){ v_cp, v_n };
    return (mfl_cp_at_0_ret){ v_cp, v_n };
}

int mfl_is_letter_0(int64_t v_cp) {
    int v_r = 0;
    v_r = 0;
    if (((v_cp >= 65) && (v_cp <= 90))) {
        v_r = 1;
        return v_r;
    }
    if (((v_cp >= 97) && (v_cp <= 122))) {
        v_r = 1;
        return v_r;
    }
    if (((((v_cp >= 192) && (v_cp <= 591)) && (v_cp != 215)) && (v_cp != 247))) {
        v_r = 1;
        return v_r;
    }
    if (((v_cp >= 880) && (v_cp <= 1023))) {
        v_r = 1;
        return v_r;
    }
    if (((v_cp >= 1024) && (v_cp <= 1279))) {
        v_r = 1;
        return v_r;
    }
    if (((v_cp >= 1425) && (v_cp <= 1524))) {
        v_r = 1;
        return v_r;
    }
    if (((v_cp >= 1536) && (v_cp <= 1791))) {
        v_r = 1;
        return v_r;
    }
    if (((v_cp >= 12352) && (v_cp <= 12543))) {
        v_r = 1;
        return v_r;
    }
    if (((v_cp >= 19968) && (v_cp <= 40959))) {
        v_r = 1;
        return v_r;
    }
    if (((v_cp >= 44032) && (v_cp <= 55215))) {
        v_r = 1;
        return v_r;
    }
    return v_r;
    return v_r;
}

int mfl_is_digit_cp_0(int64_t v_cp) {
    int v_r = 0;
    v_r = 0;
    if (((v_cp >= 48) && (v_cp <= 57))) {
        v_r = 1;
    }
    return v_r;
    return v_r;
}

int mfl_is_space_cp_0(int64_t v_cp) {
    int v_r = 0;
    v_r = 0;
    if ((((((((v_cp == 32) || (v_cp == 9)) || (v_cp == 10)) || (v_cp == 13)) || (v_cp == 12)) || (v_cp == 11)) || (v_cp == 160))) {
        v_r = 1;
    }
    return v_r;
    return v_r;
}

mfl_slice mfl_l3_bpe_piece_0(char* v_piece) {
    mfl_slice v_ids = {0};
    mfl_bytes v_pb = {0};
    mfl_slice v_syms = {0};
    int64_t v_i = 0;
    int64_t v_best = 0;
    int64_t v_best_id = 0;
    char* v_m = "";
    int64_t v_id = 0;
    mfl_slice v_ns = {0};
    v_ids = (mfl_slice){0};
    v_pb = mfl_bytes_from_str(v_piece);
    v_syms = (mfl_slice){0};
    v_i = 0;
    while (({ int64_t _sq217_0 = v_i; int64_t _sq217_1 = ((v_pb).len); (_sq217_0 < _sq217_1); })) {
        v_syms = ({ mfl_slice _sq218_0 = v_syms; char* _sq218_1 = mfl_bytes_hex(mfl_bytes_sub(v_pb, v_i, (v_i + 1))); mfl_append(_sq218_0, &((char*[1]){_sq218_1})[0], sizeof(char*)); });
        v_i = (v_i + 1);
    }
    while (({ int64_t _sq219_0 = ((v_syms).len); int64_t _sq219_1 = 1; (_sq219_0 > _sq219_1); })) {
        v_best = (-1);
        v_best_id = 999999999;
        v_i = 0;
        while (({ int64_t _sq221_0 = v_i; int64_t _sq221_1 = ({ int64_t _sq220_0 = ((v_syms).len); int64_t _sq220_1 = 1; (_sq220_0 - _sq220_1); }); (_sq221_0 < _sq221_1); })) {
            v_m = mfl_cat(((char**)(v_syms).data)[v_i], ((char**)(v_syms).data)[(v_i + 1)]);
            if (mfl_map_has(mfl_g_l3_idx, 0, v_m)) {
                v_id = ({ int64_t _g; mfl_map_get(mfl_g_l3_idx, 0, v_m, &_g); _g; });
                if ((v_id < v_best_id)) {
                    v_best_id = v_id;
                    v_best = v_i;
                }
            }
            v_i = (v_i + 1);
        }
        if ((v_best < 0)) {
            break;
        }
        v_ns = (mfl_slice){0};
        v_i = 0;
        while (({ int64_t _sq222_0 = v_i; int64_t _sq222_1 = ((v_syms).len); (_sq222_0 < _sq222_1); })) {
            if ((v_i == v_best)) {
                v_ns = mfl_append(v_ns, &((char*[1]){mfl_cat(((char**)(v_syms).data)[v_i], ((char**)(v_syms).data)[(v_i + 1)])})[0], sizeof(char*));
                v_i = (v_i + 2);
            } else {
                v_ns = mfl_append(v_ns, &((char*[1]){((char**)(v_syms).data)[v_i]})[0], sizeof(char*));
                v_i = (v_i + 1);
            }
        }
        v_syms = v_ns;
    }
    v_i = 0;
    while (({ int64_t _sq223_0 = v_i; int64_t _sq223_1 = ((v_syms).len); (_sq223_0 < _sq223_1); })) {
        if (mfl_map_has(mfl_g_l3_idx, 0, ((char**)(v_syms).data)[v_i])) {
            v_ids = mfl_append(v_ids, &((int64_t[1]){({ int64_t _g; mfl_map_get(mfl_g_l3_idx, 0, ((char**)(v_syms).data)[v_i], &_g); _g; })})[0], sizeof(int64_t));
        }
        v_i = (v_i + 1);
    }
    return v_ids;
    return v_ids;
}

mfl_slice mfl_bpe_encode_0(char* v_text) {
    mfl_slice v_toks = {0};
    v_toks = mfl_bpe_encode_opts_0(v_text, 1, 1);
    return v_toks;
    return v_toks;
}

mfl_slice mfl_bpe_encode_opts_0(char* v_text, int64_t v_add_bos, int64_t v_add_prefix) {
    mfl_slice v_toks = {0};
    mfl_bytes v_b = {0};
    int64_t v_n = 0;
    int64_t v_i = 0;
    int64_t v_j = 0;
    char* v_c = "";
    int64_t v_k = 0;
    double v_best_score = 0.0;
    int64_t v_best_id = 0;
    int64_t v_best_idx = 0;
    int64_t v_ti = 0;
    char* v_merged = "";
    int64_t v_id = 0;
    double v_sc = 0.0;
    mfl_slice v_nt = {0};
    v_toks = (mfl_slice){0};
    if ((v_add_bos == 1)) {
        v_toks = mfl_append(v_toks, &((int64_t[1]){1})[0], sizeof(int64_t));
    }
    if (({ int64_t _sq224_0 = ((int64_t)strlen(v_text)); int64_t _sq224_1 = 0; (_sq224_0 == _sq224_1); })) {
        return v_toks;
    }
    if ((v_add_prefix == 1)) {
        v_toks = mfl_append(v_toks, &((int64_t[1]){({ int64_t _g; mfl_map_get(mfl_g_vocab_idx, 0, " ", &_g); _g; })})[0], sizeof(int64_t));
    }
    v_b = mfl_bytes_from_str(v_text);
    v_n = ((v_b).len);
    v_i = 0;
    while ((v_i < v_n)) {
        v_j = (v_i + 1);
        while (({ int _sq227_0 = (v_j < v_n); int _sq227_1 = ({ int64_t _sq226_0 = ({ int64_t _sq225_0 = mfl_byte_at(v_b, v_j); int64_t _sq225_1 = 192; (_sq225_0 & _sq225_1); }); int64_t _sq226_1 = 128; (_sq226_0 == _sq226_1); }); (_sq227_0 && _sq227_1); })) {
            v_j = (v_j + 1);
        }
        v_c = mfl_bytes_str(mfl_bytes_sub(v_b, v_i, v_j));
        if (mfl_map_has(mfl_g_vocab_idx, 0, v_c)) {
            v_toks = mfl_append(v_toks, &((int64_t[1]){({ int64_t _g; mfl_map_get(mfl_g_vocab_idx, 0, v_c, &_g); _g; })})[0], sizeof(int64_t));
        } else {
            v_k = v_i;
            while ((v_k < v_j)) {
                v_toks = ({ mfl_slice _sq229_0 = v_toks; int64_t _sq229_1 = ({ int64_t _sq228_0 = mfl_byte_at(v_b, v_k); int64_t _sq228_1 = 3; (_sq228_0 + _sq228_1); }); mfl_append(_sq229_0, &((int64_t[1]){_sq229_1})[0], sizeof(int64_t)); });
                v_k = (v_k + 1);
            }
        }
        v_i = v_j;
    }
    while (1) {
        v_best_score = (0.0 - 1e+09);
        v_best_id = (-1);
        v_best_idx = (-1);
        v_ti = 0;
        while (({ int64_t _sq231_0 = v_ti; int64_t _sq231_1 = ({ int64_t _sq230_0 = ((v_toks).len); int64_t _sq230_1 = 1; (_sq230_0 - _sq230_1); }); (_sq231_0 < _sq231_1); })) {
            v_merged = mfl_cat(((char**)(mfl_g_vocab).data)[((int64_t*)(v_toks).data)[v_ti]], ((char**)(mfl_g_vocab).data)[((int64_t*)(v_toks).data)[(v_ti + 1)]]);
            if (mfl_map_has(mfl_g_vocab_idx, 0, v_merged)) {
                v_id = ({ int64_t _g; mfl_map_get(mfl_g_vocab_idx, 0, v_merged, &_g); _g; });
                v_sc = mfl_peek_f32(mfl_g_tok_scores, (v_id * 4));
                if ((v_sc > v_best_score)) {
                    v_best_score = v_sc;
                    v_best_id = v_id;
                    v_best_idx = v_ti;
                }
            }
            v_ti = (v_ti + 1);
        }
        if ((v_best_idx == (-1))) {
            break;
        }
        v_nt = (mfl_slice){0};
        v_ti = 0;
        while (({ int64_t _sq232_0 = v_ti; int64_t _sq232_1 = ((v_toks).len); (_sq232_0 < _sq232_1); })) {
            if ((v_ti == v_best_idx)) {
                v_nt = mfl_append(v_nt, &((int64_t[1]){v_best_id})[0], sizeof(int64_t));
                v_ti = (v_ti + 2);
            } else {
                v_nt = mfl_append(v_nt, &((int64_t[1]){((int64_t*)(v_toks).data)[v_ti]})[0], sizeof(int64_t));
                v_ti = (v_ti + 1);
            }
        }
        v_toks = v_nt;
    }
    return v_toks;
    return v_toks;
}

void mfl_cb_prefill_0(int64_t v_sl, mfl_slice v_toks, int64_t v_np) {
    int64_t v_i = 0;
    mfl_g_kv_off = (v_sl * mfl_g_slotstride);
    v_i = 0;
    while ((v_i < v_np)) {
        mfl_forward_0(((int64_t*)(v_toks).data)[v_i], v_i);
        v_i = (v_i + 1);
    }
    mfl_g_kv_off = 0;
    mfl_poke_i32(mfl_g_cb_pos, (v_sl * 4), v_np);
    ({ int64_t _sq233_0 = mfl_g_cb_curtok; int64_t _sq233_1 = (v_sl * 4); int64_t _sq233_2 = mfl_argmax_logits_0(); mfl_poke_i32(_sq233_0, _sq233_1, _sq233_2); });
}

void mfl_forward_0(int64_t v_token, int64_t v_pos) {
    int64_t v_i = 0;
    int64_t v_l = 0;
    int64_t v_nl = 0;
    double v_maxg = 0.0;
    int64_t v_j = 0;
    double v_g = 0.0;
    double v_ag = 0.0;
    double v_thr = 0.0;
    int64_t v_below = 0;
    double v_keptmass = 0.0;
    double v_totmass = 0.0;
    double v_h = 0.0;
    double v_v = 0.0;
    mfl_embed_token_0(v_token);
    v_i = 0;
    v_l = 0;
    v_nl = mfl_g_n_layers;
    if (((mfl_g_exit_layer > 0) && (mfl_g_exit_layer < v_nl))) {
        v_nl = mfl_g_exit_layer;
    }
    while ((v_l < v_nl)) {
        mfl_rmsnorm_0(mfl_g_s_xb, mfl_g_s_x, ((mfl_g_wbuf + mfl_g_w_rms_att) + ((v_l * mfl_g_dim) * 4)), mfl_g_dim);
        mfl_quantize_act_0(mfl_g_s_xq, mfl_g_s_xq_s, mfl_g_s_xb, mfl_g_dim);
        mfl_dispatch_rows_0(9, v_l, v_pos, mfl_g_dim);
        mfl_barrier_0();
        mfl_rope_0(v_l, v_pos);
        mfl_dispatch_rows_0(8, v_l, v_pos, mfl_g_n_heads);
        mfl_barrier_0();
        mfl_quantize_act_0(mfl_g_s_xq, mfl_g_s_xq_s, mfl_g_s_xb, mfl_g_dim);
        mfl_dispatch_rows_0(3, v_l, v_pos, mfl_g_dim);
        mfl_barrier_0();
        v_i = 0;
        while ((v_i < mfl_g_dim)) {
            ({ int64_t _sq235_0 = mfl_g_s_x; int64_t _sq235_1 = (v_i * 4); double _sq235_2 = ({ double _sq234_0 = mfl_peek_f32(mfl_g_s_x, (v_i * 4)); double _sq234_1 = mfl_peek_f32(mfl_g_s_xb2, (v_i * 4)); (_sq234_0 + _sq234_1); }); mfl_poke_f32(_sq235_0, _sq235_1, _sq235_2); });
            v_i = (v_i + 1);
        }
        mfl_rmsnorm_0(mfl_g_s_xb, mfl_g_s_x, ((mfl_g_wbuf + mfl_g_w_rms_ffn) + ((v_l * mfl_g_dim) * 4)), mfl_g_dim);
        mfl_quantize_act_0(mfl_g_s_xq, mfl_g_s_xq_s, mfl_g_s_xb, mfl_g_dim);
        mfl_dispatch_rows_0(10, v_l, v_pos, mfl_g_hidden_dim);
        mfl_barrier_0();
        if ((mfl_g_sparse_probe == 1)) {
            v_maxg = 0.0;
            v_j = 0;
            while ((v_j < mfl_g_hidden_dim)) {
                v_g = mfl_peek_f32(mfl_g_s_hb, (v_j * 4));
                v_g = ({ double _sq238_0 = v_g; double _sq238_1 = ({ double _sq237_0 = 1.0; double _sq237_1 = ({ double _sq236_0 = 1.0; double _sq236_1 = mfl_math_exp((double)((0.0 - v_g))); (_sq236_0 + _sq236_1); }); (_sq237_0 / _sq237_1); }); (_sq238_0 * _sq238_1); });
                v_ag = mfl_math_fabs((double)(v_g));
                if ((v_ag > v_maxg)) {
                    v_maxg = v_ag;
                }
                v_j = (v_j + 1);
            }
            v_thr = (0.1 * v_maxg);
            v_below = 0;
            v_keptmass = 0.0;
            v_totmass = 0.0;
            v_j = 0;
            while ((v_j < mfl_g_hidden_dim)) {
                v_g = mfl_peek_f32(mfl_g_s_hb, (v_j * 4));
                v_g = ({ double _sq241_0 = v_g; double _sq241_1 = ({ double _sq240_0 = 1.0; double _sq240_1 = ({ double _sq239_0 = 1.0; double _sq239_1 = mfl_math_exp((double)((0.0 - v_g))); (_sq239_0 + _sq239_1); }); (_sq240_0 / _sq240_1); }); (_sq241_0 * _sq241_1); });
                v_h = mfl_math_fabs((double)(({ double _sq242_0 = v_g; double _sq242_1 = mfl_peek_f32(mfl_g_s_hb2, (v_j * 4)); (_sq242_0 * _sq242_1); })));
                v_totmass = (v_totmass + v_h);
                if (({ double _sq243_0 = mfl_math_fabs((double)(v_g)); double _sq243_1 = v_thr; (_sq243_0 < _sq243_1); })) {
                    v_below = (v_below + 1);
                } else {
                    v_keptmass = (v_keptmass + v_h);
                }
                v_j = (v_j + 1);
            }
            mfl_g_sp_below = (mfl_g_sp_below + v_below);
            mfl_g_sp_total = (mfl_g_sp_total + mfl_g_hidden_dim);
            if ((v_totmass > 0.0)) {
                mfl_g_sp_massk = (mfl_g_sp_massk + (v_keptmass / v_totmass));
                mfl_g_sp_massn = (mfl_g_sp_massn + 1);
            }
        }
        v_i = 0;
        while ((v_i < mfl_g_hidden_dim)) {
            v_v = mfl_peek_f32(mfl_g_s_hb, (v_i * 4));
            v_v = ({ double _sq246_0 = v_v; double _sq246_1 = ({ double _sq245_0 = 1.0; double _sq245_1 = ({ double _sq244_0 = 1.0; double _sq244_1 = mfl_math_exp((double)((0.0 - v_v))); (_sq244_0 + _sq244_1); }); (_sq245_0 / _sq245_1); }); (_sq246_0 * _sq246_1); });
            ({ int64_t _sq248_0 = mfl_g_s_hb; int64_t _sq248_1 = (v_i * 4); double _sq248_2 = ({ double _sq247_0 = v_v; double _sq247_1 = mfl_peek_f32(mfl_g_s_hb2, (v_i * 4)); (_sq247_0 * _sq247_1); }); mfl_poke_f32(_sq248_0, _sq248_1, _sq248_2); });
            v_i = (v_i + 1);
        }
        mfl_quantize_act_0(mfl_g_s_hq, mfl_g_s_hq_s, mfl_g_s_hb, mfl_g_hidden_dim);
        mfl_dispatch_rows_0(6, v_l, v_pos, mfl_g_dim);
        mfl_barrier_0();
        v_i = 0;
        while ((v_i < mfl_g_dim)) {
            ({ int64_t _sq250_0 = mfl_g_s_x; int64_t _sq250_1 = (v_i * 4); double _sq250_2 = ({ double _sq249_0 = mfl_peek_f32(mfl_g_s_x, (v_i * 4)); double _sq249_1 = mfl_peek_f32(mfl_g_s_xb, (v_i * 4)); (_sq249_0 + _sq249_1); }); mfl_poke_f32(_sq250_0, _sq250_1, _sq250_2); });
            v_i = (v_i + 1);
        }
        v_l = (v_l + 1);
    }
    mfl_rmsnorm_0(mfl_g_s_x, mfl_g_s_x, (mfl_g_wbuf + mfl_g_w_rms_final), mfl_g_dim);
    mfl_quantize_act_0(mfl_g_s_xq, mfl_g_s_xq_s, mfl_g_s_x, mfl_g_dim);
    mfl_dispatch_rows_0(7, 0, v_pos, mfl_g_vocab_size);
    mfl_barrier_0();
}

void mfl_embed_token_0(int64_t v_token) {
    int64_t v_row = 0;
    int64_t v_sbase = 0;
    int64_t v_h = 0;
    int64_t v_g = 0;
    int64_t v_gq = 0;
    double v_sc = 0.0;
    int64_t v_i = 0;
    int64_t v_b = 0;
    v_row = (v_token * mfl_g_dim);
    v_sbase = ({ int64_t _sq251_0 = mfl_g_w_tok; int64_t _sq251_1 = mfl_wsz_0((mfl_g_vocab_size * mfl_g_dim)); (_sq251_0 + _sq251_1); });
    if ((mfl_g_quant4 == 1)) {
        v_h = (mfl_g_gs / 2);
        v_g = 0;
        while ((v_g < (mfl_g_dim / mfl_g_gs))) {
            v_gq = ((mfl_g_w_tok + (v_token * (mfl_g_dim / 2))) + (v_g * v_h));
            v_sc = mfl_peek_f32(mfl_g_wbuf, (v_sbase + (((v_row + (v_g * mfl_g_gs)) / mfl_g_gs) * 4)));
            v_i = 0;
            while ((v_i < v_h)) {
                v_b = mfl_peek_u8(mfl_g_wbuf, (v_gq + v_i));
                ({ int64_t _sq253_0 = mfl_g_s_x; int64_t _sq253_1 = (((v_g * mfl_g_gs) + v_i) * 4); double _sq253_2 = ({ double _sq252_0 = ((double)(((v_b & 15) - 8))); double _sq252_1 = v_sc; (_sq252_0 * _sq252_1); }); mfl_poke_f32(_sq253_0, _sq253_1, _sq253_2); });
                ({ int64_t _sq255_0 = mfl_g_s_x; int64_t _sq255_1 = ((((v_g * mfl_g_gs) + v_i) + v_h) * 4); double _sq255_2 = ({ double _sq254_0 = ((double)(((v_b >> 4) - 8))); double _sq254_1 = v_sc; (_sq254_0 * _sq254_1); }); mfl_poke_f32(_sq255_0, _sq255_1, _sq255_2); });
                v_i = (v_i + 1);
            }
            v_g = (v_g + 1);
        }
        return;
    }
    v_i = 0;
    while ((v_i < mfl_g_dim)) {
        ({ int64_t _sq257_0 = mfl_g_s_x; int64_t _sq257_1 = (v_i * 4); double _sq257_2 = ({ double _sq256_0 = ((double)(mfl_peek_i8(mfl_g_wbuf, ((mfl_g_w_tok + v_row) + v_i)))); double _sq256_1 = mfl_peek_f32(mfl_g_wbuf, (v_sbase + (((v_row + v_i) / mfl_g_gs) * 4))); (_sq256_0 * _sq256_1); }); mfl_poke_f32(_sq257_0, _sq257_1, _sq257_2); });
        v_i = (v_i + 1);
    }
}

void mfl_dispatch_rows_0(int64_t v_t, int64_t v_l, int64_t v_pos, int64_t v_d) {
    int64_t v_nc = 0;
    int64_t v_chunk = 0;
    int64_t v_lo = 0;
    int64_t v_hi = 0;
    v_nc = mfl_g_n_workers;
    if ((v_nc > v_d)) {
        v_nc = v_d;
    }
    v_chunk = (((v_d + v_nc) - 1) / v_nc);
    v_lo = 0;
    while ((v_lo < v_d)) {
        v_hi = (v_lo + v_chunk);
        if ((v_hi > v_d)) {
            v_hi = v_d;
        }
        if ((mfl_g_pending >= mfl_g_n_workers)) {
            ({ int64_t _r; mfl_chan_recv(mfl_g_done, &_r); _r; });
            mfl_g_pending = (mfl_g_pending - 1);
        }
        mfl_chan_send(mfl_g_jobs, &((int64_t[1]){((((v_t | (v_l << 5)) | (v_pos << 13)) | (v_lo << 25)) | (v_hi << 44))})[0]);
        mfl_g_pending = (mfl_g_pending + 1);
        v_lo = v_hi;
    }
}

void mfl_barrier_0(void) {
    while ((mfl_g_pending > 0)) {
        ({ int64_t _r; mfl_chan_recv(mfl_g_done, &_r); _r; });
        mfl_g_pending = (mfl_g_pending - 1);
    }
}

void mfl_rope_0(int64_t v_l, int64_t v_pos) {
    int64_t v_kptr = 0;
    int64_t v_i = 0;
    int64_t v_hd = 0;
    double v_freq = 0.0;
    double v_val = 0.0;
    double v_fcr = 0.0;
    double v_fci = 0.0;
    double v_q0 = 0.0;
    double v_q1 = 0.0;
    double v_k0 = 0.0;
    double v_k1 = 0.0;
    v_kptr = (mfl_g_s_key_cache + (((mfl_g_kv_off + ((v_l * mfl_g_seq_len) * mfl_g_kv_dim)) + (v_pos * mfl_g_kv_dim)) * 4));
    v_i = 0;
    while ((v_i < mfl_g_dim)) {
        v_hd = (v_i % mfl_g_head_size);
        v_freq = mfl_peek_f32(mfl_g_rope_freqs, ((v_hd / 2) * 4));
        v_val = ({ double _sq258_0 = ((double)(v_pos)); double _sq258_1 = v_freq; (_sq258_0 * _sq258_1); });
        v_fcr = mfl_math_cos((double)(v_val));
        v_fci = mfl_math_sin((double)(v_val));
        v_q0 = mfl_peek_f32(mfl_g_s_q, (v_i * 4));
        v_q1 = mfl_peek_f32(mfl_g_s_q, ((v_i * 4) + 4));
        mfl_poke_f32(mfl_g_s_q, (v_i * 4), ((v_q0 * v_fcr) - (v_q1 * v_fci)));
        mfl_poke_f32(mfl_g_s_q, ((v_i * 4) + 4), ((v_q0 * v_fci) + (v_q1 * v_fcr)));
        if ((v_i < mfl_g_kv_dim)) {
            v_k0 = mfl_peek_f32(v_kptr, (v_i * 4));
            v_k1 = mfl_peek_f32(v_kptr, ((v_i * 4) + 4));
            mfl_poke_f32(v_kptr, (v_i * 4), ((v_k0 * v_fcr) - (v_k1 * v_fci)));
            mfl_poke_f32(v_kptr, ((v_i * 4) + 4), ((v_k0 * v_fci) + (v_k1 * v_fcr)));
        }
        v_i = (v_i + 2);
    }
}

int64_t mfl_argmax_logits_0(void) {
    int64_t v_best = 0;
    double v_bv = 0.0;
    int64_t v_i = 0;
    double v_v = 0.0;
    v_best = 0;
    v_bv = mfl_peek_f32(mfl_g_s_logits, 0);
    v_i = 1;
    while ((v_i < mfl_g_vocab_size)) {
        v_v = mfl_peek_f32(mfl_g_s_logits, (v_i * 4));
        if ((v_v > v_bv)) {
            v_bv = v_v;
            v_best = v_i;
        }
        v_i = (v_i + 1);
    }
    return v_best;
    return v_best;
}

void mfl_cb_step_0(int64_t v_M) {
    int64_t v_j = 0;
    int64_t v_l = 0;
    int64_t v_s = 0;
    mfl_g_pb_B = v_M;
    v_j = 0;
    while ((v_j < v_M)) {
        ({ int64_t _sq261_0 = ({ int64_t _sq260_0 = mfl_g_cb_curtok; int64_t _sq260_1 = ({ int64_t _sq259_0 = mfl_peek_i32(mfl_g_cb_active, (v_j * 4)); int64_t _sq259_1 = 4; (_sq259_0 * _sq259_1); }); mfl_peek_i32(_sq260_0, _sq260_1); }); int64_t _sq261_1 = (mfl_g_pb_x + ((v_j * mfl_g_dim) * 4)); mfl_embed_to_0(_sq261_0, _sq261_1); });
        v_j = (v_j + 1);
    }
    v_l = 0;
    while ((v_l < mfl_g_n_layers)) {
        mfl_dispatch_rows_0(19, v_l, 0, v_M);
        mfl_barrier_0();
        mfl_dispatch_rows_0(11, v_l, 0, mfl_g_dim);
        mfl_barrier_0();
        mfl_dispatch_rows_0(28, v_l, 0, mfl_g_kv_dim);
        mfl_barrier_0();
        mfl_dispatch_rows_0(29, v_l, 0, mfl_g_kv_dim);
        mfl_barrier_0();
        mfl_dispatch_rows_0(32, v_l, 0, v_M);
        mfl_barrier_0();
        mfl_dispatch_rows_0(30, v_l, 0, (v_M * mfl_g_n_heads));
        mfl_barrier_0();
        mfl_dispatch_rows_0(21, v_l, 0, v_M);
        mfl_barrier_0();
        mfl_dispatch_rows_0(14, v_l, 0, mfl_g_dim);
        mfl_barrier_0();
        mfl_dispatch_rows_0(22, v_l, 0, v_M);
        mfl_barrier_0();
        mfl_dispatch_rows_0(20, v_l, 0, v_M);
        mfl_barrier_0();
        mfl_dispatch_rows_0(15, v_l, 0, mfl_g_hidden_dim);
        mfl_barrier_0();
        mfl_dispatch_rows_0(16, v_l, 0, mfl_g_hidden_dim);
        mfl_barrier_0();
        mfl_dispatch_rows_0(23, v_l, 0, v_M);
        mfl_barrier_0();
        mfl_dispatch_rows_0(17, v_l, 0, mfl_g_dim);
        mfl_barrier_0();
        mfl_dispatch_rows_0(24, v_l, 0, v_M);
        mfl_barrier_0();
        v_l = (v_l + 1);
    }
    mfl_dispatch_rows_0(26, 0, 0, v_M);
    mfl_barrier_0();
    mfl_dispatch_rows_0(27, 0, 0, mfl_g_vocab_size);
    mfl_barrier_0();
    mfl_dispatch_rows_0(31, 0, 0, v_M);
    mfl_barrier_0();
    v_j = 0;
    while ((v_j < v_M)) {
        v_s = mfl_peek_i32(mfl_g_cb_active, (v_j * 4));
        ({ int64_t _sq262_0 = mfl_g_cb_curtok; int64_t _sq262_1 = (v_s * 4); int64_t _sq262_2 = mfl_peek_i32(mfl_g_cb_next, (v_j * 4)); mfl_poke_i32(_sq262_0, _sq262_1, _sq262_2); });
        ({ int64_t _sq264_0 = mfl_g_cb_pos; int64_t _sq264_1 = (v_s * 4); int64_t _sq264_2 = ({ int64_t _sq263_0 = mfl_peek_i32(mfl_g_cb_pos, (v_s * 4)); int64_t _sq263_1 = 1; (_sq263_0 + _sq263_1); }); mfl_poke_i32(_sq264_0, _sq264_1, _sq264_2); });
        v_j = (v_j + 1);
    }
}

void mfl_embed_to_0(int64_t v_token, int64_t v_dst) {
    int64_t v_row = 0;
    int64_t v_sbase = 0;
    int64_t v_i = 0;
    v_row = (v_token * mfl_g_dim);
    v_sbase = (mfl_g_w_tok + (mfl_g_vocab_size * mfl_g_dim));
    v_i = 0;
    while ((v_i < mfl_g_dim)) {
        ({ int64_t _sq266_0 = v_dst; int64_t _sq266_1 = (v_i * 4); double _sq266_2 = ({ double _sq265_0 = ((double)(mfl_peek_i8(mfl_g_wbuf, ((mfl_g_w_tok + v_row) + v_i)))); double _sq265_1 = mfl_peek_f32(mfl_g_wbuf, (v_sbase + (((v_row + v_i) / mfl_g_gs) * 4))); (_sq265_0 * _sq265_1); }); mfl_poke_f32(_sq266_0, _sq266_1, _sq266_2); });
        v_i = (v_i + 1);
    }
}

char* mfl_piece_text_0(int64_t v_prev, int64_t v_token) {
    char* v_s = "";
    if ((mfl_g_tok_l3 == 1)) {
        v_s = "";
        if ((v_token < 128000)) {
            v_s = ((char**)(mfl_g_vocab_l3).data)[v_token];
        }
        return v_s;
    }
    v_s = mfl_decode_piece_0(v_prev, v_token);
    return v_s;
    return v_s;
}

char* mfl_decode_piece_0(int64_t v_prev, int64_t v_token) {
    char* v_piece = "";
    v_piece = ((char**)(mfl_g_vocab).data)[v_token];
    if (({ int _sq267_0 = (v_prev == 1); int _sq267_1 = mfl_has_prefix(v_piece, " "); (_sq267_0 && _sq267_1); })) {
        v_piece = ({ char* _sq268_0 = v_piece; int64_t _sq268_1 = 1; int64_t _sq268_2 = ((int64_t)strlen(v_piece)); mfl_substr(_sq268_0, _sq268_1, _sq268_2); });
    }
    if (({ int _sq269_0 = mfl_has_prefix(v_piece, "<0x"); int _sq269_1 = mfl_has_suffix(v_piece, ">"); (_sq269_0 && _sq269_1); })) {
        v_piece = mfl_bytes_str(mfl_bytes_unhex(mfl_substr(v_piece, 3, 5)));
    }
    return v_piece;
    return v_piece;
}

__attribute__((constructor)) static void mfl_globals_init(void) {
    mfl_g_dim = 0;
    mfl_g_hidden_dim = 0;
    mfl_g_n_layers = 0;
    mfl_g_n_heads = 0;
    mfl_g_n_kv_heads = 0;
    mfl_g_vocab_size = 0;
    mfl_g_seq_len = 0;
    mfl_g_head_size = 0;
    mfl_g_kv_dim = 0;
    mfl_g_gs = 0;
    mfl_g_shared_cls = 0;
    mfl_g_quant4 = 0;
    mfl_g_wbuf = 0;
    mfl_g_w_rms_att = 0;
    mfl_g_w_rms_ffn = 0;
    mfl_g_w_rms_final = 0;
    mfl_g_w_tok = 0;
    mfl_g_wq_b = 0;
    mfl_g_wk_b = 0;
    mfl_g_wv_b = 0;
    mfl_g_wo_b = 0;
    mfl_g_w1_b = 0;
    mfl_g_w2_b = 0;
    mfl_g_w3_b = 0;
    mfl_g_wcls_b = 0;
    mfl_g_s_x = 0;
    mfl_g_s_xb = 0;
    mfl_g_s_xb2 = 0;
    mfl_g_s_hb = 0;
    mfl_g_s_hb2 = 0;
    mfl_g_s_q = 0;
    mfl_g_s_att = 0;
    mfl_g_s_logits = 0;
    mfl_g_s_key_cache = 0;
    mfl_g_s_val_cache = 0;
    mfl_g_kv_off = 0;
    mfl_g_slotstride = 0;
    mfl_g_cb_slots = 8;
    mfl_g_cb_active = 0;
    mfl_g_cb_pos = 0;
    mfl_g_cb_curtok = 0;
    mfl_g_cb_kt = 0;
    mfl_g_cb_vt = 0;
    mfl_g_cb_next = 0;
    mfl_g_s_xq = 0;
    mfl_g_s_xq_s = 0;
    mfl_g_s_hq = 0;
    mfl_g_s_hq_s = 0;
    mfl_g_vocab = (mfl_slice){0};
    mfl_g_vocab_idx = mfl_make_map(1, sizeof(int64_t));
    mfl_g_tok_scores = 0;
    mfl_g_pb_tile = 256;
    mfl_g_pb_enabled = 1;
    mfl_g_prof_qkv = 0;
    mfl_g_prof_attn = 0;
    mfl_g_prof_o = 0;
    mfl_g_prof_ffn = 0;
    mfl_g_pb_B = 0;
    mfl_g_pb_tstart = 0;
    mfl_g_pb_x = 0;
    mfl_g_pb_xb = 0;
    mfl_g_pb_xb2 = 0;
    mfl_g_pb_q = 0;
    mfl_g_pb_hb = 0;
    mfl_g_pb_hb2 = 0;
    mfl_g_pb_xq = 0;
    mfl_g_pb_xqs = 0;
    mfl_g_pb_hq = 0;
    mfl_g_pb_hqs = 0;
    mfl_g_n_workers = 1;
    mfl_g_jobs = mfl_make_chan(sizeof(int64_t), 0, 0, 0);
    mfl_g_done = mfl_make_chan(sizeof(int64_t), 0, 0, 0);
    mfl_g_pending = 0;
    mfl_g_rng_state = 12345;
    mfl_g_cache_buf = 0;
    mfl_g_cache_len = 0;
    mfl_g_spec_enabled = 0;
    mfl_g_sparse_probe = 0;
    mfl_g_exit_layer = 0;
    mfl_g_sp_below = 0;
    mfl_g_sp_total = 0;
    mfl_g_sp_massk = 0.0;
    mfl_g_sp_massn = 0;
    mfl_g_spec_k = 8;
    mfl_g_spec_maxb = 16;
    mfl_g_spec_logits = 0;
    mfl_g_spec_argmax = 0;
    mfl_g_spec_tbuf = 0;
    mfl_g_spec_steps = 0;
    mfl_g_spec_tokens = 0;
    mfl_g_gen_prefill_reused = 0;
    mfl_g_rope_freqs = 0;
    mfl_g_rope_theta = 10000.0;
    mfl_g_l3_scaling = 0;
    mfl_g_l3_factor = 1.0;
    mfl_g_l3_low = 1.0;
    mfl_g_l3_high = 1.0;
    mfl_g_l3_orig = 8192;
    mfl_g_tok_l3 = 0;
    mfl_g_vocab_l3 = (mfl_slice){0};
    mfl_g_l3_idx = mfl_make_map(1, sizeof(int64_t));
}
int main(int argc, char** argv) { signal(SIGPIPE, SIG_IGN); mfl_argc = argc; mfl_argv = argv; mfl_main(); return 0; }
