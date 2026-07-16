#!/usr/bin/env python3
# Generate a TINY synthetic MoE transformer + a numpy reference token stream.
# Purpose: prove the pure-MFL MoE forward (router -> top-k -> SwiGLU experts ->
# weighted sum) + expert-STREAMING is token-identical, with zero model download.
#
# Format (little-endian):
#   header: 12 int32 = magic, D, n_heads, n_kv_heads, head_dim, I, E, K, L, vocab, seq_len, norm_topk
#   dense (fp32), in order:
#     embed[vocab*D]
#     per layer L: in_ln[D], q[D*D], k[kv*D], v[kv*D], o[D*D], post_ln[D], gate[E*D]
#     final_norm[D], lm_head[vocab*D]
#   experts (fp32), the STREAMED region, per (layer,expert):
#     gate_proj[I*D], up_proj[I*D], down_proj[D*I]
# Weight convention: [out,in] row-major; y[o]=sum_i W[o,i]*x[i].
import numpy as np, struct, json, sys

np.random.seed(1234)
D, NH, NKV, HD = 32, 4, 4, 8       # hidden, heads, kv-heads (no GQA here), head_dim
I, E, K, L = 64, 8, 2, 2           # inter, n_experts, top-k, layers
VOCAB, SEQ, NORM_TOPK = 48, 64, 1
KV = NKV * HD
THETA, EPS = 10000.0, 1e-5
MAGIC = 0x4D6F4531                 # "MoE1"

def rnd(*shape): return (np.random.randn(*shape) * 0.08).astype(np.float32)

embed   = rnd(VOCAB, D)
lm_head = rnd(VOCAB, D)
fnorm   = (np.ones(D) + rnd(D)*0.1).astype(np.float32)
layers = []
experts = []   # [(l,e)] -> (gate_proj, up_proj, down_proj)
for l in range(L):
    ly = dict(
        in_ln=(np.ones(D)+rnd(D)*0.1).astype(np.float32),
        q=rnd(D, D), k=rnd(KV, D), v=rnd(KV, D), o=rnd(D, D),
        post_ln=(np.ones(D)+rnd(D)*0.1).astype(np.float32),
        gate=rnd(E, D),
    )
    layers.append(ly)
    for e in range(E):
        experts.append((rnd(I, D), rnd(I, D), rnd(D, I)))

def rmsnorm(x, w):
    ss = np.sqrt(np.mean(x*x) + EPS)
    return (x / ss) * w

def silu(x): return x / (1.0 + np.exp(-x))

def rope(vec, pos):  # vec [n_heads*head_dim], rotate pairs within each head
    out = vec.copy()
    for i in range(0, len(vec), 2):
        hd = i % HD
        freq = THETA ** (-(hd) / HD)   # pair index hd/2 -> exponent hd/HD
        val = pos * freq
        c, s = np.cos(val), np.sin(val)
        out[i]   = vec[i]*c - vec[i+1]*s
        out[i+1] = vec[i]*s + vec[i+1]*c
    return out

def forward(x, pos, kc, vc):
    for li in range(L):
        ly = layers[li]
        h = rmsnorm(x, ly['in_ln'])
        q = ly['q'] @ h
        k = ly['k'] @ h
        v = ly['v'] @ h
        q = rope(q, pos); k = rope(k, pos)
        kc[li][pos] = k; vc[li][pos] = v
        attn = np.zeros(D, dtype=np.float32)
        for hh in range(NH):
            qh = q[hh*HD:(hh+1)*HD]
            sc = np.array([qh @ kc[li][t][hh*HD:(hh+1)*HD] for t in range(pos+1)]) / np.sqrt(HD)
            sc = np.exp(sc - sc.max()); sc /= sc.sum()
            oh = sum(sc[t] * vc[li][t][hh*HD:(hh+1)*HD] for t in range(pos+1))
            attn[hh*HD:(hh+1)*HD] = oh
        x = x + ly['o'] @ attn
        # MoE
        h2 = rmsnorm(x, ly['post_ln'])
        logits = ly['gate'] @ h2
        pr = np.exp(logits - logits.max()); pr /= pr.sum()
        idx = np.argsort(-pr)[:K]
        w = pr[idx].copy()
        if NORM_TOPK: w = w / w.sum()
        moe = np.zeros(D, dtype=np.float32)
        for kk in range(K):
            gp, up, dn = experts[li*E + idx[kk]]
            g = silu(gp @ h2); u = up @ h2
            moe += w[kk] * (dn @ (g*u))
        x = x + moe
    h = rmsnorm(x, fnorm)
    return lm_head @ h

def generate(prompt, n_new):
    kc = [np.zeros((SEQ, KV), np.float32) for _ in range(L)]
    vc = [np.zeros((SEQ, KV), np.float32) for _ in range(L)]
    toks = list(prompt); pos = 0
    for t in prompt[:-1]:
        forward(embed[t], pos, kc, vc); pos += 1
    cur = prompt[-1]
    out = []
    for _ in range(n_new):
        lg = forward(embed[cur], pos, kc, vc); pos += 1
        cur = int(np.argmax(lg)); out.append(cur); toks.append(cur)
    return out

prompt = [1, 2, 3, 4, 5]
ref = generate(prompt, 12)
print("ref tokens:", ref)

# write model
out = sys.argv[1] if len(sys.argv) > 1 else "models/moe-spike.bin"
with open(out, "wb") as f:
    f.write(struct.pack("<12i", MAGIC, D, NH, NKV, HD, I, E, K, L, VOCAB, SEQ, NORM_TOPK))
    def wr(a): f.write(a.astype("<f4").tobytes())
    wr(embed)
    for ly in layers:
        wr(ly['in_ln']); wr(ly['q']); wr(ly['k']); wr(ly['v']); wr(ly['o']); wr(ly['post_ln']); wr(ly['gate'])
    wr(fnorm); wr(lm_head)
    for li in range(L):
        for e in range(E):
            gp, up, dn = experts[li*E + e]
            wr(gp); wr(up); wr(dn)
json.dump({"prompt": prompt, "ref": ref}, open(out + ".ref.json", "w"))
print("wrote", out, "and", out + ".ref.json")
