#!/usr/bin/env python3
# Ground-truth OLMoE-1B-7B forward in numpy (fp32), reading safetensors directly
# (bf16->fp32, experts loaded lazily). Must reproduce ref.json's full_ids from
# prompt_ids -> validates the exact arch before porting to MFL.
import json, os, sys, struct, numpy as np

HF = sys.argv[1] if len(sys.argv) > 1 else "olmoe-hf"
D, NH, HD, I, E, K, L, VOCAB = 2048, 16, 128, 1024, 64, 8, 16, 50304
import os
THETA = 10000.0
EPS = float(os.environ.get("EPS","1e-5"))

# ---- safetensors reader (bf16 -> fp32) ----
class ST:
    def __init__(self, hf):
        idx = json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]
        self.shards = {}   # shard -> (mmap, header, data_start)
        self.loc = {}      # name -> shard
        for name, shard in idx.items():
            self.loc[name] = shard
            if shard not in self.shards:
                path = os.path.join(hf, shard)
                f = open(path, "rb")
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
                self.shards[shard] = (np.memmap(path, dtype=np.uint8, mode="r"), header, 8 + n)
    def get(self, name):
        shard = self.loc[name]
        mm, header, base = self.shards[shard]
        h = header[name]
        s, e = h["data_offsets"]
        raw = mm[base + s: base + e]
        if h["dtype"] == "BF16":
            u16 = raw.view(np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        elif h["dtype"] == "F32":
            arr = raw.view(np.float32)
        elif h["dtype"] == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        else:
            raise ValueError(h["dtype"])
        return arr.reshape(h["shape"]).astype(np.float32)

st = ST(HF)
def W(n): return st.get(n)

def rms(x, w):
    return (x / np.sqrt(np.mean(x*x, -1, keepdims=True) + EPS)) * w

def silu(x): return x / (1.0 + np.exp(-x))

inv_freq = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))  # [64]
def rope(vec, pos):  # vec [NH, HD]
    ang = pos * inv_freq                      # [64]
    cos = np.cos(np.concatenate([ang, ang]))  # [128]
    sin = np.sin(np.concatenate([ang, ang]))
    half = HD // 2
    rot = np.concatenate([-vec[:, half:], vec[:, :half]], -1)
    return (vec * cos + rot * sin).astype(np.float32)

Kc = [np.zeros((0, NH, HD), np.float32) for _ in range(L)]
Vc = [np.zeros((0, NH, HD), np.float32) for _ in range(L)]

def layer(li, x, pos):  # x [D] at absolute position pos
    ln = W(f"model.layers.{li}.input_layernorm.weight")
    h = rms(x, ln)
    q = rms(W(f"model.layers.{li}.self_attn.q_proj.weight") @ h, W(f"model.layers.{li}.self_attn.q_norm.weight"))
    k = rms(W(f"model.layers.{li}.self_attn.k_proj.weight") @ h, W(f"model.layers.{li}.self_attn.k_norm.weight"))
    v = W(f"model.layers.{li}.self_attn.v_proj.weight") @ h
    q = rope(q.reshape(NH, HD), pos)
    k = rope(k.reshape(NH, HD), pos)
    v = v.reshape(NH, HD)
    Kc[li] = np.concatenate([Kc[li], k[None]], 0)
    Vc[li] = np.concatenate([Vc[li], v[None]], 0)
    T = Kc[li].shape[0]
    attn = np.zeros((NH, HD), np.float32)
    for hh in range(NH):
        sc = (Kc[li][:, hh, :] @ q[hh]) / np.sqrt(HD)   # [T]
        sc = np.exp(sc - sc.max()); sc /= sc.sum()
        attn[hh] = sc @ Vc[li][:, hh, :]
    ao = W(f"model.layers.{li}.self_attn.o_proj.weight") @ attn.reshape(D)
    x = x + ao
    # moe
    hn = rms(x, W(f"model.layers.{li}.post_attention_layernorm.weight"))
    rl = W(f"model.layers.{li}.mlp.gate.weight") @ hn        # [E]
    rw = np.exp(rl - rl.max()); rw /= rw.sum()
    sel = np.argsort(-rw)[:K]
    moe = np.zeros(D, np.float32)
    for e in sel:
        gp = W(f"model.layers.{li}.mlp.experts.{e}.gate_proj.weight")
        up = W(f"model.layers.{li}.mlp.experts.{e}.up_proj.weight")
        dn = W(f"model.layers.{li}.mlp.experts.{e}.down_proj.weight")
        moe += rw[e] * (dn @ (silu(gp @ hn) * (up @ hn)))
    return x + moe

embed = W("model.embed_tokens.weight")
fnorm = W("model.norm.weight")
lmh = W("lm_head.weight")
def forward(tok, pos):
    x = embed[tok].astype(np.float32).copy()
    for li in range(L):
        x = layer(li, x, pos)
    return lmh @ rms(x, fnorm)

ref = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "colibri-orig/ref.json"))
pids = ref["prompt_ids"]
gen_target = ref["full_ids"][len(pids):]
pos = 0
for t in pids[:-1]:
    forward(t, pos); pos += 1
cur = pids[-1]
out = []
for _ in range(len(gen_target)):
    lg = forward(cur, pos); pos += 1
    cur = int(np.argmax(lg)); out.append(cur)
print("target:", gen_target)
print("numpy :", out)
print("MATCH" if out == gen_target else "MISMATCH")
