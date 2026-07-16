#!/usr/bin/env python3
# Ground-truth Qwen2.5-1.5B-Instruct forward in numpy fp32: GQA (12/2), QKV bias,
# tied embeddings, rope theta 1e6 (native 32k, no scaling), no qk-norm.
import json, os, sys, struct, numpy as np
from tokenizers import Tokenizer

HF = sys.argv[1] if len(sys.argv) > 1 else "qwen-hf"
D, NH, NKV, HD, I, L, VOCAB = 1536, 12, 2, 128, 8960, 28, 151936
KV = NKV * HD          # 256
KVMUL = NH // NKV      # 6
THETA, EPS = 1000000.0, 1e-6

class ST:
    def __init__(self, hf):
        path = os.path.join(hf, "model.safetensors"); f = open(path, "rb")
        n = struct.unpack("<Q", f.read(8))[0]; self.hdr = json.loads(f.read(n))
        self.mm = np.memmap(path, dtype=np.uint8, mode="r"); self.base = 8 + n
    def has(self, name): return name in self.hdr
    def get(self, name):
        h = self.hdr[name]; s, e = h["data_offsets"]; raw = self.mm[self.base+s:self.base+e]
        if h["dtype"] == "BF16": arr = ((raw.view(np.uint16).astype(np.uint32)) << 16).view(np.float32)
        elif h["dtype"] == "F32": arr = raw.view(np.float32)
        else: arr = raw.view(np.float16).astype(np.float32)
        return arr.reshape(h["shape"]).astype(np.float32)

st = ST(HF); W = st.get
print("QKV bias present:", st.has("model.layers.0.self_attn.q_proj.bias"))
def rms(x, w): return (x / np.sqrt(np.mean(x*x, -1, keepdims=True) + EPS)) * w
def silu(x): return x / (1.0 + np.exp(-x))
inv_freq = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
def rope(vec, pos):  # vec [n, HD]
    ang = pos * inv_freq
    cos = np.cos(np.concatenate([ang, ang])); sin = np.sin(np.concatenate([ang, ang]))
    half = HD // 2; rot = np.concatenate([-vec[:, half:], vec[:, :half]], -1)
    return (vec * cos + rot * sin).astype(np.float32)

Kc = [np.zeros((0, NKV, HD), np.float32) for _ in range(L)]
Vc = [np.zeros((0, NKV, HD), np.float32) for _ in range(L)]
def layer(li, x, pos):
    h = rms(x, W(f"model.layers.{li}.input_layernorm.weight"))
    q = (W(f"model.layers.{li}.self_attn.q_proj.weight") @ h + W(f"model.layers.{li}.self_attn.q_proj.bias")).reshape(NH, HD)
    k = (W(f"model.layers.{li}.self_attn.k_proj.weight") @ h + W(f"model.layers.{li}.self_attn.k_proj.bias")).reshape(NKV, HD)
    v = (W(f"model.layers.{li}.self_attn.v_proj.weight") @ h + W(f"model.layers.{li}.self_attn.v_proj.bias")).reshape(NKV, HD)
    q = rope(q, pos); k = rope(k, pos)
    Kc[li] = np.concatenate([Kc[li], k[None]], 0); Vc[li] = np.concatenate([Vc[li], v[None]], 0)
    attn = np.zeros((NH, HD), np.float32)
    for hh in range(NH):
        kk = Kc[li][:, hh // KVMUL, :]; vv = Vc[li][:, hh // KVMUL, :]
        sc = (kk @ q[hh]) / np.sqrt(HD)
        sc = np.exp(sc - sc.max()); sc /= sc.sum()
        attn[hh] = sc @ vv
    x = x + W(f"model.layers.{li}.self_attn.o_proj.weight") @ attn.reshape(D)
    hn = rms(x, W(f"model.layers.{li}.post_attention_layernorm.weight"))
    g = silu(W(f"model.layers.{li}.mlp.gate_proj.weight") @ hn)
    u = W(f"model.layers.{li}.mlp.up_proj.weight") @ hn
    return x + W(f"model.layers.{li}.mlp.down_proj.weight") @ (g * u)

embed = W("model.embed_tokens.weight"); fnorm = W("model.norm.weight")
lmh = W("lm_head.weight") if st.has("lm_head.weight") else embed   # tied
def forward(tok, pos):
    x = embed[tok].astype(np.float32).copy()
    for li in range(L): x = layer(li, x, pos)
    return lmh @ rms(x, fnorm)

tk = Tokenizer.from_file(os.path.join(HF, "tokenizer.json"))
prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
ids = tk.encode(prompt).ids
print("prompt ids:", ids)
pos = 0
for t in ids[:-1]: forward(t, pos); pos += 1
cur = ids[-1]; out = []
for _ in range(12):
    lg = forward(cur, pos); pos += 1; cur = int(np.argmax(lg)); out.append(cur)
    if cur == 151645: break   # <|im_end|>
print("gen ids:", out)
print("text:", repr(tk.decode(out)))
