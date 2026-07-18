#!/usr/bin/env python3
# Ground-truth Qwen3-Embedding-0.6B in numpy fp32 (no torch): text -> L2-normalized
# embedding. Causal LM forward (vectorized over the sequence), LAST-token pooling
# after the final RMSNorm, then L2-normalize; --dim N = MRL truncate THEN re-norm.
# Matches the official usage: append the EOS/pad token <|endoftext|> (151643)
# before pooling; query side optionally wrapped as "Instruct: {task}\nQuery:{text}".
# Usage: tools_qwen3e_ref.py <hf-dir> <text> [--dim N] [--instruct TASK] [--ids-only]
import json, os, sys, struct, numpy as np
from tokenizers import Tokenizer

args = [a for a in sys.argv[1:]]
def opt(name, default=None):
    if name in args:
        i = args.index(name); v = args[i+1]; del args[i:i+2]; return v
    return default
DIM = int(opt("--dim", "0"))
TASK = opt("--instruct", None)
IDS_ONLY = "--ids-only" in args
if IDS_ONLY: args.remove("--ids-only")
HF = args[0] if len(args) > 0 else "qwen3-embed-hf"
TEXT = args[1] if len(args) > 1 else "The capital of France is Paris."

cfg = json.load(open(os.path.join(HF, "config.json")))
D = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
HD = cfg["head_dim"]; L = cfg["num_hidden_layers"]
KVMUL = NH // NKV; THETA = float(cfg["rope_theta"]); EPS = cfg["rms_norm_eps"]
EOS = cfg["eos_token_id"]  # 151643 <|endoftext|> (also pad) — appended before pooling

class ST:
    def __init__(self, hf):
        idxp = os.path.join(hf, "model.safetensors.index.json")
        self.shards = {}; self.loc = {}
        if os.path.exists(idxp):
            wm = json.load(open(idxp))["weight_map"]
            for name, sh in wm.items():
                self.loc[name] = sh
                if sh not in self.shards:
                    p = os.path.join(hf, sh); f = open(p, "rb")
                    n = struct.unpack("<Q", f.read(8))[0]
                    self.shards[sh] = (np.memmap(p, dtype=np.uint8, mode="r"), json.loads(f.read(n)), 8+n)
        else:
            p = os.path.join(hf, "model.safetensors"); f = open(p, "rb")
            n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n))
            self.shards["_"] = (np.memmap(p, dtype=np.uint8, mode="r"), hdr, 8+n)
            for k in hdr:
                if k != "__metadata__": self.loc[k] = "_"
    def get(self, name):
        # this checkpoint stores names without the "model." wrapper prefix
        if name not in self.loc and name.startswith("model.") and name[6:] in self.loc: name = name[6:]
        mm, hdr, base = self.shards[self.loc[name]]; h = hdr[name]
        s, e = h["data_offsets"]; raw = mm[base+s:base+e]
        if h["dtype"] == "BF16": arr = ((raw.view(np.uint16).astype(np.uint32)) << 16).view(np.float32)
        elif h["dtype"] == "F32": arr = raw.view(np.float32)
        else: arr = raw.view(np.float16).astype(np.float32)
        return arr.reshape(h["shape"]).astype(np.float32)
st = ST(HF); W = st.get

def rms(x, w): return (x / np.sqrt(np.mean(x*x, -1, keepdims=True) + EPS)) * w
def silu(x): return x / (1.0 + np.exp(-x))
inv_freq = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
def rope(vec, T):  # vec (T, nh, HD)
    ang = np.arange(T)[:, None] * inv_freq[None, :]           # (T, HD/2)
    cos = np.cos(np.concatenate([ang, ang], -1))[:, None, :]  # (T, 1, HD)
    sin = np.sin(np.concatenate([ang, ang], -1))[:, None, :]
    half = HD // 2
    rot = np.concatenate([-vec[..., half:], vec[..., :half]], -1)
    return (vec * cos + rot * sin).astype(np.float32)

def forward_last_hidden(ids):
    T = len(ids)
    x = W("model.embed_tokens.weight")[ids].astype(np.float32).copy()  # (T, D)
    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)            # causal
    for li in range(L):
        p = f"model.layers.{li}."
        h = rms(x, W(p + "input_layernorm.weight"))
        q = (h @ W(p + "self_attn.q_proj.weight").T).reshape(T, NH, HD)
        k = (h @ W(p + "self_attn.k_proj.weight").T).reshape(T, NKV, HD)
        v = (h @ W(p + "self_attn.v_proj.weight").T).reshape(T, NKV, HD)
        q = rms(q, W(p + "self_attn.q_norm.weight")); k = rms(k, W(p + "self_attn.k_norm.weight"))
        q = rope(q, T); k = rope(k, T)
        attn = np.zeros((T, NH, HD), np.float32)
        for hh in range(NH):
            kk = k[:, hh // KVMUL, :]; vv = v[:, hh // KVMUL, :]
            sc = (q[:, hh, :] @ kk.T) / np.sqrt(HD) + mask
            sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
            attn[:, hh, :] = sc @ vv
        x = x + attn.reshape(T, NH*HD) @ W(p + "self_attn.o_proj.weight").T
        hn = rms(x, W(p + "post_attention_layernorm.weight"))
        g = silu(hn @ W(p + "mlp.gate_proj.weight").T)
        u = hn @ W(p + "mlp.up_proj.weight").T
        x = x + (g * u) @ W(p + "mlp.down_proj.weight").T
    return rms(x[-1], W("model.norm.weight"))  # final RMSNorm, last token

tk = Tokenizer.from_file(os.path.join(HF, "tokenizer.json"))
text = TEXT if TASK is None else f"Instruct: {TASK}\nQuery:{TEXT}"
ids = tk.encode(text).ids            # post_processor already appends EOS <|endoftext|>
if len(ids) == 0 or ids[-1] != EOS: ids = ids + [EOS]
if IDS_ONLY:
    print(json.dumps({"ids": ids})); sys.exit(0)
h = forward_last_hidden(ids)
if DIM > 0: h = h[:DIM]
emb = h / np.sqrt((h * h).sum())
print(json.dumps({"embedding": [round(float(v), 8) for v in emb],
                  "dim": len(emb), "tokens": len(ids), "ids": ids}))
