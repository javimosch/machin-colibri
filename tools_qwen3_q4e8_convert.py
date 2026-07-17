#!/usr/bin/env python3
# Qwen2.5-1.5B-Instruct -> colibri dense-GQA .bin (akQ1).
# GQA (q [D,D], k/v [KV,D]), QKV bias (fp32), tied embeddings (lm_head=embed).
# attn q/k/v/o + FFN gate/up/down -> int8 Q8_0; embed/norms/biases fp32.
import json, os, sys, struct, numpy as np

HF = sys.argv[1] if len(sys.argv) > 1 else "qwen-hf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/qwen3-q4e8.bin"
D, NH, NKV, HD, I, L, VOCAB = 2048, 16, 8, 128, 6144, 28, 151936
KV = NKV * HD
SEQ, GS, THETA, MAGIC = 40960, 64, 1000000, 0x616B5133  # "akQ3"

class ST:
    def __init__(self, hf):
        import os
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
    def has(self, name): return name in self.loc
    def get(self, name):
        mm, hdr, base = self.shards[self.loc[name]]; h = hdr[name]
        s, e = h["data_offsets"]; raw = mm[base+s:base+e]
        if h["dtype"] == "BF16": arr = ((raw.view(np.uint16).astype(np.uint32)) << 16).view(np.float32)
        elif h["dtype"] == "F32": arr = raw.view(np.float32)
        else: arr = raw.view(np.float16).astype(np.float32)
        return arr.reshape(h["shape"]).astype(np.float32)
st = ST(HF); W = st.get
def q8(w):
    out, inn = w.shape; assert inn % GS == 0, (out, inn)
    wg = w.reshape(out, inn // GS, GS)
    scale = np.abs(wg).max(-1) / 127.0
    scale = np.where(scale == 0, 1e-9, scale).astype(np.float32)
    q = np.round(wg / scale[..., None]).clip(-127, 127).astype(np.int8).reshape(out, inn)
    return q.tobytes(), scale.tobytes()

def q4(w):
    out, inn = w.shape; assert inn % GS == 0, (out, inn)
    half = GS // 2
    wg = w.reshape(out, inn // GS, GS)
    scale = np.abs(wg).max(-1) / 7.0
    scale = np.where(scale == 0, 1e-9, scale).astype(np.float32)
    q = np.round(wg / scale[..., None]).clip(-8, 7).astype(np.int32)
    q = (q + 8).astype(np.uint8)                       # 0..15
    lo = q[..., :half]; hi = q[..., half:]
    packed = (lo | (hi << 4)).astype(np.uint8)         # (out, ng, half)
    return packed.reshape(out, inn // 2).tobytes(), scale.tobytes()

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
qblk = D*D//2 + 4*(D*D // GS)
kvblk = KV*D//2 + 4*(KV*D // GS)
attn_layer = 2*qblk + 2*kvblk           # q, k, v, o
gu = I*D//2 + 4*(I*D // GS)
ffn_layer = 3*gu

with open(OUT, "wb") as f:
    hdr = [0]*32
    hdr[0:12] = [MAGIC, D, NH, NKV, HD, I, L, VOCAB, SEQ, GS, THETA, 48]  # q4 layers + q8 embed
    HDRB = 128
    embed_off = HDRB
    norms_off = embed_off + (VOCAB*D + 4*(VOCAB*D // GS))   # Q8 embed/lm_head   # embed int8 (tied lm_head)
    fnorm_off = norms_off + L*(2*D*4 + 2*HD*4)
    attn_off = fnorm_off + D*4
    ffn_off = attn_off + L*attn_layer
    hdr[12:17] = [embed_off, norms_off, fnorm_off, attn_off, ffn_off]
    f.write(struct.pack("<32i", *hdr))
    eq, es = q8(W("model.embed_tokens.weight")); f.write(eq); f.write(es)  # embed int8 = lm_head (tied)
    for l in range(L):
        f.write(W(f"model.layers.{l}.input_layernorm.weight").astype("<f4").tobytes())
        f.write(W(f"model.layers.{l}.post_attention_layernorm.weight").astype("<f4").tobytes())
        f.write(W(f"model.layers.{l}.self_attn.q_norm.weight").astype("<f4").tobytes())
        f.write(W(f"model.layers.{l}.self_attn.k_norm.weight").astype("<f4").tobytes())
    f.write(W("model.norm.weight").astype("<f4").tobytes())
    assert f.tell() == attn_off, (f.tell(), attn_off)
    for l in range(L):
        for suff in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            q, s = q4(W(f"model.layers.{l}.self_attn.{suff}.weight")); f.write(q); f.write(s)
    assert f.tell() == ffn_off, (f.tell(), ffn_off)
    for l in range(L):
        for suff in ["gate_proj", "up_proj", "down_proj"]:
            q, s = q4(W(f"model.layers.{l}.mlp.{suff}.weight")); f.write(q); f.write(s)
print(f"wrote {OUT}  {os.path.getsize(OUT)/1e9:.2f} GB")
print(f"offsets: embed={embed_off} norms={norms_off} fnorm={fnorm_off} attn={attn_off} ffn={ffn_off}")
print(f"strides: attn_layer={attn_layer} ffn_layer={ffn_layer} qblk={qblk} kvblk={kvblk} gu={gu}")
