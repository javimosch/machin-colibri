#!/usr/bin/env python3
# Qwen2.5-1.5B-Instruct -> colibri dense-GQA .bin (akQ1).
# GQA (q [D,D], k/v [KV,D]), QKV bias (fp32), tied embeddings (lm_head=embed).
# attn q/k/v/o + FFN gate/up/down -> int8 Q8_0; embed/norms/biases fp32.
import json, os, sys, struct, numpy as np

HF = sys.argv[1] if len(sys.argv) > 1 else "qwen-hf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/qwen-q8.bin"
D, NH, NKV, HD, I, L, VOCAB = 1536, 12, 2, 128, 8960, 28, 151936
KV = NKV * HD
SEQ, GS, THETA, MAGIC = 32768, 64, 1000000, 0x616B5131  # "akQ1"

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
def q8(w):
    out, inn = w.shape; assert inn % GS == 0, (out, inn)
    wg = w.reshape(out, inn // GS, GS)
    scale = np.abs(wg).max(-1) / 127.0
    scale = np.where(scale == 0, 1e-9, scale).astype(np.float32)
    q = np.round(wg / scale[..., None]).clip(-127, 127).astype(np.int8).reshape(out, inn)
    return q.tobytes(), scale.tobytes()

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
qblk = D*D + 4*(D*D // GS)
kvblk = KV*D + 4*(KV*D // GS)
attn_layer = 2*qblk + 2*kvblk           # q, k, v, o
gu = I*D + 4*(I*D // GS)
ffn_layer = 3*gu
bias_layer = (D + 2*KV) * 4             # q_bias[D], k_bias[KV], v_bias[KV]

with open(OUT, "wb") as f:
    hdr = [0]*32
    hdr[0:12] = [MAGIC, D, NH, NKV, HD, I, L, VOCAB, SEQ, GS, THETA, 0]
    HDRB = 128
    embed_off = HDRB
    norms_off = embed_off + (VOCAB*D + 4*(VOCAB*D // GS))   # embed int8 (tied lm_head)
    bias_off = norms_off + L*(2*D*4)
    fnorm_off = bias_off + L*bias_layer
    attn_off = fnorm_off + D*4
    ffn_off = attn_off + L*attn_layer
    hdr[12:18] = [embed_off, norms_off, bias_off, fnorm_off, attn_off, ffn_off]
    f.write(struct.pack("<32i", *hdr))
    eq, es = q8(W("model.embed_tokens.weight")); f.write(eq); f.write(es)  # embed int8 = lm_head (tied)
    for l in range(L):
        for suff in ["input_layernorm", "post_attention_layernorm"]:
            f.write(W(f"model.layers.{l}.{suff}.weight").astype("<f4").tobytes())
    for l in range(L):
        for suff in ["q_proj", "k_proj", "v_proj"]:
            f.write(W(f"model.layers.{l}.self_attn.{suff}.bias").astype("<f4").tobytes())
    f.write(W("model.norm.weight").astype("<f4").tobytes())
    assert f.tell() == attn_off, (f.tell(), attn_off)
    for l in range(L):
        for suff in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            q, s = q8(W(f"model.layers.{l}.self_attn.{suff}.weight")); f.write(q); f.write(s)
    assert f.tell() == ffn_off, (f.tell(), ffn_off)
    for l in range(L):
        for suff in ["gate_proj", "up_proj", "down_proj"]:
            q, s = q8(W(f"model.layers.{l}.mlp.{suff}.weight")); f.write(q); f.write(s)
print(f"wrote {OUT}  {os.path.getsize(OUT)/1e9:.2f} GB")
print(f"offsets: embed={embed_off} norms={norms_off} bias={bias_off} fnorm={fnorm_off} attn={attn_off} ffn={ffn_off}")
print(f"strides: attn_layer={attn_layer} ffn_layer={ffn_layer} bias_layer={bias_layer} qblk={qblk} kvblk={kvblk} gu={gu}")
