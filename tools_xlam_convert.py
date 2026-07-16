#!/usr/bin/env python3
# xLAM-1b-fc-r (dense Llama arch) safetensors -> colibri dense .bin (akD1).
# attn q/k/v/o + FFN gate/up/down + lm_head -> int8 Q8_0; embed/norms fp32.
import json, os, sys, struct, numpy as np

HF = sys.argv[1] if len(sys.argv) > 1 else "xlam-hf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/xlam-q8.bin"
LMBITS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
D, NH, HD, I, L, VOCAB = 2048, 16, 128, 5504, 24, 32256
SEQ, GS, THETA, RSCALE, MAGIC = 2048, 64, 100000, 4, 0x616B4431   # "akD1"

class ST:
    def __init__(self, hf):
        path = os.path.join(hf, "model.safetensors"); f = open(path, "rb")
        n = struct.unpack("<Q", f.read(8))[0]; self.hdr = json.loads(f.read(n))
        self.mm = np.memmap(path, dtype=np.uint8, mode="r"); self.base = 8 + n
    def get(self, name):
        h = self.hdr[name]; s, e = h["data_offsets"]; raw = self.mm[self.base+s:self.base+e]
        if h["dtype"] == "BF16": arr = ((raw.view(np.uint16).astype(np.uint32)) << 16).view(np.float32)
        elif h["dtype"] == "F32": arr = raw.view(np.float32)
        else: arr = raw.view(np.float16).astype(np.float32)
        return arr.reshape(h["shape"]).astype(np.float32)

st = ST(HF); W = st.get
def q8(w):
    out, inn = w.shape; assert inn % GS == 0
    wg = w.reshape(out, inn // GS, GS)
    scale = np.abs(wg).max(-1) / 127.0
    scale = np.where(scale == 0, 1e-9, scale).astype(np.float32)
    q = np.round(wg / scale[..., None]).clip(-127, 127).astype(np.int8).reshape(out, inn)
    return q.tobytes(), scale.tobytes()

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
blk_attn = D*D + 4*(D*D // GS); attn_layer = 4 * blk_attn
gu = I*D + 4*(I*D // GS)                     # gate/up block [I,D]
dn = D*I + 4*(D*I // GS)                      # down block [D,I]
ffn_layer = gu + gu + dn
lmhead_bytes = (VOCAB*D + 4*(VOCAB*D // GS)) if LMBITS == 8 else VOCAB*D*4

with open(OUT, "wb") as f:
    hdr = [0]*32
    hdr[0:13] = [MAGIC, D, NH, NH, HD, I, L, VOCAB, SEQ, GS, LMBITS, RSCALE, THETA]
    HDRB = 128
    embed_off = HDRB
    norms_off = embed_off + VOCAB*D*4
    fnorm_off = norms_off + L*(2*D*4)          # in_ln, post_ln per layer
    lmhead_off = fnorm_off + D*4
    attn_off = lmhead_off + lmhead_bytes
    ffn_off = attn_off + L*attn_layer
    hdr[13:19] = [embed_off, norms_off, fnorm_off, lmhead_off, attn_off, ffn_off]
    f.write(struct.pack("<32i", *hdr))
    f.write(W("model.embed_tokens.weight").astype("<f4").tobytes())
    for l in range(L):
        for suff in ["input_layernorm", "post_attention_layernorm"]:
            f.write(W(f"model.layers.{l}.{suff}.weight").astype("<f4").tobytes())
    f.write(W("model.norm.weight").astype("<f4").tobytes())
    lmw = W("lm_head.weight")
    if LMBITS == 8:
        q, s = q8(lmw); f.write(q); f.write(s)
    else:
        f.write(lmw.astype("<f4").tobytes())
    assert f.tell() == attn_off, (f.tell(), attn_off)
    for l in range(L):
        for suff in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            q, s = q8(W(f"model.layers.{l}.self_attn.{suff}.weight")); f.write(q); f.write(s)
    assert f.tell() == ffn_off, (f.tell(), ffn_off)
    for l in range(L):
        for suff in ["gate_proj", "up_proj", "down_proj"]:
            q, s = q8(W(f"model.layers.{l}.mlp.{suff}.weight")); f.write(q); f.write(s)
print(f"wrote {OUT}  {os.path.getsize(OUT)/1e9:.2f} GB (lm_head {'int8' if LMBITS==8 else 'fp32'})")
print(f"offsets: embed={embed_off} norms={norms_off} fnorm={fnorm_off} lmhead={lmhead_off} attn={attn_off} ffn={ffn_off}")
print(f"strides: attn_layer={attn_layer} ffn_layer={ffn_layer} gu={gu} dn={dn}")
