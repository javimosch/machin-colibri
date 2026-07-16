#!/usr/bin/env python3
# OLMoE-1B-7B safetensors -> colibri MoE .bin (akM1).
# Quantized: attn q/k/v/o, router gate, experts -> int8 Q8_0 (group GS).
# fp32 (resident, small/lookup): embed, lm_head, all RMSNorm weights.
# Experts laid out contiguous per (layer,expert) for mmap streaming.
import json, os, sys, struct, numpy as np

HF = sys.argv[1] if len(sys.argv) > 1 else "olmoe-hf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/olmoe-q8.bin"
EBITS = int(sys.argv[3]) if len(sys.argv) > 3 else 8   # expert quant: 8 (Q8_0) or 4 (Q4)
LMBITS = int(sys.argv[4]) if len(sys.argv) > 4 else 32  # lm_head: 32 (fp32) or 8 (int8)
D, NH, HD, I, E, K, L, VOCAB = 2048, 16, 128, 1024, 64, 8, 16, 50304
SEQ, GS, MAGIC = 512, 64, 0x616B4D31   # "akM1"

class ST:
    def __init__(self, hf):
        idx = json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]
        self.loc, self.shards = idx, {}
        for name, shard in idx.items():
            if shard not in self.shards:
                path = os.path.join(hf, shard); f = open(path, "rb")
                n = struct.unpack("<Q", f.read(8))[0]
                self.shards[shard] = (np.memmap(path, dtype=np.uint8, mode="r"), json.loads(f.read(n)), 8 + n)
    def get(self, name):
        mm, header, base = self.shards[self.loc[name]]; h = header[name]
        s, e = h["data_offsets"]; raw = mm[base + s: base + e]
        if h["dtype"] == "BF16": arr = ((raw.view(np.uint16).astype(np.uint32)) << 16).view(np.float32)
        elif h["dtype"] == "F32": arr = raw.view(np.float32)
        elif h["dtype"] == "F16": arr = raw.view(np.float16).astype(np.float32)
        else: raise ValueError(h["dtype"])
        return arr.reshape(h["shape"]).astype(np.float32)

st = ST(HF); W = st.get

def q8(w):  # [out,in] -> bytes: int8 q [out*in] then f32 scales [out*(in//GS)]
    out, inn = w.shape; assert inn % GS == 0
    wg = w.reshape(out, inn // GS, GS)
    scale = np.abs(wg).max(-1) / 127.0
    scale = np.where(scale == 0, 1e-9, scale).astype(np.float32)
    q = np.round(wg / scale[..., None]).clip(-127, 127).astype(np.int8).reshape(out, inn)
    return q.tobytes(), scale.tobytes()

def q4(w):  # [out,in] -> packed uint8 [out*(in//2)] (dot_q4 layout) + f32 scales [out*(in//GS)]
    out, inn = w.shape; assert inn % GS == 0
    half = GS // 2
    wg = w.reshape(out, inn // GS, GS)
    scale = np.abs(wg).max(-1) / 7.0
    scale = np.where(scale == 0, 1e-9, scale).astype(np.float32)
    q = (np.round(wg / scale[..., None]).clip(-8, 7).astype(np.int32) + 8)  # 0..15
    lo = q[..., :half]; hi = q[..., half:]
    packed = (lo | (hi << 4)).astype(np.uint8).reshape(out, inn // 2)
    return packed.tobytes(), scale.tobytes()

def qexp(w): return q4(w) if EBITS == 4 else q8(w)
expert_matbytes = (I * D // 2 if EBITS == 4 else I * D)   # per expert sub-matrix int bytes

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
blk_attn = D * D + 4 * (D * D // GS)              # one int8 matrix [D,D] + scales
attn_layer_stride = 4 * blk_attn
router_layer = E * D + 4 * (E * D // GS)
expert_stride = 3 * (expert_matbytes + 4 * (I * D // GS))
lmhead_bytes = (VOCAB * D + 4 * (VOCAB * D // GS)) if LMBITS == 8 else (VOCAB * D * 4)

with open(OUT, "wb") as f:
    hdr = [0] * 32
    hdr[0:12] = [MAGIC, D, NH, NH, HD, I, E, K, L, VOCAB, SEQ, GS]
    HDRB = 128
    embed_off = HDRB
    norms_off = embed_off + VOCAB * D * 4
    fnorm_off = norms_off + L * (4 * D * 4)          # in_ln,q_norm,k_norm,post_ln per layer
    lmhead_off = fnorm_off + D * 4
    attn_off = lmhead_off + lmhead_bytes
    router_off = attn_off + L * attn_layer_stride
    experts_off = router_off + L * router_layer
    hdr[12:19] = [embed_off, norms_off, fnorm_off, lmhead_off, attn_off, router_off, experts_off]
    hdr[19] = EBITS
    hdr[20] = LMBITS
    f.write(struct.pack("<32i", *hdr))
    f.write(W("model.embed_tokens.weight").astype("<f4").tobytes())          # embed fp32
    for l in range(L):                                                       # per-layer norms fp32
        for suff in ["input_layernorm", "self_attn.q_norm", "self_attn.k_norm", "post_attention_layernorm"]:
            f.write(W(f"model.layers.{l}.{suff}.weight").astype("<f4").tobytes())
    f.write(W("model.norm.weight").astype("<f4").tobytes())                  # final norm fp32
    lmw = W("lm_head.weight")                                                # lm_head fp32 or int8
    if LMBITS == 8:
        q, s = q8(lmw); f.write(q); f.write(s)
    else:
        f.write(lmw.astype("<f4").tobytes())
    assert f.tell() == attn_off, (f.tell(), attn_off)
    for l in range(L):                                                       # attn int8
        for suff in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            q, s = q8(W(f"model.layers.{l}.self_attn.{suff}.weight")); f.write(q); f.write(s)
    assert f.tell() == router_off, (f.tell(), router_off)
    for l in range(L):                                                       # router int8
        q, s = q8(W(f"model.layers.{l}.mlp.gate.weight")); f.write(q); f.write(s)
    assert f.tell() == experts_off, (f.tell(), experts_off)
    for l in range(L):                                                       # experts int8 or int4
        for e in range(E):
            for suff in ["gate_proj", "up_proj", "down_proj"]:
                q, s = qexp(W(f"model.layers.{l}.mlp.experts.{e}.{suff}.weight")); f.write(q); f.write(s)
        if l % 4 == 0: print(f"  layer {l}/{L} experts written", flush=True)
sz = os.path.getsize(OUT)
print(f"wrote {OUT}  {sz/1e9:.2f} GB  (experts int{EBITS}, lm_head {'int8' if LMBITS==8 else 'fp32'})")
print(f"offsets: embed={embed_off} norms={norms_off} fnorm={fnorm_off} lmhead={lmhead_off} attn={attn_off} router={router_off} experts={experts_off}")
print(f"strides: attn_layer={attn_layer_stride} router_layer={router_layer} expert={expert_stride}")
