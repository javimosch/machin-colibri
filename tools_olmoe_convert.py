#!/usr/bin/env python3
# OLMoE-1B-7B safetensors -> colibri MoE .bin (akM1).
# Quantized: attn q/k/v/o, router gate, experts -> int8 Q8_0 (group GS).
# fp32 (resident, small/lookup): embed, lm_head, all RMSNorm weights.
# Experts laid out contiguous per (layer,expert) for mmap streaming.
import json, os, sys, struct, numpy as np

HF = sys.argv[1] if len(sys.argv) > 1 else "olmoe-hf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/olmoe-q8.bin"
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

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
blk_attn = D * D + 4 * (D * D // GS)              # one int8 matrix [D,D] + scales
attn_layer_stride = 4 * blk_attn
router_layer = E * D + 4 * (E * D // GS)
expert_stride = 3 * I * D + 3 * 4 * (I * D // GS)

# assemble body sections in memory (dense fp32 small; big parts streamed to file)
with open(OUT, "wb") as f:
    hdr = [0] * 32
    hdr[0:12] = [MAGIC, D, NH, NH, HD, I, E, K, L, VOCAB, SEQ, GS]
    HDRB = 128
    embed_off = HDRB
    norms_off = embed_off + VOCAB * D * 4
    fnorm_off = norms_off + L * (4 * D * 4)          # in_ln,q_norm,k_norm,post_ln per layer
    lmhead_off = fnorm_off + D * 4
    attn_off = lmhead_off + VOCAB * D * 4
    router_off = attn_off + L * attn_layer_stride
    experts_off = router_off + L * router_layer
    hdr[12:19] = [embed_off, norms_off, fnorm_off, lmhead_off, attn_off, router_off, experts_off]
    f.write(struct.pack("<32i", *hdr))
    # embed fp32
    f.write(W("model.embed_tokens.weight").astype("<f4").tobytes())
    # per-layer norms fp32
    for l in range(L):
        for suff in ["input_layernorm", "self_attn.q_norm", "self_attn.k_norm", "post_attention_layernorm"]:
            f.write(W(f"model.layers.{l}.{suff}.weight").astype("<f4").tobytes())
    # final norm + lm_head fp32
    f.write(W("model.norm.weight").astype("<f4").tobytes())
    f.write(W("lm_head.weight").astype("<f4").tobytes())
    assert f.tell() == attn_off, (f.tell(), attn_off)
    # attn int8: per layer q,k,v,o
    for l in range(L):
        for suff in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            q, s = q8(W(f"model.layers.{l}.self_attn.{suff}.weight")); f.write(q); f.write(s)
    assert f.tell() == router_off, (f.tell(), router_off)
    # router int8
    for l in range(L):
        q, s = q8(W(f"model.layers.{l}.mlp.gate.weight")); f.write(q); f.write(s)
    assert f.tell() == experts_off, (f.tell(), experts_off)
    # experts int8, contiguous per (layer,expert): gate_q,gate_s,up_q,up_s,down_q,down_s
    for l in range(L):
        for e in range(E):
            for suff in ["gate_proj", "up_proj", "down_proj"]:
                q, s = q8(W(f"model.layers.{l}.mlp.experts.{e}.{suff}.weight")); f.write(q); f.write(s)
        if l % 4 == 0: print(f"  layer {l}/{L} experts written", flush=True)
sz = os.path.getsize(OUT)
print(f"wrote {OUT}  {sz/1e9:.2f} GB")
print(f"offsets: embed={embed_off} norms={norms_off} fnorm={fnorm_off} lmhead={lmhead_off} attn={attn_off} router={router_off} experts={experts_off}")
print(f"strides: attn_layer={attn_layer_stride} router_layer={router_layer} expert={expert_stride}")
