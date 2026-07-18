#!/usr/bin/env python3
# Qwen3-Embedding-0.6B -> anvil embedding .bin (akQE).
# KEY DELTA vs qwen3 chat: NH*HD (QD=2048) != D (1024): q_proj [QD,D],
# o_proj [D,QD]. No lm_head at all (embedding model: output = hidden states).
# attn q/k/v/o + FFN gate/up/down + embed -> int8 Q8_0; norms fp32.
import json, os, sys, struct, numpy as np

HF = sys.argv[1] if len(sys.argv) > 1 else "qwen3-embed-hf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/qwen3e-q8.bin"

cfg = json.load(open(os.path.join(HF, "config.json")))
D = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
HD = cfg["head_dim"]; I = cfg["intermediate_size"]; L = cfg["num_hidden_layers"]
VOCAB = cfg["vocab_size"]; THETA = int(cfg["rope_theta"]); SEQ = cfg["max_position_embeddings"]
QD = NH * HD; KV = NKV * HD
GS, MAGIC = 64, 0x616B5145  # "akQE"
print(f"config: D={D} QD={QD} NH={NH} NKV={NKV} HD={HD} I={I} L={L} VOCAB={VOCAB} theta={THETA}")

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
    def rn(self, name):
        # this checkpoint stores names without the "model." wrapper prefix
        if name not in self.loc and name.startswith("model.") and name[6:] in self.loc: return name[6:]
        return name
    def has(self, name): return self.rn(name) in self.loc
    def get(self, name):
        name = self.rn(name)
        mm, hdr, base = self.shards[self.loc[name]]; h = hdr[name]
        s, e = h["data_offsets"]; raw = mm[base+s:base+e]
        if h["dtype"] == "BF16": arr = ((raw.view(np.uint16).astype(np.uint32)) << 16).view(np.float32)
        elif h["dtype"] == "F32": arr = raw.view(np.float32)
        else: arr = raw.view(np.float16).astype(np.float32)
        return arr.reshape(h["shape"]).astype(np.float32)
st = ST(HF); W = st.get

# trust the files, not the brief: verify actual shapes
assert W("model.layers.0.self_attn.q_proj.weight").shape == (QD, D)
assert W("model.layers.0.self_attn.k_proj.weight").shape == (KV, D)
assert W("model.layers.0.self_attn.o_proj.weight").shape == (D, QD)
assert W("model.embed_tokens.weight").shape == (VOCAB, D)
assert not st.has("model.layers.0.self_attn.q_proj.bias"), "unexpected qkv bias"
print("shapes verified: q [QD,D], k/v [KV,D], o [D,QD], embed [VOCAB,D]; no lm_head needed (tied/unused)")

def q8(w):
    out, inn = w.shape; assert inn % GS == 0, (out, inn)
    wg = w.reshape(out, inn // GS, GS)
    scale = np.abs(wg).max(-1) / 127.0
    scale = np.where(scale == 0, 1e-9, scale).astype(np.float32)
    q = np.round(wg / scale[..., None]).clip(-127, 127).astype(np.int8).reshape(out, inn)
    return q.tobytes(), scale.tobytes()

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
qblk = QD*D + 4*(QD*D // GS)            # q_proj int8 + scales
kvblk = KV*D + 4*(KV*D // GS)           # k/v_proj
oblk = D*QD + 4*(D*QD // GS)            # o_proj (NOT square: [D, QD])
attn_layer = qblk + oblk + 2*kvblk      # q, k, v, o
gu = I*D + 4*(I*D // GS)
ffn_layer = 3*gu

with open(OUT, "wb") as f:
    hdr = [0]*32
    hdr[0:12] = [MAGIC, D, NH, NKV, HD, I, L, VOCAB, SEQ, GS, THETA, QD]
    HDRB = 128
    embed_off = HDRB
    norms_off = embed_off + (VOCAB*D + 4*(VOCAB*D // GS))   # embed int8
    fnorm_off = norms_off + L*(2*D*4 + 2*HD*4)
    attn_off = fnorm_off + D*4
    ffn_off = attn_off + L*attn_layer
    hdr[12:17] = [embed_off, norms_off, fnorm_off, attn_off, ffn_off]
    f.write(struct.pack("<32i", *hdr))
    eq, es = q8(W("model.embed_tokens.weight")); f.write(eq); f.write(es)
    for l in range(L):
        f.write(W(f"model.layers.{l}.input_layernorm.weight").astype("<f4").tobytes())
        f.write(W(f"model.layers.{l}.post_attention_layernorm.weight").astype("<f4").tobytes())
        f.write(W(f"model.layers.{l}.self_attn.q_norm.weight").astype("<f4").tobytes())
        f.write(W(f"model.layers.{l}.self_attn.k_norm.weight").astype("<f4").tobytes())
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
print(f"offsets: embed={embed_off} norms={norms_off} fnorm={fnorm_off} attn={attn_off} ffn={ffn_off}")
print(f"strides: attn_layer={attn_layer} ffn_layer={ffn_layer} qblk={qblk} oblk={oblk} kvblk={kvblk} gu={gu}")
