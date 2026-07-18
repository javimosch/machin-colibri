#!/usr/bin/env python3
"""Convert Prism Ternary-Bonsai-1.7B GGUF Q2_0 (type 42, g128) -> anvil akQ3 Q2 bin.

Anvil layout matches qwen3 Q4/Q8 (magic akQ3) with:
  GS=128, wbits=2, packed 2-bit codes (4 per byte, low bits first), fp32 group scales.
  w = (q - 1) * scale, q in {0,1,2} (ternary).
"""
import os, sys, struct, numpy as np
from types import SimpleNamespace
from gguf.gguf_reader import GGUFReader, GGMLQuantizationType
from gguf.constants import GGML_QUANT_SIZES

GGUF = sys.argv[1] if len(sys.argv) > 1 else "models/bonsai-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/bonsai17-q2.bin"
MAGIC = 0x616B5133  # akQ3
GS = 128

def _build_tensors(self, start_offs, fields):
    tensors = []
    for field in fields:
        _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
        name = str(bytes(name_data), encoding="utf-8")
        raw_t = int(raw_dtype[0])
        n_elems = int(np.prod(dims))
        np_dims = tuple(reversed(dims.tolist()))
        if raw_t == 42:
            n_bytes = n_elems * 34 // 128
            item_count, item_type = n_bytes, np.uint8
        else:
            ggml_type = GGMLQuantizationType(raw_t)
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            if ggml_type == GGMLQuantizationType.F16:
                item_count, item_type = n_elems, np.float16
            elif ggml_type == GGMLQuantizationType.F32:
                item_count, item_type = n_elems, np.float32
            else:
                item_count, item_type = n_bytes, np.uint8
        data_offs = int(start_offs + offset_tensor[0])
        data = np.frombuffer(self.data, dtype=item_type, count=item_count, offset=data_offs)
        tensors.append(SimpleNamespace(
            name=name, prism_type=raw_t, shape=tuple(int(x) for x in dims),
            np_dims=np_dims, n_elements=n_elems, data=data,
        ))
    self.tensors = tensors

GGUFReader._build_tensors = _build_tensors
r = GGUFReader(GGUF)
T = {t.name: t for t in r.tensors}

def field(k):
    return r.fields[k].contents()

D = int(field("qwen3.embedding_length"))
NH = int(field("qwen3.attention.head_count"))
NKV = int(field("qwen3.attention.head_count_kv"))
HD = int(field("qwen3.attention.key_length"))
I = int(field("qwen3.feed_forward_length"))
L = int(field("qwen3.block_count"))
SEQ = min(int(field("qwen3.context_length")), 8192)  # no YaRN in anvil rope yet
THETA = int(float(field("qwen3.rope.freq_base")))
VOCAB = int(T["token_embd.weight"].np_dims[0])
QD = NH * HD
KV = NKV * HD
assert D % GS == 0 and I % GS == 0 and QD % GS == 0
print(f"config: D={D} NH={NH} NKV={NKV} HD={HD} QD={QD} I={I} L={L} VOCAB={VOCAB} SEQ={SEQ} theta={THETA}")

def dequant_q2(t):
    """Return float32 weight matrix shaped [out, in] from Prism Q2_0 tensor."""
    raw = np.asarray(t.data).view(np.uint8)
    ne0, ne1 = t.shape  # ggml: ne0 innermost
    assert t.n_elements == ne0 * ne1
    nb = t.n_elements // 128
    blocks = raw.reshape(nb, 34)
    scales = np.frombuffer(np.ascontiguousarray(blocks[:, :2]), dtype="<f2").astype(np.float32)
    codes = blocks[:, 2:].astype(np.uint8)
    q = np.stack([codes & 3, (codes >> 2) & 3, (codes >> 4) & 3, (codes >> 6) & 3], axis=-1).reshape(nb, 128)
    w = (q.astype(np.float32) - 1.0) * scales[:, None]
    return w.reshape(ne1, ne0)  # [out, in]

def pack_q2(w):
    """Pack [out,in] float ternary-ish weights into anvil Q2 (codes + fp32 scales)."""
    out, inn = w.shape
    assert inn % GS == 0, (out, inn)
    wg = w.reshape(out, inn // GS, GS)
    # recover scale as maxabs (matches Prism for exact ternary)
    scale = np.abs(wg).max(-1).astype(np.float32)
    scale = np.where(scale == 0, np.float32(1e-9), scale)
    q = np.round(wg / scale[..., None]).clip(-1, 1).astype(np.int8) + 1  # 0..2
    # pack 4 codes / byte, low bits first
    q4 = q.reshape(out, inn // GS, GS // 4, 4)
    packed = (q4[..., 0] | (q4[..., 1] << 2) | (q4[..., 2] << 4) | (q4[..., 3] << 6)).astype(np.uint8)
    return packed.reshape(out, inn // 4).tobytes(), scale.tobytes()

def f32_norm(name):
    t = T[name]
    assert t.prism_type == 0
    return np.asarray(t.data, dtype=np.float32).astype("<f4").tobytes()

qblk = QD * D // 4 + 4 * (QD * D // GS)
kvblk = KV * D // 4 + 4 * (KV * D // GS)
attn_layer = 2 * qblk + 2 * kvblk
gu = I * D // 4 + 4 * (I * D // GS)
ffn_layer = 3 * gu

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "wb") as f:
    hdr = [0] * 32
    hdr[0:12] = [MAGIC, D, NH, NKV, HD, I, L, VOCAB, SEQ, GS, THETA, 2]  # wbits=2
    HDRB = 128
    embed_off = HDRB
    norms_off = embed_off + (VOCAB * D // 4 + 4 * (VOCAB * D // GS))
    fnorm_off = norms_off + L * (2 * D * 4 + 2 * HD * 4)
    attn_off = fnorm_off + D * 4
    ffn_off = attn_off + L * attn_layer
    hdr[12:17] = [embed_off, norms_off, fnorm_off, attn_off, ffn_off]
    f.write(struct.pack("<32i", *hdr))

    eq, es = pack_q2(dequant_q2(T["token_embd.weight"]))
    f.write(eq); f.write(es)

    for l in range(L):
        f.write(f32_norm(f"blk.{l}.attn_norm.weight"))
        f.write(f32_norm(f"blk.{l}.ffn_norm.weight"))
        f.write(f32_norm(f"blk.{l}.attn_q_norm.weight"))
        f.write(f32_norm(f"blk.{l}.attn_k_norm.weight"))
    f.write(f32_norm("output_norm.weight"))
    assert f.tell() == attn_off, (f.tell(), attn_off)

    for l in range(L):
        for name in [f"blk.{l}.attn_q.weight", f"blk.{l}.attn_k.weight",
                     f"blk.{l}.attn_v.weight", f"blk.{l}.attn_output.weight"]:
            q, s = pack_q2(dequant_q2(T[name])); f.write(q); f.write(s)
    assert f.tell() == ffn_off, (f.tell(), ffn_off)

    for l in range(L):
        for name in [f"blk.{l}.ffn_gate.weight", f"blk.{l}.ffn_up.weight",
                     f"blk.{l}.ffn_down.weight"]:
            q, s = pack_q2(dequant_q2(T[name])); f.write(q); f.write(s)

print(f"wrote {OUT}  {os.path.getsize(OUT)/1e6:.1f} MB")
print(f"offsets: embed={embed_off} norms={norms_off} fnorm={fnorm_off} attn={attn_off} ffn={ffn_off}")
print(f"strides: attn_layer={attn_layer} ffn_layer={ffn_layer} qblk={qblk} kvblk={kvblk} gu={gu}")
