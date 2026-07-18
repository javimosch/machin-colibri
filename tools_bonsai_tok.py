#!/usr/bin/env python3
"""Export GPT-2 BPE tokenizer from a Prism/Qwen GGUF into anvil qTK1 tok.bin."""
import struct, sys, numpy as np
from types import SimpleNamespace
from gguf.gguf_reader import GGUFReader, GGMLQuantizationType
from gguf.constants import GGML_QUANT_SIZES

GGUF = sys.argv[1] if len(sys.argv) > 1 else "models/bonsai-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/bonsai17-tok.bin"
MAGIC = 0x71544B31  # qTK1

def _build_tensors(self, start_offs, fields):
    tensors = []
    for field in fields:
        _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
        name = str(bytes(name_data), encoding="utf-8")
        raw_t = int(raw_dtype[0])
        n_elems = int(np.prod(dims))
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
        tensors.append(SimpleNamespace(name=name, data=data))
    self.tensors = tensors

GGUFReader._build_tensors = _build_tensors
r = GGUFReader(GGUF)
def field(k):
    return r.fields[k].contents()

tokens = list(field("tokenizer.ggml.tokens"))
merges = list(field("tokenizer.ggml.merges"))
VOCAB_SIZE = len(tokens)
print(f"vocab={VOCAB_SIZE} merges={len(merges)}")

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
b2u = bytes_to_unicode()

def wstr(f, s):
    if not isinstance(s, str):
        s = str(s)
    b = s.encode("utf-8"); f.write(struct.pack("<i", len(b))); f.write(b)

with open(OUT, "wb") as f:
    f.write(struct.pack("<3i", MAGIC, VOCAB_SIZE, len(merges)))
    for t in tokens: wstr(f, t)
    for m in merges: wstr(f, m if isinstance(m, str) else " ".join(m))
    for b in range(256): wstr(f, b2u[b])
    f.write(bytes(range(256)))
print(f"wrote {OUT}: vocab={VOCAB_SIZE} merges={len(merges)}")
