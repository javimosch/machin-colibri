#!/usr/bin/env python3
# Export the OLMoE (GPT-2/NeoX byte-level BPE) tokenizer to a flat binary the
# pure-MFL server loads: id->token-string (byte-level charspace) + ranked merges.
import json, struct, sys

TK = sys.argv[1] if len(sys.argv) > 1 else "olmoe-hf/tokenizer.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/olmoe-tok.bin"
VOCAB_SIZE = 151936
MAGIC = 0x71544B31  # "qTK1"

tk = json.load(open(TK))["model"]
vocab = tk["vocab"]          # token(byte-level str) -> id
merges = tk["merges"]        # ranked "A B"

id2tok = [""] * VOCAB_SIZE
for t, i in vocab.items():
    if 0 <= i < VOCAB_SIZE:
        id2tok[i] = t

# GPT-2 bytes_to_unicode: each of the 256 bytes -> a printable byte-level char
def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
b2u = bytes_to_unicode()

def wstr(f, s):
    b = s.encode("utf-8"); f.write(struct.pack("<i", len(b))); f.write(b)

with open(OUT, "wb") as f:
    f.write(struct.pack("<3i", MAGIC, VOCAB_SIZE, len(merges)))
    for t in id2tok: wstr(f, t)
    for m in merges: wstr(f, m if isinstance(m, str) else " ".join(m))
    for b in range(256): wstr(f, b2u[b]) # byte -> byte-level char (utf-8)
    f.write(bytes(range(256)))           # identity blob: lets MFL emit any raw byte for decode
print(f"wrote {OUT}: vocab={VOCAB_SIZE} merges={len(merges)} + 256 byte-level chars + id-blob")
