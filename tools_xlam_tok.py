#!/usr/bin/env python3
# Export the xLAM/deepseek byte-level BPE tokenizer to a flat binary the pure-MFL
# server loads: id->token (byte-level charspace) + ranked merges + byte table.
import json, struct, sys

TK = sys.argv[1] if len(sys.argv) > 1 else "xlam-hf/tokenizer.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/xlam-tok.bin"
VOCAB_SIZE = 32256
MAGIC = 0x78544B31  # "xTK1"

d = json.load(open(TK))
tk = d["model"]
vocab = tk["vocab"]
merges = tk["merges"]

id2tok = [""] * VOCAB_SIZE
for t, i in vocab.items():
    if 0 <= i < VOCAB_SIZE:
        id2tok[i] = t
# added/special tokens (BOS 32013, EOT 32021, byte-fallbacks) — content kept for decode
for t in d.get("added_tokens", []):
    if 0 <= t["id"] < VOCAB_SIZE:
        id2tok[t["id"]] = t["content"]

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
    for m in merges: wstr(f, m)
    for b in range(256): wstr(f, b2u[b])
    f.write(bytes(range(256)))
print(f"wrote {OUT}: vocab={VOCAB_SIZE} merges={len(merges)} + byte table + id-blob")
