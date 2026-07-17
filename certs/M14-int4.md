# M14 — int4 (Q4) Qwen3-1.7B with VNNI dot_q4 (2026-07-17)

The scalar int4 path (M11) was 1.08× and abandoned. With the AVX-VNNI kernel
infra (M13) the nibble unpack is essentially free, so int4 finally pays off —
and it helps **all** text (unlike speculation, which only helps repetitive).

## Kernel (machin core, branch vnni-dot-q8)
`mfl_q4_dot` + `mfl_matmul_q4_batch` via `vpdpbusd`. The stored nibble (value+8,
unsigned 0..15) is the unsigned operand, the int8 activation the signed operand:
`acc = Σ(w+8)·x = true_dot + 8·Σx`, subtract `8·Σx`. Bit-identical to the scalar
reference (verified `vnni==ref` on known data). Micro-bench (read-once parallel,
i5-13400): **dot_q4 4.2× dot_q8** — half the bytes + fast unpack.

## Model + engine
- `tools_qwen3_q4_convert.py`: split-nibble Q4 (group of GS=64 → GS/2 packed bytes,
  scale = maxabs/7, clip [-8,7], store +8), fp32 group scales kept. `wbits=4` in
  header[11]. **924 MB** (vs 1.8 GB Q8 — exactly half).
- `qwen3_q4_engine.src`: copy of the Q8 engine with all weight-byte offsets halved
  (qblk/kvblk/gu/embed_soff, the 3 dot sites, the 7 batched cases, the 7 single-token
  callers) + dot_q8→dot_q4, matmul_q8_batch→matmul_q4_batch, and a split-nibble
  `embed_row` dequant. Same function names → `serve_qwen3.src` builds against it
  unchanged.

## Measured (rbm21, Qwen3-1.7B, 12 threads, 2k ctx)
| Config | tok/s | vs Q8-scalar baseline |
|---|---|---|
| Q8 scalar (original) | 13.0 | 1.0× |
| Q8 VNNI decode | 15.2 | 1.17× |
| **Q4 VNNI decode** | **19.6** | **1.51×** |
| **Q4 + spec (repeat-10×)** | **28.7** | **2.2×** |

Coherent + correct ("The capital of France is Paris."; counts 1..60). int4 decode
is memory-bytes-bound and Q4 halves the bytes → the ~1.5× decode holds on *any*
prompt; stacking speculation on repetitive output reaches ~2.2×.

## Quality caveat / deployment
Q4 is lossier than Q8 (symmetric /7, no outlier handling). For a 1.7B model this is
usually acceptable (cf. llama.cpp Q4_0) but not free. **Production stays Q8** (VNNI +
spec + warm) for quality; Q4 is deployed on rbm21 as `serve_q4` + `qwen3-q4.bin` for
opt-in speed/size. Choosing Q4 is a quality-vs-speed call for the operator.

Follow-ups: activation-aware / outlier-preserving Q4 (better quality), Q4 KV cache,
self-host port of the q4 builtins.
