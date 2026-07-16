# M7 certification — OLMoE-1B-7B in pure MFL (2026-07-13)

**A 7B-class Mixture-of-Experts model running in pure MFL, on a laptop CPU, with
experts streamed from disk — token-identical to the fp32 reference.** This is the
escape from the balanced-roofline wall proven in `docs/PERFORMANCE-FRONTIER.md`:
MoE decouples quality (total params) from speed (active params).

## Model
`allenai/OLMoE-1B-7B-0924` (base): **6.9B total / 1.3B active** — 64 experts per
layer, top-8 routed, 16 layers, D=2048, 16 heads (no GQA), qk-norm, vocab 50304,
RoPE θ=1e4. Downloaded bf16 (13.8 GB) → converted to **7.96 GB** colibri `.bin`:
attn/router/experts int8 (Q8_0, group 64); embed/lm_head/norms fp32.

## Correctness — token-identical to the fp32 ground truth
`tools_olmoe_ref.py` is a numpy fp32 forward reading the HF safetensors directly
(bf16→fp32, experts lazy). The pure-MFL int8 engine reproduces it exactly:

```
prompt : "The capital of France is"   ids [510, 5347, 273, 6181, 310]
numpy fp32 : [7785, 15, 187, 187, 510, 5347, 273, 253, 1986, 2077, 310, 5041]
pure MFL   : [7785, 15, 187, 187, 510, 5347, 273, 253, 1986, 2077, 310, 5041]   ← 12/12 identical
decoded    : "The capital of France is Paris.\n\nThe capital of the United States is Washington"
```
int8 quantization did not flip a single token vs fp32. (Both diverge from the
original colibrì's `ref.json` at generation-token 6 on a near-synonym — a
reduction-order precision difference in how that reference was produced, the same
near-tie behavior documented in M5; the arch is confirmed by the 5-token exact
prefix, the coherent correct completion, and the `q_norm.weight=[2048]` shape.)

## Streaming
Experts live in the mmap'd `.bin` and are read **only when routed** — the OS page
cache is the per-layer LRU. A 12-token run faulted in **622 / 1024** experts; the
warm working set fits in RAM (8 GB model on a 16 GB box; would fit rbm21's 10 GB).

## Speed (local laptop, 8 cores, CPU-capped `taskset`/`nice`, warm cache)
| threads | tok/s |
|---|---|
| 1 | 3.8 |
| 6 | 9.1 |
| 8 | **11.5** |

Per-token weight traffic ≈ 1.5 GB (806 MB experts + 256 MB attn + 412 MB fp32
lm_head); × 11.5 tok/s ≈ **17 GB/s**, near the DDR ceiling — the engine is already
memory-bound at the active-param footprint. (Quantizing lm_head to int8 would cut
~300 MB/token — a further speed lever traded against logit precision; kept fp32
here for the token-identical result.)

## Quantization variants (`tools_olmoe_convert.py <hf> <out> <ebits> <lmbits>`)
| experts | lm_head | .bin size | tokens vs fp32 | tok/s (8 thr, warm) |
|---|---|---|---|---|
| int8 | fp32 | 7.96 GB | identical (12/12) | 9.4 |
| int4 | fp32 | 4.74 GB | identical (12/12) | 7.4 |
| **int8** | **int8** | 7.65 GB | **identical (12/12)** | **14.5** ⚡ best speed |
| **int4** | **int8** | **4.43 GB** | **identical (12/12)** | 11.2 — best footprint |

**Every config is token-identical to the fp32 reference** — quantization never
flipped a token, even int4-experts + int8-lm_head.

Two independent levers, both measured:
- **int8 lm_head** is the big speed win (9.4→14.5, and 7.4→11.2): the fp32 lm_head
  was 412 MB/token, int8 cuts it to 103 MB — decode is memory-bound so this is
  near-linear. Kept the token-identical result too.
- **int4 experts** halve the expert footprint (6.85→3.4 GB) and stay
  token-identical, but are *slower* on this compute-balanced box (the same
  balanced-roofline wall the dense int4 hit — `dot_q4` nibble-unpack compute
  outweighs the memory saved). A footprint lever, not a speed one.

Pick by constraint: **int8+int8** for max speed (14.5 tok/s), **int4+int8** for the
smallest box (4.43 GB, still 11.2 tok/s).

## Why it matters
The dense 1B hit a ~20 tok/s wall priced at *total* params. OLMoE delivers
**7B-class quality at ~1.3B active cost** — 11.5 tok/s on a laptop — and streaming
makes the total model size bounded by disk, not RAM. Same pure-MFL engine, same
`dot_q8` kernel, `mmap_file` streaming. The disruption was in the model, and the
runtime carries it in pure MFL.

## Reproduce
```bash
# download (13.8 GB bf16)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/OLMoE-1B-7B-0924', allow_patterns=['*.safetensors','*.json'], local_dir='olmoe-hf')"
python3 tools_olmoe_ref.py olmoe-hf ref.json          # numpy fp32 ground truth
python3 tools_olmoe_convert.py olmoe-hf models/olmoe-q8.bin   # -> 7.96 GB int8 .bin
machin encode olmoe_engine.src > olmoe.mfl && machin build olmoe.mfl -o olmoe
COLIBRI_THREADS=8 ./olmoe models/olmoe-q8.bin
```
