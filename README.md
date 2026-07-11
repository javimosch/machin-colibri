# colibri

A clean-room LLM inference engine in **pure MFL** ([machin](https://github.com/javimosch/machin)) — zero dependencies, no Python at runtime. Inspired by the philosophy of [JustVugg/colibri](https://github.com/JustVugg/colibri) (single small engine, quantized, self-contained), aimed at small models instead of giant MoEs.

**The challenge:** run a 1B-parameter model on CPU at **20+ tok/s**, matmul included, in MFL.

## Status

- **M0 — correctness: DONE.** fp32 forward pass of llama2.c-format checkpoints (`colibri.src`). Greedy decode of `stories15M` matches karpathy's `run.c` **token-for-token over 200 tokens**. ~66 tok/s single-thread naive (15M model).
- **M1 — real 1B model, int8: DONE.** `colibri_q.src` runs llama2.c *version-2* (Q8_0 group-quantized) checkpoints — GQA, BPE prompt encoding included. On **TinyLlama-1.1B-Chat**: greedy argmax **identical to `runq.c` at all 48 compared positions**, and **2.0–2.1 tok/s single-thread** — faster than the reference scalar C (`runq.c -O2`: 1.67 tok/s) on the same box (i7-class 8-core, DDR4).
- **M2 — the 20 tok/s challenge:** parallel matmul (machin goroutines, inferred race-freedom) + lower-bit quantization. 20 tok/s × 1.17 GB ≈ 23 GB/s memory traffic — right at commodity-DDR4 bandwidth, so int8 threads alone won't be enough; int4 is on the table.

## Run

```sh
mkdir -p models
curl -L -o models/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
machin encode colibri.src > colibri.mfl
machin build colibri.mfl -o colibri
./colibri models/stories15M.bin 200
```

### 1B int8

Convert once with llama2.c's `export.py` (needs a GQA patch for `--hf`: set `n_kv_heads` from `num_key_value_heads` and permute k_proj with kv dims), then:

```sh
machin encode colibri_q.src > colibri_q.mfl
machin build colibri_q.mfl -o colibri_q
./colibri_q models/tinyllama-1.1b-q80.bin 64 "The capital of France is"
```

`DL=1` dumps per-position argmax+logits for diffing against a similarly-patched `runq.c`.

## Design notes

- Weights and activations live in raw `alloc`'d buffers accessed with `peek_f32`/`poke_f32` — storage stays fp32 (4 bytes/weight, same numerics class as llama2.c), math runs in f64.
- RoPE computed on the fly (adjacent-pair rotation), MHA with KV cache, SwiGLU FFN, greedy sampling.
- Prompt encoding (BPE) not yet implemented — generation starts from BOS. M1 item.
