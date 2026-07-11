# colibri

A clean-room LLM inference engine in **pure MFL** ([machin](https://github.com/javimosch/machin)) — zero dependencies, no Python at runtime. Inspired by the philosophy of [JustVugg/colibri](https://github.com/JustVugg/colibri) (single small engine, quantized, self-contained), aimed at small models instead of giant MoEs.

**The challenge:** run a 1B-parameter model on CPU at **20+ tok/s**, matmul included, in MFL.

## Status

- **M0 — correctness: DONE.** fp32 forward pass of llama2.c-format checkpoints (`colibri.src`). Greedy decode of `stories15M` matches karpathy's `run.c` **token-for-token over 200 tokens**. ~66 tok/s single-thread naive (15M model).
- **M1 — real 1B model, int8: DONE.** `colibri_q.src` runs llama2.c *version-2* (Q8_0 group-quantized) checkpoints — GQA, BPE prompt encoding included. On **TinyLlama-1.1B-Chat**: greedy argmax **identical to `runq.c` at all 48 compared positions**, and **2.0–2.1 tok/s single-thread** — faster than the reference scalar C (`runq.c -O2`: 1.67 tok/s) on the same box (i7-class 8-core, DDR4).
- **M2 — the 20 tok/s challenge: HIT (burst) / 19.0 sustained.** Worker-pool parallel matmul (goroutines + packed-int jobs over channels) + machin's new `dot_i8`/`peek_i8` builtins (drove them upstream: [machin#435](https://github.com/javimosch/machin/pull/435)).

## The numbers (TinyLlama-1.1B-Chat, Q8_0, greedy)

| Setup | tok/s |
|---|---|
| `runq.c -O2` reference (same box, 1 thread) | 1.67 |
| colibri single-thread | 6.8 |
| **colibri 6 threads, cold** (i5-11320H 4C/8T laptop) | **21.8** |
| colibri 6 threads, sustained 200 tok (cold start → 91 °C) | 19.0 |
| colibri 12 threads on i5-13400T, `nice -19` **alongside** a training job using 8 cores | 17.9 |

Interactive use (the actual agent/chat pattern — short bursts) runs 20+; fully-sustained generation on the thermally-limited laptop lands at 19.0. Output is greedy-argmax-identical to `runq.c` at every compared position in all configurations.

The 2.1 → 21.8 arc: `-O3 -march=native` (2.5) → worker pool (6.1) → `peek_i8` (8.4) → 512-bit vectors (9.9) → int8 activations (15.7) → `dot_i8` + fused qkv/w1·w3 jobs (21.8). An int4 path (v3 `ak43` format, `export_q4.py`) works and is checked in, but its nibble-unpack is compute-bound — int8 is faster end-to-end on current autovectorizers; parked.

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
