# colibri

A clean-room LLM inference engine in **pure MFL** ([machin](https://github.com/javimosch/machin)) — zero dependencies, no Python at runtime. Inspired by the philosophy of [JustVugg/colibri](https://github.com/JustVugg/colibri) (single small engine, quantized, self-contained), aimed at small models instead of giant MoEs.

**The challenge:** run a 1B-parameter model on CPU at **20+ tok/s**, matmul included, in MFL.

## Status

- **M0 — correctness: DONE.** fp32 forward pass of llama2.c-format checkpoints. Greedy decode of `stories15M` matches karpathy's `run.c` **token-for-token over 200 tokens**. ~66 tok/s single-thread naive (15M model).
- **M1 — real 1B model, int8, honest single-thread tok/s:** next.
- **M2 — the 20 tok/s challenge:** parallel matmul (machin goroutines, inferred race-freedom) + quantization on defined hardware.

## Run

```sh
mkdir -p models
curl -L -o models/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
machin encode colibri.src > colibri.mfl
machin build colibri.mfl -o colibri
./colibri models/stories15M.bin 200
```

## Design notes

- Weights and activations live in raw `alloc`'d buffers accessed with `peek_f32`/`poke_f32` — storage stays fp32 (4 bytes/weight, same numerics class as llama2.c), math runs in f64.
- RoPE computed on the fly (adjacent-pair rotation), MHA with KV cache, SwiGLU FFN, greedy sampling.
- Prompt encoding (BPE) not yet implemented — generation starts from BOS. M1 item.
