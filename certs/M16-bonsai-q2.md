# M16 — Ternary-Bonsai-1.7B Q2_0 in machin-anvil (2026-07-18)

First ternary/Q2 path in anvil: Prism **Ternary-Bonsai-1.7B** GGUF `Q2_0`
(type 42, group-128, `w=(q-1)*scale`) converted to akQ3 and run with new
machin builtins `dot_q2` / `matmul_q2_batch`.

**Full agent reproduction steps:** [`docs/BONSAI-Q2-REPRO.md`](../docs/BONSAI-Q2-REPRO.md)  
**Tool-calling bench:** [`certs/M16-bonsai-tools.md`](M16-bonsai-tools.md) (from `tools_bonsai_tools_bench.py`)

## What shipped

| Piece | Path |
|---|---|
| machin builtins | `dot_q2`, `matmul_q2_batch` (+ missing `matmul_q4_batch`) in machin `codegen.go` / `types.go` / `guide.go` |
| Converter | `tools_bonsai_q2_convert.py` — GGUF→`models/bonsai17-q2.bin` (akQ3, `GS=128`, `wbits=2`) |
| Tokenizer | `tools_bonsai_tok.py` — GGUF BPE→`models/bonsai17-tok.bin` (vocab **151669**) |
| Engine | `qwen3_q2_engine.src` — Q4 twin with weight strides `/4`, `dot_q2`, ternary `embed_row` |
| Binaries | `bonsai17_cli`, `serve_bonsai17` (same API surface as `serve_qwen3`) |

Checkpoint ~**462 MB** (vs Qwen3-1.7B Q8 1.83 GB / Q4 924 MB).

## Smoke (this laptop: i5-11320H, 6 threads)

```bash
COLIBRI_THREADS=6 ./bonsai17_cli models/bonsai17-q2.bin '<prompt token ids>'
```

Prompt: Qwen chat asking for the capital of France (one word).
Greedy 12-token continuation decoded to:

`<|im_end|>` + newline + `The capital of France is Paris`

So the answer is correct; chat-template quirks (leading `im_end`) are expected
with a bare assistant-prefill and no thinking-mode handling yet.

End-to-end (prefill+decode) on that short run ≈ **2.2 tok/s** on this box —
decode-only should be higher; not yet micro-benched against Q4/Q8 on the same
prompt.

## Notes / limits

- **YaRN disabled:** GGUF advertises 32k with YaRN×4 from 8k; anvil rope is
  plain θ=1e6, so header `SEQ=8192`. Fine for short agent turns.
- **Vocab 151669** (not 151936) — must use `bonsai17-tok.bin`, not `qwen3-tok.bin`.
- **machin rebuild required** for `dot_q2` (installed from `/home/jarancibia/ai/machin`).
- Kernels are scalar (correctness first); VNNI/LUT later if decode is still
  bandwidth-bound after measuring.

## Rebuild

```bash
# machin (once, after pulling builtin patches)
cd ../machin && go build -o ~/.local/bin/machin .

cd ../machin-anvil
python3 tools_bonsai_q2_convert.py   # needs models/bonsai-gguf/*.gguf
python3 tools_bonsai_tok.py
machin encode qwen3_q2_engine.src qwen3_cli.src > bonsai17_cli.mfl && machin build bonsai17_cli.mfl -o bonsai17_cli
machin encode $MACHIN/framework/machweb.src qwen3_q2_engine.src qwen_tok.src serve_qwen3.src \
  > serve_bonsai17.mfl && machin build serve_bonsai17.mfl -o serve_bonsai17

./serve_bonsai17 models/bonsai17-q2.bin 8096 models/bonsai17-tok.bin
```
