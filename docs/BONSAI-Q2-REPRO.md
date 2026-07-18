# Ternary-Bonsai-1.7B Q2_0 — agent reproduction guide

How to rebuild the **machin-anvil** Ternary-Bonsai-1.7B path from scratch so
another agent gets the same artefacts and behaviour. Companion cert:
[`certs/M16-bonsai-q2.md`](../certs/M16-bonsai-q2.md).

## Goal artefacts

| Artefact | Path | Notes |
|---|---|---|
| GGUF source | `models/bonsai-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf` | ~442 MB, Prism type **42** |
| Anvil checkpoint | `models/bonsai17-q2.bin` | ~462 MB, magic `akQ3`, `GS=128`, `wbits=2` |
| Tokenizer | `models/bonsai17-tok.bin` | vocab **151669** (not 151936) |
| Engine | `qwen3_q2_engine.src` | |
| CLI / server | `bonsai17_cli`, `serve_bonsai17` | OpenAI-compat on `serve_qwen3.src` |
| machin builtins | `dot_q2`, `matmul_q2_batch` | required; also adds `matmul_q4_batch` |

`models/` is gitignored — regenerate locally.

## Prerequisites

- System Python **3.10** with `numpy` + `gguf`  
  (`/usr/bin/python3` — AppImage `python3` often cannot import user site-packages).
- Rebuilt **machin** containing `dot_q2` (see below).
- ~1 GB free disk for GGUF + anvil bin.

## 1. machin builtins (required)

Repo: sibling `../machin` (or wherever `machin` is built from).

Add / keep:

- `mfl_dot_q2` — activations int8, weights 2-bit packed (4 codes/byte, **low bits first**), `w = (q-1)*scale`, `q∈{0,1,2}`
- `mfl_matmul_q2_batch` — same contract as `matmul_q8_batch`, weight row `n/4` bytes
- `mfl_matmul_q4_batch` — was referenced by Q4 engines but missing from stock machin

Wire in `codegen.go`, `types.go`, `guide.go`, then:

```bash
cd ../machin
go build -o ~/.local/bin/machin .
machin guide | grep -E 'dot_q2|matmul_q2'
```

## 2. Download GGUF

```bash
cd machin-anvil
mkdir -p models/bonsai-gguf
# Prefer HF CLI if authenticated; else curl the resolve URL:
hf download prism-ml/Ternary-Bonsai-1.7B-gguf Ternary-Bonsai-1.7B-Q2_0.gguf \
  --local-dir models/bonsai-gguf
# Expected size ≈ 463290464 bytes
```

**Do not** use stock `gguf.GGUFReader` without a type-42 patch — Prism’s
`Q2_0` is **GGML type id 42**, unknown to gguf≤0.19. Both converters monkey-patch
`GGUFReader._build_tensors` (block=128, bytes/block=34 = fp16 scale + 32 code bytes).

## 3. Convert tokenizer + weights

```bash
/usr/bin/python3 tools_bonsai_tok.py \
  models/bonsai-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf \
  models/bonsai17-tok.bin

/usr/bin/python3 tools_bonsai_q2_convert.py \
  models/bonsai-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf \
  models/bonsai17-q2.bin
```

Expected converter stdout (dims must match):

```
config: D=2048 NH=16 NKV=8 HD=128 QD=2048 I=6144 L=28 VOCAB=151669 SEQ=8192 theta=1000000
wrote models/bonsai17-q2.bin  ~484 MB on disk reporting / ~462 MiB ls
```

### Transformation (GGUF → akQ3)

1. Read each type-42 tensor: blocks of 34 bytes → fp16 scale + 128 ternary codes.
2. Unpack codes low-bits-first: `q = (byte >> 2k) & 3`, `w = (q-1)*scale`.
3. Layout: GGUF shape `(ne0,ne1)` with `ne0` innermost → float matrix `[ne1, ne0]` = **`[out, in]`** (HF-style).
4. Re-pack anvil Q2: per row, groups of `GS=128`; scale = maxabs; codes `round(w/scale)+1` clipped to `{0,1,2}`; 4 codes/byte low-first; **fp32** scales (not fp16).
5. Write akQ3 header (`0x616B5133`) + sections identical to Qwen3 Q4 twin:
   embed (tied lm_head) → per-layer norms (attn/ffn/q/k) → output_norm → attn q/k/v/o → ffn gate/up/down.
6. `SEQ=8192` (GGUF has 32k + YaRN; anvil rope has no YaRN yet). `wbits=2`, `GS=128`.

Tensor name map (GGUF → role):

| GGUF | Anvil role |
|---|---|
| `token_embd.weight` | embed + tied lm_head |
| `blk.{i}.attn_norm` / `ffn_norm` | input / post-attn RMSNorm |
| `blk.{i}.attn_q_norm` / `attn_k_norm` | qk-norm |
| `blk.{i}.attn_{q,k,v,output}.weight` | attention projections |
| `blk.{i}.ffn_{gate,up,down}.weight` | SwiGLU MLP |
| `output_norm.weight` | final RMSNorm |

## 4. Build binaries

```bash
export MACHIN=../machin   # path to machin repo (for machweb.src)

machin encode qwen3_q2_engine.src qwen3_cli.src > bonsai17_cli.mfl
machin build bonsai17_cli.mfl -o bonsai17_cli

machin encode $MACHIN/framework/machweb.src qwen3_q2_engine.src qwen_tok.src serve_qwen3.src \
  > serve_bonsai17.mfl
machin build serve_bonsai17.mfl -o serve_bonsai17
```

## 5. Smoke checks

**Greedy text (CLI, pre-tokenized ids):**

```bash
# Encode ChatML with the GGUF BPE (see tools in M16 / prior session), then:
COLIBRI_THREADS=6 ./bonsai17_cli models/bonsai17-q2.bin '<comma-sep ids>'
```

Known-good: capital-of-France prompt → continuation contains `The capital of France is Paris`.

**Server:**

```bash
COLIBRI_THREADS=6 COLIBRI_NOTHINK=1 \
  ./serve_bonsai17 models/bonsai17-q2.bin 8096 models/bonsai17-tok.bin
```

```bash
curl -s http://127.0.0.1:8096/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai17-q2","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32}'
```

## 6. Tool-calling bench

```bash
/usr/bin/python3 tools_bonsai_tools_bench.py --base http://127.0.0.1:8096/v1
```

Writes `certs/M16-bonsai-tools.md` with per-case pass/fail (20-tool catalog,
target tool selection, 0–5 typed arguments).

## Pitfalls (do not regress)

| Mistake | Symptom |
|---|---|
| Use `qwen3-tok.bin` (151936) | OOB / garbage tokens |
| Stock gguf reader without type 42 | `ValueError: 42 is not a valid GGMLQuantizationType` |
| AppImage `python3` for convert | broken numpy / missing gguf |
| Old machin without `dot_q2` | link/compile error on `dot_q2` |
| Expect 32k context | rope wrong past 8k (no YaRN) |
| Transpose GGUF matrices wrong | fluent garbage, no factual answers |

## License / upstream

- Weights: PrismML Ternary-Bonsai-1.7B, Apache 2.0  
  https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf  
- Base arch: Qwen3-1.7B (GQA, SwiGLU, qk-norm, θ=1e6)
