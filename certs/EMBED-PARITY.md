# EMBED-PARITY — Qwen3-Embedding-0.6B int8 engine vs numpy fp32 oracle

Engine: `qwen3e` (pure MFL, int8 Q8_0, 2 threads) · Model: `models/qwen3e-q8.bin` · Oracle: `tools_qwen3e_ref.py` (numpy fp32, HF tokenizers)

Semantics: last-token pooling after final RMSNorm, EOS `<|endoftext|>` (151643) appended, L2-normalized, D=1024.

## Per-text cosine (gate: >= 0.99 each)

| text | tokens | ids match | cosine(engine, oracle) | engine ms |
|---|---|---|---|---|
| short-en | 8 | yes | 0.998823 | 1419 |
| short-en-2 | 9 | yes | 0.997866 | 1721 |
| paris-para | 48 | yes | 0.998886 | 7585 |
| long-en | 72 | yes | 0.998923 | 12338 |
| french | 25 | yes | 0.998711 | 4424 |
| spanish | 26 | yes | 0.998441 | 4837 |
| chinese | 23 | yes | 0.998753 | 3705 |
| code | 27 | yes | 0.999206 | 4204 |
| json-code | 35 | yes | 0.999357 | 6196 |
| query-instruct | 26 | yes | 0.998929 | 4888 |

Worst cosine: **0.997866** — gate PASS

## Pairwise-similarity-matrix agreement

- Full per-row ranking identical: **7/10** rows
- Pairwise order agreement (all (i; a,b) triples): **357/360** (99.17%)
- Max abs deviation between similarity matrices: **0.016888**

## Engine pairwise similarity matrix

| |short-en|short-en-2|paris-para|long-en|french|spanish|chinese|code|json-code|query-instruct|
|---|---|---|---|---|---|---|---|---|---|---|
| short-en |1.000|0.331|0.765|0.225|0.678|0.126|0.765|0.152|0.126|0.722|
| short-en-2 |0.331|1.000|0.232|0.243|0.174|0.135|0.206|0.064|0.172|0.066|
| paris-para |0.765|0.232|1.000|0.249|0.694|0.094|0.862|0.132|0.175|0.699|
| long-en |0.225|0.243|0.249|1.000|0.194|0.406|0.184|0.220|0.265|0.130|
| french |0.678|0.174|0.694|0.194|1.000|0.121|0.707|0.098|0.157|0.653|
| spanish |0.126|0.135|0.094|0.406|0.121|1.000|0.054|0.164|0.259|0.046|
| chinese |0.765|0.206|0.862|0.184|0.707|0.054|1.000|0.076|0.129|0.663|
| code |0.152|0.064|0.132|0.220|0.098|0.164|0.076|1.000|0.342|0.076|
| json-code |0.126|0.172|0.175|0.265|0.157|0.259|0.129|0.342|1.000|0.091|
| query-instruct |0.722|0.066|0.699|0.130|0.653|0.046|0.663|0.076|0.091|1.000|

## Throughput (glane indexing predictor)

50 short paragraphs (avg ~41 tokens each, batched prefill), sequential, `COLIBRI_THREADS=2`, `nice -n 19`, on the laptop:

```
{"chunks":50,"tokens":2067,"ms":336604,"chunks_per_s":0.148543,"tok_per_s":6.14075,"threads":2}
```

**~0.15 chunks/s (~6.7 s per ~41-token paragraph) at 2 threads.** Scales with threads and prompt batching; rbm21 (i5-13400T, AVX-VNNI) prefills the 1.7B chat model ~3x faster per weight-byte than this laptop, and the 0.6B moves ~3x fewer bytes.

## Serve smoke (OpenAI-compatible)

`COLIBRI_THREADS=2 ./serve_qwen3e models/qwen3e-q8.bin 8095 models/qwen3e-tok.bin`

- `GET /health` -> `{"ok":true,"model":"qwen3-embedding-0.6b","dim":1024}`
- `POST /v1/embeddings {"input":"The capital of France is Paris."}` -> 1024-dim vector, byte-identical values to the CLI for the same text.
- `POST /v1/embeddings {"input":["hello world","bonjour le monde"],"dimensions":8}` -> two indexed embeddings, MRL-truncated to 8 dims and re-L2-normalized; usage.prompt_tokens counted.
- Optional `"instruct": "<task>"` field wraps each input as `Instruct: {task}\nQuery:{text}` (query side; doc side = raw text).

Cold load: 134 ms (mmap).
