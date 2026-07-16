# M8 certification — xLAM-1b-fc-r (agentic, function-calling) in pure MFL (2026-07-13)

A **1B function-calling-specialized** model (Salesforce xLAM-1b-fc-r, deepseek-1.3b
/ dense Llama arch) running in pure MFL — token-identical to fp32, ~5× smaller
than the OLMoE MoE, with **working OpenAI function-calling** for agentic clients.

## Model
`Salesforce/xLAM-1b-fc-r` — LlamaForCausalLM, 1.35B dense: D=2048, 24 layers,
16 heads (no GQA), inter 5504, vocab 32256, RoPE θ=1e5 with **linear scaling ÷4**,
RMSNorm (no qk-norm), separate lm_head. BFCL ≈ 75% (beats Claude-3-Opus/GPT-3.5
on tool-use). Download 2.69 GB bf16 → **1.62 GB** int8 `.bin`.

## Correctness — token-identical to fp32
`tools_xlam_ref.py` (numpy fp32 forward reading safetensors directly) vs the pure-
MFL int8 engine:
```
prompt: "The capital of France is"  ids [32013,546,6075,280,7243,317]
numpy fp32 : [8873,13,185,185,185,546,6075,280,11384,317,18904,13]
pure MFL   : [8873,13,185,185,185,546,6075,280,11384,317,18904,13]   ← 12/12 identical
decoded    : " Paris.\n\n\nThe capital of Germany is Berlin."
```
Bug found + fixed en route: the quant scratch (`xqb`/`xsb`) must be sized for the
*widest* matmul input — xLAM's FFN down_proj takes II=5504 > D=2048 (unlike OLMoE
where II<D), so a D-sized buffer overflowed → NaN. Fixed to `max(D, II)`.

## Tokenizer — pure MFL, exact
`xlam_tok.src`: the deepseek byte-level BPE (multi-split pretokenizer — letters /
punct / **individual digits** / newlines — + ranked-merge BPE + BOS 32013).
Validated against the `tokenizers` library:
```
"The capital of France is" -> [32013,546,6075,280,7243,317]   exact
"get_weather(city=Paris)"  -> [32013,703,62,31125,7,23861,28,3693,262,8]  exact
"Hello, world! 123"        -> [32013,17535,11,1835,0,207,16,17,18]  exact (individual digits)
```

## Server — chat + function-calling (OpenAI-compatible)
`serve_xlam.src`: deepseek chat template (`### Instruction:` / `### Response:`) +
prompt-based tool injection + tool-call JSON parsing → OpenAI `tool_calls`.
Verified via curl:
```
"What is the capital of France?"          -> "The capital of France is Paris."
"weather in Tokyo?" (+get_weather tool)   -> tool_calls: get_weather({"city":"Tokyo"})
```
The function-calling is real and correct — the deliverable "pi-capable agentic
model" is met at the model+API level.

## Speed / footprint
~11 tok/s (8 threads, warm), 1.62 GB on disk (int8) — **~5× smaller than OLMoE**
(the reason for choosing it). Similar decode speed (24 layers + inter 5504 ≈
OLMoE's active FFN). Idle-release: 27 MB. Deploy: `deploy/xlam.user.service`
(user systemd, :8092).

## Making pi practical: batched prefill + prefix cache (ported 2026-07-13)

pi resends a large agentic system prompt every turn. The engine now has both
levers the dense engine has:

- **Batched prefill** (`batch_layers`/`prefill_batch`, `COLIBRI_TILE`, off with
  `COLIBRI_NOBATCH=1`) — a tile of positions runs through each layer together so
  each weight is read once per tile. **Verified bit-identical to per-token** on
  coherent input ("…Italy is Rome. The capital of Spain is Madrid").
- **Single-slot KV prefix cache** (`cache_buf`/`cache_common`) — persists the KV
  across requests; a repeated prefix is skipped entirely.

### Discovery — measured on this laptop (632-token prompt, 8 threads):
| | cold prefill | repeated prompt (prefix cache) |
|---|---|---|
| per-token | 53.5 s | — |
| **batched** | **40.9 s** | **0.45 s** |

Two honest findings:
1. **Batched prefill is only ~1.3× on this box** (53.5→40.9 s), *not* the large
   speedup one expects. Same **balanced-roofline wall** as everywhere else in
   this repo: batching trades weight-*reads* for compute (B× the matmul FLOPs),
   and this CPU is compute-balanced, so the shared weight-read doesn't dominate.
   Batched prefill is the right lever on a memory-bound box; here it's marginal.
2. **The prefix cache is the real multi-turn win: ~91×** (40.9 s → 0.45 s) on a
   repeated prompt. End-to-end through the server with a 40×-repeated system
   prompt + a tool: **req1 (cold) 20.7 s → req2 (warm) 2.0 s**, both returning the
   correct `get_weather({"city":"Paris"})`.

**Net for pi:** the first agentic turn pays the cold prefill (~tens of seconds for
a multi-thousand-token system prompt), but every turn after reuses the cached
prefix and only prefills the delta — so the loop is practical. The cold first turn
is a floor set by the hardware (632 tokens × 1.35B params of compute), not the
engine.
