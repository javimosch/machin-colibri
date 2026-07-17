# M13 — AVX-VNNI int8 kernel + the 2× compounding (2026-07-17)

M11 found decode is memory-latency-bound; M12 shipped speculative decoding but
capped it at ~1.6× because the batched forward (which speculation rides on) was
only 1.85× decode — the int8 dot kernel was compute-bound. This closes that gap.

## The kernel
`mfl_q8_dot` (shared by `dot_q8` and `matmul_q8_batch`) now uses **`vpdpbusd`**
(one instruction = 32 int8 MACs into int32) on AVX-VNNI, replacing the scalar
accumulate. `vpdpbusd` is unsigned×signed, but Q8_0 is signed×signed, so the
activation is offset to unsigned (`XOR 0x80` == +128) and the `+128·Σw` bias is
subtracted back (`Σw` via `vpdpbusd` against a ones vector). The per-group int32
sum is **exact integer arithmetic identical to the scalar path** → output is
**bit-identical**, zero accuracy cost. Scalar fallback when no VNNI or `gs%32≠0`.
Machin core: `codegen.go` prelude, guarded `__AVXVNNI__ / __AVX512VNNI__`, x86_64
non-wasm. (Branch `vnni-dot-q8`; self-host port + `dot_q4` are follow-ups.)

## Why decode gains little but prefill gains a lot
int8 matmul reads 1 byte and does 1 MAC per element — memory and compute are 1:1
coupled. **Decode** (B=1, no weight reuse) is memory-latency-bound, so a faster
MAC only helps at the margin: **13 → 15.2 tok/s (+17%)**. **Prefill/batched** (B≫1,
weight read once, reused) is compute-bound, so `vpdpbusd` lands fully:
**33 → 47 tok/s (+42%)**.

## The compounding — this is the point
Speculative decoding (M12) verifies K drafts in one batched forward. VNNI speeds
that batched forward, so the two multiply:

| Config | tok/s (repeat-10× task) | vs baseline |
|---|---|---|
| Scalar, no spec (baseline) | 13.4 | 1.0× |
| VNNI alone | 15.7 | 1.17× |
| Spec alone | 16.8 | 1.25× |
| **VNNI + spec** | **26.5** | **1.98×** |

Bit-identical output throughout. On highly-repetitive/echoing output (code edits,
structured text — the agentic-client common case) the stack now runs ~2× the
original decode rate. On novel/chat text it degrades gracefully to VNNI-decode
(~1.17×). Deployed on rbm21 `qwen3.service` (VNNI binary + `COLIBRI_SPEC=8` + warm).

## Correcting M11
M11 said "VNNI won't help decode." Precise version: it barely helps *pure* decode
(+17%, memory-bound), but it helps the *batched* path a lot (+42%), and via
speculation that batched speedup reaches decode — the 2× the M11 teardown said was
only reachable by "reuse weights across tokens." That's exactly what spec+VNNI do.
