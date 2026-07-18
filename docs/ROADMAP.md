# Roadmap — stable OSS self-hosted agentic stack

**North star:** machin-anvil is a **stable open-source, self-hosted** LLM runtime
where **two main agentic models** are fully tested at **>20 tok/s on cheap
hardware** (low-power desktop / laptop-class CPU, no GPU required).

## Definition of done

| Criterion | Meaning |
|---|---|
| Stable OSS | Tagged release, docs site, changelog, reproducible convert+build, systemd units |
| Self-hosted | OpenAI-compatible `/v1` on localhost; no cloud API key; data stays on box |
| Two main agentic models | General instruct + tools-capable; both drive real agent loops (tau/pi) |
| Fully tested | Cert docs with token parity / tool benches + measured tok/s on reference HW |
| >20 tok/s | Sustained greedy decode on **rbm21-class** cheap CPU (i5-13400T / similar DDR4) |

## Scoreboard (2026-07)

| Model | Role | Decode (cheap HW) | Tools / agent | Status vs goal |
|---|---|---|---|---|
| **Llama-3.2-1B-Instruct** Q8 | General instruct | **~21 tok/s** (rbm21) | Glue via serve; tau/roam OK | ✅ speed · ✅ certified |
| **Qwen3-1.7B** Q4 VNNI | Instruct + tools | **~19.6 tok/s** (~20); 28.7 w/ spec | ChatML tools + `tool_choice` | ✅ ~20 · ✅ serving |
| **xLAM-1b-fc-r** Q8 | Tool specialist | ~11 tok/s | Strong FC; BFCL-class | ✅ tools · ❌ speed |
| **Ternary-Bonsai-1.7B** Q2 | Dense ternary / tools | laptop cold tools slow; density win | **12/12** 20-tool bench | ✅ tools · ⏳ 20 tok/s |

**Closest pair to the goal today:** Llama-3.2-1B + Qwen3-1.7B Q4 — both near or
above 20 tok/s on cheap hardware, with OpenAI serving and agent glue. Closing
the gap means (1) locking a tagged “agent pair” release, (2) pushing the second
tool model (Qwen3-Q4 or a faster Bonsai path) through the same >20 tok/s +
agent-loop certification as Llama.

## Next milestones

1. **v0.1 stable** — document + package the Llama + Qwen3-Q4 pair; publish GH Pages; changelog cadence.
2. **Agent-loop cert** — measured tau/pi edit→run→verify on both models at >20 tok/s warm.
3. **Second tool model ≥20 tok/s** — either Q4/VNNI polish on Qwen3 or ternary kernels that hit the bandwidth win on rbm21.
4. **Release hygiene** — `ANVIL_*` env aliases, model download scripts, one-command serve.

## Hardware reference

Primary numbers: **rbm21** — Intel i5-13400T, AVX-VNNI, DDR4. Secondary: laptop
i5-11320H (Iris Xe only) for ternary/Bonsai development.
