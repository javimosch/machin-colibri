# M4 glue certifications — 2026-07-11

Local server: `COLIBRI_THREADS=6 ./colibri-serve models/tinyllama-1.1b-q80.bin 8090`
(i5-11320H laptop, same binary/model as the M2 benchmarks)

## tau (github.com/javimosch/tau, Zig agent CLI)

Requires tau PR #64 (`TAU_ENDPOINT` override).

```sh
TAU_ENDPOINT="http://127.0.0.1:8090/v1/chat/completions" TAU_API_KEY=local \
  tau --model openai/tinyllama-1.1b-q80 --mode json \
  "What is the capital of France? Answer in one short sentence."
```

Output (streamed through tau's own SSE reassembly):

```
{"chunk":" Paris","done":false}
{"chunk":".","done":false}
{"model":"tinyllama-1.1b-q80","done":true}
```

## roam (github.com/javimosch/roam, remote agent runtime)

No roam changes needed — its stock `--provider openai --api-base` path:

```sh
ROAM_API_KEY=local roam send --local --provider openai \
  --api-base http://127.0.0.1:8090/v1 --model tinyllama-1.1b-q80 \
  --max-iters 2 --tokens 3000 --goal "Say READY and finish."
```

Journal (job f39f16dd): status **done**, 1 iteration, **168 tokens** (from the
server's `usage` accounting):

```
[system] agent loop started [openai] (model=tinyllama-1.1b-q80, maxiter=2, tokens<=3000, shell=0)
[llm]    READY / Finish / Sure, here's the summary: The user has completed the task.
[system] agent ended its turn (finish_reason=stop)
```

## Scope honesty

TinyLlama-1.1B is a toy brain: these certs prove the *plumbing* — OpenAI wire
format, SSE streaming, usage accounting, agent-loop compatibility — not agentic
competence. Reliable tool-calling needs a stronger model (M5: Llama-3.2-1B).
