# M6 certification — server-side tool calling + systemd (2026-07-11)

## Prompt-based tool calling (works with any model, no native tool tokens)

1B models don't reliably emit native OpenAI `tool_calls`, so the server does
what llama.cpp/ollama do for small models: inject the tool schemas + an output
contract into the system prompt, generate, then parse the model's JSON back
into a proper OpenAI `tool_calls` response with `finish_reason: "tool_calls"`.

Request adds standard OpenAI `tools: [{type:function, function:{name,
description, parameters}}]`. The server:
- appends a tool instruction + one worked example to the system prompt
- renders prior assistant `tool_calls` and role:"tool" results back into the
  chat template (the multi-turn round-trip)
- parses the model output tolerantly — 1B models drift between formats, so the
  parser accepts all of: `{"tool_call":{"name","arguments"}}`,
  `{"tool_call":"NAME","arguments":{}}`, `{"name","arguments"|"parameters"}`,
  `{"function":...}`, and strips ```json fences / surrounding prose
  (extract_obj = balanced-brace scan around the anchor). 7/7 format variants
  parse in the unit test.

Both non-streaming (single JSON with `tool_calls`) and streaming (one SSE
`delta.tool_calls` chunk in the shape tau reassembles) are supported.

## Certs (rbm21, Llama-3.2-1B-Instruct)

- curl direct: request with a `bash` tool → response
  `content:null, tool_calls:[{function:{name:"bash",arguments:"{\"command\":\"ls\"}"}}],
  finish_reason:"tool_calls"`, and `usage.prompt_tokens_cached:137` (prefix
  cache reused the tool-schema prompt).
- **tau `--no-stream --tools bash,read,ls` full round-trip**: model emits
  tool_call → tau executes `ls` locally → returns `note.txt` (the real file) →
  server templates the tool result back → model answers in plain text citing
  `note.txt`. End-to-end agent loop over a pure-MFL server.
- non-tool regression: "Capital of Japan? One word." → "Tokyo." (finish stop).

### Honesty

The plumbing (OpenAI tools wire format, multi-turn tool-result templating,
tolerant parse) is solid. Answer *quality* is 1B-model-limited: after the tool
result the model padded its file list with hallucinated entries. A 3B/8B model
on the same server would be materially more reliable; the server is the same.

## systemd unit (deploy/colibri.service)

`/etc/systemd/system/colibri.service` on rbm21: Type=simple, Restart=on-failure,
Nice=15, MemoryMax=3G, enabled (survives reboot). This also fixed a real
operational issue: an ssh-backgrounded server (`nohup ... &`) died when its
launching ssh channel closed; under systemd it's fully detached and
auto-restarts. Manage with `systemctl {status,restart,stop} colibri`;
logs via `journalctl -u colibri`.

Deploy a new build: scp the emitted C, `gcc ... -lssl -lcrypto -o serve_l3.new`,
`mv serve_l3.new serve_l3 && systemctl restart colibri`.
