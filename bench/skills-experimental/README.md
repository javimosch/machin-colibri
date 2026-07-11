# Skills — in-context reliability for small models

Skills are the cheap way to make a 1B model reliable at a task **without
changing its weights**: curated instructions + few-shot examples that the
server injects into the system prompt when they're relevant. It's
"finetuning" you can read, edit, and version as plain text.

## How it works

`colibri-serve` loads every `*.skill` file in its skills directory
(`COLIBRI_SKILLS`, default `./skills`) at startup. On each chat request it
scores the skills against the last user message by trigger-word overlap and
injects the top matches (default 2) into the system prompt — ahead of any tool
schema. Nothing changes for the client: it's the same OpenAI API, the same
model, just a better prompt.

## Skill file format

```
name: <short-name>
triggers: <space-separated keywords matched against the user message>
priority: <int, breaks ties among equally-matching skills>
---
<the guidance the model sees: rules + a few worked examples>
```

## The shipped skills

- **tool-use** — makes tool calls come out in the exact
  `{"tool_call": {"name": ..., "arguments": {...}}}` shape, with correct/incorrect
  examples. Targets the observed 1B failure of emitting `{"tool_call": "ls", ...}`
  (name as a bare string) or bare `{"name": ...}`.
- **faithful-results** — after a tool result, answer from the result only, no
  invented padding. Targets the observed failure of hallucinating extra files
  after a real `ls`.

## Writing a skill

Keep it short and concrete. A 1B model follows *examples* far better than
prose — show 2-4 correct cases and 1-2 explicitly-wrong ones. Pick triggers
that appear in the kind of request the skill should fire on. Measure: the
`bench/eval_skills.py`-style harness (valid-output rate with vs without the
skill, sampled) tells you whether a skill actually helps before you ship it.
