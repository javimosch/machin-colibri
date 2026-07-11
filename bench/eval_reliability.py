import json, urllib.request, sys

# Eval on rbm21 (fast, idle). Two metrics, each discriminating:
#  A. tool-call CANONICAL-exactness: does the raw model text use the exact
#     {"tool_call":{"name":...,"arguments":{...}}} shape (not a variant the
#     tolerant parser rescues)? Skills should raise this.
#  B. faithfulness: given a tool result listing exactly 3 known files, does the
#     answer mention ONLY those files (no hallucinated extras)? The M6 failure.
BASE = "http://127.0.0.1:8090/v1/chat/completions"
TOOLS = [
  {"type":"function","function":{"name":"bash","description":"run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
  {"type":"function","function":{"name":"ls","description":"list a directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
]
KNOWN = ["alpha.txt", "beta.md", "gamma.py"]
TOOL_RESULT = "\n".join(KNOWN)

def chat(msgs, tools=None, seed=0, mt=80, temp=0.7):
    payload = {"messages":msgs, "max_tokens":mt, "temperature":temp, "seed":seed}
    if tools: payload["tools"] = tools
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())

def faithful_rate(n=12):
    # Multi-turn: user asks to list, assistant called ls, tool returned KNOWN,
    # now assistant must report the files. Count answers that hallucinate.
    ok = 0
    for s in range(n):
        msgs = [
          {"role":"user","content":"List the files in this folder and tell me what's here."},
          {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"ls","arguments":"{\"path\": \".\"}"}}]},
          {"role":"tool","tool_call_id":"call_1","content":TOOL_RESULT},
        ]
        try:
            r = chat(msgs, seed=200+s, mt=90, temp=0.7)
            ans = (r["choices"][0]["message"].get("content") or "").lower()
        except Exception:
            continue
        # hallucination check: any plausible filename token NOT in KNOWN
        import re
        names = set(re.findall(r"[a-z0-9_\-]+\.[a-z]{1,4}", ans))
        known = set(x.lower() for x in KNOWN)
        extra = names - known
        # faithful = mentions >=2 of the known files AND invents no other filename
        hit = len(names & known)
        if hit >= 2 and not extra: ok += 1
    return ok, n

VALID = {"bash","ls"}
TOOLP = [
  "List the files in the current directory.",
  "How many lines are in main.c? Use a tool.",
  "Show me the running processes.",
]
def toolval_rate(n=6):
    ok = 0; total = 0
    for pr in TOOLP:
        for s in range(n):
            total += 1
            try:
                r = chat([{"role":"user","content":pr}], tools=TOOLS, seed=500+s, mt=64, temp=0.7)
                ch = r["choices"][0]
                if ch.get("finish_reason") == "tool_calls":
                    fn = ch["message"]["tool_calls"][0]["function"]
                    if fn["name"] in VALID and isinstance(json.loads(fn["arguments"]), dict): ok += 1
            except Exception:
                pass
    return ok, total

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    label = sys.argv[2] if len(sys.argv) > 2 else "cond"
    fok, fn = faithful_rate(n)
    print("%s faithful: %d/%d = %d%%" % (label, fok, fn, 100*fok//fn if fn else 0)); sys.stdout.flush()
    tok, tn = toolval_rate(max(4, n//2))
    print("%s toolval : %d/%d = %d%%" % (label, tok, tn, 100*tok//tn if tn else 0)); sys.stdout.flush()
