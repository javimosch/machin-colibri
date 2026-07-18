#!/usr/bin/env python3
# Parity certification: engine int8 embedding vs numpy fp32 oracle over diverse
# texts. Per-text cosine (gate: >= 0.99) + pairwise-similarity-matrix ranking
# agreement. Writes certs/EMBED-PARITY.md. Sequential, nice'd by the caller.
import json, os, subprocess, sys, numpy as np

HF = "qwen3-embed-hf"
BIN = "./qwen3e"
MODEL = "models/qwen3e-q8.bin"
THREADS = os.environ.get("COLIBRI_THREADS", "2")

TEXTS = [
    ("short-en", "The capital of France is Paris."),
    ("short-en-2", "Photosynthesis converts sunlight into chemical energy."),
    ("paris-para", "Paris is the capital and largest city of France. Known for the Eiffel Tower and the Louvre, it has been a major European center of finance, diplomacy, commerce, culture and science since the 17th century."),
    ("long-en", "Gradient descent is an iterative optimization algorithm for finding a local minimum of a differentiable function. The idea is to take repeated steps in the opposite direction of the gradient of the function at the current point, because this is the direction of steepest descent. The learning rate controls the step size, and choosing it poorly can cause divergence or painfully slow convergence."),
    ("french", "La tour Eiffel est un monument emblematique situe a Paris, la capitale de la France."),
    ("spanish", "El aprendizaje automatico permite a las computadoras aprender patrones a partir de datos sin ser programadas explicitamente."),
    ("chinese", "巴黎是法国的首都，也是法国最大的城市，以埃菲尔铁塔和卢浮宫闻名。"),
    ("code", "def cosine(a, b):\n    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))"),
    ("json-code", "{\"endpoint\": \"/v1/embeddings\", \"method\": \"POST\", \"body\": {\"input\": [\"text\"], \"model\": \"qwen3-embedding\"}}"),
    ("query-instruct", "what is the capital of france"),
]
INSTRUCT_FOR = {"query-instruct": "Given a web search query, retrieve relevant passages that answer the query"}

def run_engine(text, task):
    cmd = [BIN, "embed", MODEL, text]
    if task: cmd += ["--instruct", task]
    env = dict(os.environ, COLIBRI_THREADS=THREADS)
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)

def run_oracle(text, task):
    cmd = ["nice", "-n", "19", "python3", "tools_qwen3e_ref.py", HF, text]
    if task: cmd += ["--instruct", task]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)

def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

rows, E, O = [], [], []
for name, text in TEXTS:
    task = INSTRUCT_FOR.get(name)
    e = run_engine(text, task)
    o = run_oracle(text, task)
    ev = np.array(e["embedding"], np.float64); ov = np.array(o["embedding"], np.float64)
    ids_match = e["ids"] == o["ids"]
    c = cos(ev, ov)
    rows.append((name, len(e["ids"]), ids_match, c, e["ms"]))
    E.append(ev); O.append(ov)
    print(f"{name:16s} tokens={len(e['ids']):3d} ids_match={ids_match} cosine={c:.6f} engine_ms={e['ms']}")

E = np.stack(E); O = np.stack(O)
SE = E @ E.T; SO = O @ O.T
n = len(TEXTS)
# ranking agreement: per row, order of the other texts by similarity
rank_ok = 0; pair_ok = 0; pair_tot = 0
for i in range(n):
    oe = np.argsort(-np.delete(SE[i], i)); oo = np.argsort(-np.delete(SO[i], i))
    if list(oe) == list(oo): rank_ok += 1
    for a in range(n):
        for b in range(a+1, n):
            if a == i or b == i: continue
            pair_tot += 1
            if (SE[i][a] > SE[i][b]) == (SO[i][a] > SO[i][b]): pair_ok += 1
mad = float(np.abs(SE - SO).max())

os.makedirs("certs", exist_ok=True)
with open("certs/EMBED-PARITY.md", "w") as f:
    f.write("# EMBED-PARITY — Qwen3-Embedding-0.6B int8 engine vs numpy fp32 oracle\n\n")
    f.write(f"Engine: `qwen3e` (pure MFL, int8 Q8_0, {THREADS} threads) · Model: `{MODEL}` · Oracle: `tools_qwen3e_ref.py` (numpy fp32, HF tokenizers)\n\n")
    f.write("Semantics: last-token pooling after final RMSNorm, EOS `<|endoftext|>` (151643) appended, L2-normalized, D=1024.\n\n")
    f.write("## Per-text cosine (gate: >= 0.99 each)\n\n")
    f.write("| text | tokens | ids match | cosine(engine, oracle) | engine ms |\n|---|---|---|---|---|\n")
    for name, nt, im, c, ms in rows:
        f.write(f"| {name} | {nt} | {'yes' if im else 'NO'} | {c:.6f} | {ms} |\n")
    worst = min(r[3] for r in rows)
    f.write(f"\nWorst cosine: **{worst:.6f}** — gate {'PASS' if worst >= 0.99 else 'FAIL'}\n\n")
    f.write("## Pairwise-similarity-matrix agreement\n\n")
    f.write(f"- Full per-row ranking identical: **{rank_ok}/{n}** rows\n")
    f.write(f"- Pairwise order agreement (all (i; a,b) triples): **{pair_ok}/{pair_tot}** ({100.0*pair_ok/pair_tot:.2f}%)\n")
    f.write(f"- Max abs deviation between similarity matrices: **{mad:.6f}**\n\n")
    f.write("## Engine pairwise similarity matrix\n\n| |" + "|".join(t[0] for t in TEXTS) + "|\n")
    f.write("|---|" + "---|"*n + "\n")
    for i in range(n):
        f.write(f"| {TEXTS[i][0]} |" + "|".join(f"{SE[i][j]:.3f}" for j in range(n)) + "|\n")
print(f"\nrank rows identical: {rank_ok}/{n}, pairwise agreement {pair_ok}/{pair_tot}, max|dS|={mad:.6f}")
print("wrote certs/EMBED-PARITY.md")
worst = min(r[3] for r in rows)
sys.exit(0 if worst >= 0.99 else 2)
