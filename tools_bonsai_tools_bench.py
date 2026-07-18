#!/usr/bin/env python3
"""Tool-calling bench for Ternary-Bonsai-1.7B via anvil OpenAI-compatible server.

Catalog of 20 tools; each case asks for one target tool, with 0..5 typed args.
Writes certs/M16-bonsai-tools.md and prints a summary.
"""
from __future__ import annotations

import argparse, json, time, urllib.request
from pathlib import Path

OUT = Path("certs/M16-bonsai-tools.md")

# 20 tools — distinct verbs so selection is unambiguous
TOOLS = [
    {"name": "get_weather", "desc": "Get current weather for a city",
     "params": {"city": ("string", True), "units": ("string", False)}},
    {"name": "get_time", "desc": "Get the current local time for a timezone",
     "params": {"timezone": ("string", True)}},
    {"name": "add_numbers", "desc": "Add two integers and return the sum",
     "params": {"a": ("integer", True), "b": ("integer", True)}},
    {"name": "multiply", "desc": "Multiply two floating-point numbers",
     "params": {"x": ("number", True), "y": ("number", True)}},
    {"name": "set_alarm", "desc": "Set an alarm at an hour with optional label",
     "params": {"hour": ("integer", True), "minute": ("integer", True), "label": ("string", False)}},
    {"name": "search_web", "desc": "Search the public web for a query string",
     "params": {"query": ("string", True), "limit": ("integer", False)}},
    {"name": "send_email", "desc": "Send an email message to a recipient",
     "params": {"to": ("string", True), "subject": ("string", True), "body": ("string", True)}},
    {"name": "create_note", "desc": "Create a short text note with a title",
     "params": {"title": ("string", True), "content": ("string", True)}},
    {"name": "list_files", "desc": "List files in a directory path",
     "params": {"path": ("string", True)}},
    {"name": "read_file", "desc": "Read the contents of a file by path",
     "params": {"path": ("string", True)}},
    {"name": "write_file", "desc": "Write text content to a file path",
     "params": {"path": ("string", True), "content": ("string", True), "append": ("boolean", False)}},
    {"name": "http_get", "desc": "Perform an HTTP GET request to a URL",
     "params": {"url": ("string", True)}},
    {"name": "http_post", "desc": "Perform an HTTP POST with a JSON body",
     "params": {"url": ("string", True), "json_body": ("string", True)}},
    {"name": "translate_text", "desc": "Translate text into a target language code",
     "params": {"text": ("string", True), "target_lang": ("string", True)}},
    {"name": "convert_units", "desc": "Convert a numeric value between unit names",
     "params": {"value": ("number", True), "from_unit": ("string", True), "to_unit": ("string", True)}},
    {"name": "geo_distance", "desc": "Compute distance between two lat/lon points",
     "params": {"lat1": ("number", True), "lon1": ("number", True), "lat2": ("number", True), "lon2": ("number", True)}},
    {"name": "schedule_meeting", "desc": "Schedule a meeting with attendees",
     "params": {"title": ("string", True), "start_iso": ("string", True), "duration_min": ("integer", True),
                "attendees": ("string", True), "room": ("string", False)}},
    {"name": "database_query", "desc": "Run a read-only SQL SELECT query",
     "params": {"sql": ("string", True), "limit": ("integer", False)}},
    {"name": "toggle_light", "desc": "Turn a smart light on or off by room name",
     "params": {"room": ("string", True), "on": ("boolean", True)}},
    {"name": "ping_host", "desc": "Ping a network hostname once and report reachability",
     "params": {}},
]

SAMPLE_ARGS = {
    "city": "Tokyo", "units": "celsius", "timezone": "Europe/Paris",
    "a": 7, "b": 35, "x": 2.5, "y": 4.0,
    "hour": 7, "minute": 30, "label": "wake",
    "query": "machin anvil llm", "limit": 5,
    "to": "a@b.co", "subject": "Hi", "body": "Hello there",
    "title": "Ideas", "content": "Ship Q2", "path": "/tmp/x.txt",
    "append": True, "url": "https://example.com", "json_body": "{\"ok\":true}",
    "text": "bonjour", "target_lang": "en",
    "value": 100.0, "from_unit": "km", "to_unit": "mi",
    "lat1": 48.85, "lon1": 2.35, "lat2": 35.68, "lon2": 139.69,
    "start_iso": "2026-07-20T10:00:00Z", "duration_min": 30,
    "attendees": "alice,bob", "room": "A1",
    "sql": "SELECT 1", "on": False,
}


def openai_tools():
    out = []
    for t in TOOLS:
        props, required = {}, []
        for pname, (ptype, req) in t["params"].items():
            props[pname] = {"type": ptype, "description": f"{pname} ({ptype})"}
            if req:
                required.append(pname)
        schema = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["desc"],
                "parameters": schema,
            },
        })
    return out


def cases():
    """Build evaluation cases: selection + arity 0..5 with mixed types."""
    by = {t["name"]: t for t in TOOLS}
    cs = []
    # zero-arg tool selection
    cs.append({
        "id": "pick_ping_0arg",
        "user": "Check whether host example.com is reachable using the available tools. Call the right tool.",
        "expect_name": "ping_host",
        "expect_keys": [],
        "expect_types": {},
    })
    # 1-arg string
    cs.append({
        "id": "pick_time_1str",
        "user": "What time is it in timezone Europe/Paris? Use a tool.",
        "expect_name": "get_time",
        "expect_keys": ["timezone"],
        "expect_types": {"timezone": str},
        "expect_vals": {"timezone": "Europe/Paris"},
    })
    # 1-arg path
    cs.append({
        "id": "pick_list_1str",
        "user": "List files under /var/log using a tool.",
        "expect_name": "list_files",
        "expect_keys": ["path"],
        "expect_types": {"path": str},
        "expect_vals": {"path": "/var/log"},
    })
    # 2-arg ints
    cs.append({
        "id": "pick_add_2int",
        "user": "Use a tool to add the integers 7 and 35.",
        "expect_name": "add_numbers",
        "expect_keys": ["a", "b"],
        "expect_types": {"a": int, "b": int},
        "expect_vals": {"a": 7, "b": 35},
    })
    # 2-arg floats
    cs.append({
        "id": "pick_mul_2num",
        "user": "Multiply 2.5 by 4.0 with a tool.",
        "expect_name": "multiply",
        "expect_keys": ["x", "y"],
        "expect_types": {"x": (int, float), "y": (int, float)},
        "expect_vals": {"x": 2.5, "y": 4.0},
    })
    # 2-arg string + bool
    cs.append({
        "id": "pick_light_2mix",
        "user": "Turn off the smart light in the kitchen using a tool.",
        "expect_name": "toggle_light",
        "expect_keys": ["room", "on"],
        "expect_types": {"room": str, "on": bool},
        "expect_vals": {"room": "kitchen", "on": False},
    })
    # 3-arg mixed
    cs.append({
        "id": "pick_convert_3mix",
        "user": "Convert 100 km to mi using the unit conversion tool.",
        "expect_name": "convert_units",
        "expect_keys": ["value", "from_unit", "to_unit"],
        "expect_types": {"value": (int, float), "from_unit": str, "to_unit": str},
        "expect_vals": {"value": 100, "from_unit": "km", "to_unit": "mi"},
    })
    # 3-arg email
    cs.append({
        "id": "pick_email_3str",
        "user": "Send email to a@b.co with subject Hi and body Hello there. Use a tool.",
        "expect_name": "send_email",
        "expect_keys": ["to", "subject", "body"],
        "expect_types": {"to": str, "subject": str, "body": str},
    })
    # 4-arg geo
    cs.append({
        "id": "pick_geo_4num",
        "user": "Compute distance from lat 48.85 lon 2.35 to lat 35.68 lon 139.69 with a tool.",
        "expect_name": "geo_distance",
        "expect_keys": ["lat1", "lon1", "lat2", "lon2"],
        "expect_types": {k: (int, float) for k in ["lat1", "lon1", "lat2", "lon2"]},
    })
    # 5-arg meeting
    cs.append({
        "id": "pick_meet_5mix",
        "user": ("Schedule a meeting titled Sprint with start 2026-07-20T10:00:00Z, "
                 "duration 30 minutes, attendees alice,bob, room A1. Use a tool."),
        "expect_name": "schedule_meeting",
        "expect_keys": ["title", "start_iso", "duration_min", "attendees", "room"],
        "expect_types": {
            "title": str, "start_iso": str, "duration_min": int,
            "attendees": str, "room": str,
        },
    })
    # distractor: weather among 20
    cs.append({
        "id": "pick_weather_1str",
        "user": "What is the weather in Tokyo right now? Call one tool.",
        "expect_name": "get_weather",
        "expect_keys": ["city"],
        "expect_types": {"city": str},
        "expect_vals": {"city": "Tokyo"},
    })
    # distractor: translate
    cs.append({
        "id": "pick_translate_2str",
        "user": "Translate 'bonjour' to language code en using a tool.",
        "expect_name": "translate_text",
        "expect_keys": ["text", "target_lang"],
        "expect_types": {"text": str, "target_lang": str},
    })
    # ensure catalog size used
    assert len(by) == 20
    for c in cs:
        assert c["expect_name"] in by
    return cs


def chat(base: str, user: str, max_tokens: int = 256) -> dict:
    body = {
        "model": "bonsai17-q2",
        "messages": [
            {"role": "system", "content": "You are a tool-using assistant. Always call exactly one function when tools are available. Do not invent tool names."},
            {"role": "user", "content": user},
        ],
        "tools": openai_tools(),
        "tool_choice": "required",
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    return data, time.time() - t0


def extract_call(data: dict):
    try:
        msg = data["choices"][0]["message"]
    except Exception:
        return None, None, data
    tcs = msg.get("tool_calls") or []
    if not tcs:
        # fallback: parse content
        content = msg.get("content") or ""
        return None, None, {"raw_content": content, "full": data}
    fn = tcs[0].get("function") or {}
    name = fn.get("name")
    args_s = fn.get("arguments") or "{}"
    try:
        args = json.loads(args_s) if isinstance(args_s, str) else args_s
    except Exception:
        args = {"_parse_error": args_s}
    return name, args, None


def type_ok(val, spec):
    if isinstance(spec, tuple):
        return isinstance(val, spec)
    return isinstance(val, spec)


def evaluate(case, name, args):
    reasons = []
    ok_name = name == case["expect_name"]
    if not ok_name:
        reasons.append(f"name want={case['expect_name']} got={name}")
    ok_keys = True
    ok_types = True
    ok_vals = True
    if args is None:
        args = {}
    for k in case.get("expect_keys", []):
        if k not in args:
            ok_keys = False
            reasons.append(f"missing arg {k}")
    for k, spec in case.get("expect_types", {}).items():
        if k in args and not type_ok(args[k], spec):
            # allow stringified numbers from models
            if spec in (int, (int, float)) or spec == (int, float):
                try:
                    float(args[k])
                    continue
                except Exception:
                    pass
            if spec is bool and str(args[k]).lower() in ("true", "false", "0", "1"):
                continue
            ok_types = False
            reasons.append(f"type {k}: {type(args[k]).__name__}")
    for k, want in case.get("expect_vals", {}).items():
        if k not in args:
            continue
        got = args[k]
        if isinstance(want, bool):
            g = got if isinstance(got, bool) else str(got).lower() in ("true", "1", "yes", "off" if want is False else "on")
            # special-case on:false → accept false/0/"false"/"off"
            if want is False:
                g = got is False or got == 0 or str(got).lower() in ("false", "0", "off", "no")
            else:
                g = got is True or got == 1 or str(got).lower() in ("true", "1", "on", "yes")
            if not g:
                ok_vals = False
                reasons.append(f"val {k} want={want} got={got}")
        elif isinstance(want, (int, float)):
            try:
                if abs(float(got) - float(want)) > 1e-6:
                    ok_vals = False
                    reasons.append(f"val {k} want={want} got={got}")
            except Exception:
                ok_vals = False
                reasons.append(f"val {k} want={want} got={got}")
        else:
            if str(got).lower() != str(want).lower() and str(want).lower() not in str(got).lower():
                ok_vals = False
                reasons.append(f"val {k} want={want} got={got}")
    passed = ok_name and ok_keys and ok_types and ok_vals
    return passed, reasons, {"name_ok": ok_name, "keys_ok": ok_keys, "types_ok": ok_types, "vals_ok": ok_vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8096/v1")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    results = []
    for case in cases():
        print(f"→ {case['id']} ...", flush=True)
        try:
            data, dt = chat(args.base, case["user"])
            name, call_args, err = extract_call(data)
            if err and name is None:
                passed, reasons = False, ["no tool_calls in response"]
                flags = {}
                call_args = None
            else:
                passed, reasons, flags = evaluate(case, name, call_args or {})
            results.append({
                "id": case["id"], "ok": passed, "seconds": round(dt, 2),
                "got_name": name, "got_args": call_args, "reasons": reasons, "flags": flags,
                "expect": case["expect_name"],
            })
            print(f"  {'PASS' if passed else 'FAIL'} {dt:.1f}s name={name} args={call_args} {reasons}", flush=True)
        except Exception as e:
            results.append({
                "id": case["id"], "ok": False, "seconds": None,
                "got_name": None, "got_args": None, "reasons": [str(e)], "flags": {},
                "expect": case["expect_name"],
            })
            print(f"  ERROR {e}", flush=True)

    n = len(results)
    n_ok = sum(1 for r in results if r["ok"])
    n_name = sum(1 for r in results if r.get("flags", {}).get("name_ok"))
    lines = []
    lines.append("# M16 — Ternary-Bonsai-1.7B tool-calling bench\n")
    lines.append(f"Server: `{args.base}` · catalog **{len(TOOLS)} tools** · "
                 f"**{n_ok}/{n}** cases fully passed · **{n_name}/{n}** correct tool name.\n")
    lines.append("| case | expect | got | name | keys/types/vals | time | result |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        fl = r.get("flags") or {}
        ktv = "/".join([
            "Y" if fl.get("name_ok") else "n",
            "Y" if fl.get("keys_ok") else "n",
            "Y" if fl.get("types_ok") else "n",
            "Y" if fl.get("vals_ok") else "n",
        ]) if fl else "-"
        lines.append(
            f"| `{r['id']}` | `{r['expect']}` | `{r['got_name']}` | "
            f"{'Y' if fl.get('name_ok') else 'n'} | {ktv} | {r['seconds']}s | "
            f"{'PASS' if r['ok'] else 'FAIL'} |"
        )
    lines.append("\n## Failures / notes\n")
    for r in results:
        if not r["ok"]:
            lines.append(f"- **{r['id']}**: {', '.join(r['reasons']) or 'unknown'}; args=`{r['got_args']}`")
    lines.append("\n## Method\n")
    lines.append("- OpenAI `/v1/chat/completions` with `tool_choice:\"required\"` and all 20 tools.")
    lines.append("- `COLIBRI_NOTHINK=1` recommended so thinking traces do not drown the call.")
    lines.append("- Args checked for required keys, JSON types (with stringified number/bool tolerance), and soft value match when specified.")
    lines.append(f"- Harness: `tools_bonsai_tools_bench.py`.\n")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}: {n_ok}/{n} pass, {n_name}/{n} name-ok")


if __name__ == "__main__":
    main()
