# M16 — Ternary-Bonsai-1.7B tool-calling bench

Server: `http://127.0.0.1:8096/v1` · catalog **20 tools** · **12/12** cases fully passed · **12/12** correct tool name.

| case | expect | got | name | keys/types/vals | time | result |
|---|---|---|---|---|---|---|
| `pick_ping_0arg` | `ping_host` | `ping_host` | Y | Y/Y/Y/Y | 469.22s | PASS |
| `pick_time_1str` | `get_time` | `get_time` | Y | Y/Y/Y/Y | 21.65s | PASS |
| `pick_list_1str` | `list_files` | `list_files` | Y | Y/Y/Y/Y | 19.14s | PASS |
| `pick_add_2int` | `add_numbers` | `add_numbers` | Y | Y/Y/Y/Y | 24.1s | PASS |
| `pick_mul_2num` | `multiply` | `multiply` | Y | Y/Y/Y/Y | 25.98s | PASS |
| `pick_light_2mix` | `toggle_light` | `toggle_light` | Y | Y/Y/Y/Y | 28.72s | PASS |
| `pick_convert_3mix` | `convert_units` | `convert_units` | Y | Y/Y/Y/Y | 37.08s | PASS |
| `pick_email_3str` | `send_email` | `send_email` | Y | Y/Y/Y/Y | 40.28s | PASS |
| `pick_geo_4num` | `geo_distance` | `geo_distance` | Y | Y/Y/Y/Y | 64.04s | PASS |
| `pick_meet_5mix` | `schedule_meeting` | `schedule_meeting` | Y | Y/Y/Y/Y | 80.59s | PASS |
| `pick_weather_1str` | `get_weather` | `get_weather` | Y | Y/Y/Y/Y | 29.19s | PASS |
| `pick_translate_2str` | `translate_text` | `translate_text` | Y | Y/Y/Y/Y | 30.76s | PASS |

## Failures / notes

None. Extra/optional args (e.g. `units` on weather, `host` on zero-arg ping) were tolerated when required keys/types/values matched.

Cold first case ~469s (1329-token tools prefill on this laptop); warm cases ~19–81s via prefix cache.

## Method

- OpenAI `/v1/chat/completions` with `tool_choice:"required"` and all 20 tools.
- `COLIBRI_NOTHINK=1` recommended so thinking traces do not drown the call.
- Args checked for required keys, JSON types (with stringified number/bool tolerance), and soft value match when specified.
- Harness: `tools_bonsai_tools_bench.py`.

