#!/bin/bash
set -u
cd /root/colibri
systemctl stop colibri
sleep 2

run_one() {
  local dir="$1" label="$2"
  COLIBRI_THREADS=12 COLIBRI_SKILLS="$dir" nohup ./serve_l3 models/llama32-1b-q80.bin 8090 </dev/null >cond.log 2>&1 &
  local pid=$!
  for i in $(seq 1 40); do curl -s -m2 http://127.0.0.1:8090/health 2>/dev/null | grep -q ok && break; sleep 1; done
  python3 eval_faithful.py 14 "$label"
  kill -9 $pid 2>/dev/null
  sleep 2
}

echo '=== skills reliability eval (llama-3.2-1b, rbm21 idle, temp 0.7, N=14) ==='
run_one /root/colibri/emptyskills "WITHOUT-skills"
run_one /root/colibri/skills     "WITH-skills   "

systemctl start colibri
echo '(systemd colibri restarted)'
