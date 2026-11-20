#!/bin/bash
# Failure-mode tests against tests/mock_server.py: a server that goes silent, one that loops, replies cut by the length
# limit, a server that is down. Each must end on its own with the right message; none may hang.
cd "$(dirname "$0")"; T=$(mktemp -d); PORT=18977
cat > $T/config.json <<C
{"profile": "ok", "stall_seconds": 20, "server_down_wait": 25, "nodes": [],
 "profiles": {"ok": {"base_url": "http://127.0.0.1:$PORT", "model": "ok", "context": 8000, "tokenize": true},
              "hang": {"base_url": "http://127.0.0.1:$PORT", "model": "hang", "context": 8000},
              "repeat": {"base_url": "http://127.0.0.1:$PORT", "model": "repeat", "context": 8000},
              "length": {"base_url": "http://127.0.0.1:$PORT", "model": "length", "context": 8000},
              "down": {"base_url": "http://127.0.0.1:9", "model": "x", "context": 8000}}}
C
export SPK_CONFIG=$T/config.json SPK_SESSIONS=$T/sessions
python3 mock_server.py $PORT & MOCK=$!; sleep 1
run() { local name=$1 want=$2; shift 2; local t0=$(date +%s)
  out=$(timeout 200 python3 ../spk.py run -m "$name" --think off "test" 2>&1); rc=$?; local dt=$(( $(date +%s) - t0 ))
  if echo "$out" | grep -q "$want" && [ $rc -ne 124 ]; then echo "PASS $name (${dt}s): $want"; else echo "FAIL $name (rc $rc, ${dt}s)"; echo "$out" | tail -8; fi; }
run ok "ok"
run length "continuing (3/6)"
run repeat "stopped: the reply was repeating itself"
run hang "lost the request twice"
run down "did not come back"
kill $MOCK; rm -rf $T
