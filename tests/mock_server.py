"""A fake OpenAI-compatible server that misbehaves on purpose, for tests/run_tests.sh.
The model name picks the behaviour: hang (accepts, never answers; /health says idle), repeat (loops one sentence),
length (cuts every reply at a few words with finish_reason length, three times, then finishes), ok."""
import http.server, json, sys, threading, time

calls = {"length": 0, "repeat": 0}


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._json({"status": "ok", "active": 0, "queued": 0})

    def _sse(self, chunks, finish):
        try:
            self._sse_inner(chunks, finish)
        except (BrokenPipeError, ConnectionResetError):
            pass                       # the client stopped reading (it cut a looping reply): expected

    def _sse_inner(self, chunks, finish):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for c in chunks:
            self.wfile.write(f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': c}, 'finish_reason': None}]})}\n\n".encode())
            self.wfile.flush(); time.sleep(0.01)
        self.wfile.write(f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        self.close_connection = True

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self._json({"count": sum(len(json.dumps(m)) // 4 for m in body.get("messages", [])), "tokens": []})
        mode = body.get("model")
        last = body["messages"][-1]["content"] or ""
        if "Summarize" in last:
            return self._sse(["summary: the user asked things."], "stop")
        if mode == "hang":
            time.sleep(3600); return
        if mode == "repeat":
            calls["repeat"] += 1
            if calls["repeat"] == 1:
                return self._sse(["这一句话会一直重复下去。"] * 400, "stop")
            return self._sse(["好的，结论是：没问题。"], "stop")
        if mode == "length":
            calls["length"] += 1
            if calls["length"] <= 3:
                return self._sse([f"第{calls['length']}段，", "写到一半"], "length")
            return self._sse(["写到一半的话写完了。"], "stop")
        return self._sse(["ok"], "stop")


srv = http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H)
srv.daemon_threads = True
srv.serve_forever()
