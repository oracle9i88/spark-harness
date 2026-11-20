"""
spk web: the same harness in a browser. One local server process holds the sessions; a turn runs on the server, so
closing or reloading the page loses nothing: the page reconnects and replays what it missed.

  spk web [--host 0.0.0.0] [--port 8790]

Access needs the token in ~/.spk/web-token (made on first start): open the printed link once per device, it sets a
cookie. Anyone with the link can read files on this machine through the tools, so keep it to yourself.
"""
import http.server, json, os, re, secrets, socket, subprocess, sys, threading, time, urllib.parse

import spk

WEB = os.path.join(spk.HERE, "web")
UPLOADS = os.path.join(spk.HOME, "uploads")
TOKEN_FILE = os.path.join(spk.HOME, "web-token")
MAX_UPLOAD = 200 << 20


def token():
    try:
        t = open(TOKEN_FILE).read().strip()
        if t:
            return t
    except FileNotFoundError:
        pass
    os.makedirs(spk.HOME, exist_ok=True)
    t = secrets.token_urlsafe(24)
    with open(TOKEN_FILE, "w") as f:
        f.write(t)
    os.chmod(TOKEN_FILE, 0o600)
    return t


class WebUI:
    """spk's UI interface, as a list of events the page streams (and replays after a reconnect)"""
    can_confirm = True

    def __init__(self):
        self.events, self.cond, self.answers = [], threading.Condition(), {}

    def push(self, type_, data=None):
        with self.cond:
            self.events.append({"seq": len(self.events), "type": type_, "data": data, "t": time.time()})
            if len(self.events) > 20000:                       # keep seq numbers, drop old bodies
                for e in self.events[:-20000]:
                    e["data"] = None
            self.cond.notify_all()

    def info(self, m): self.push("info", m)
    def warn(self, m): self.push("warn", m)
    def error(self, m): self.push("error", m)
    def tool(self, m): self.push("tool", m)
    def waiting(self, m): self.push("waiting", m)
    def think(self, t): self.push("think", t)
    def text(self, t): self.push("text", t)
    def end_reply(self, stats): self.push("end_reply", stats)
    def status(self, st): self.push("status", st)
    def user(self, text): self.push("user", text)

    def confirm(self, q):
        cid = secrets.token_hex(6)
        ev = threading.Event()
        self.answers[cid] = [ev, False]
        self.push("confirm", {"cid": cid, "q": q})
        ok = ev.wait(600) and self.answers[cid][1]
        self.answers.pop(cid, None)
        return ok


class WebSession:
    def __init__(self, c, s):
        self.s, self.ui, self.busy, self.lock = s, WebUI(), False, threading.Lock()
        s.ui = self.ui
        s.cancel = threading.Event()
        self.srv = spk.Server(c, s.profile or c["profile"])
        s.profile = self.srv.name
        self.srv.ui, self.srv.cancel = self.ui, s.cancel
        self.tools = spk.Tools(c, s.web, self.ui)

    @property
    def id(self):
        return os.path.basename(self.s.path)[:-5]

    def set_profile(self, name):
        self.srv = spk.Server(self.s.c, name)
        self.srv.ui, self.srv.cancel = self.ui, self.s.cancel
        self.s.profile = name

    def status(self):
        try:
            return spk.status(self.s, self.srv, self.tools)
        except Exception as e:
            self.ui.error(f"status: {e}")

    def run_turn(self, text):
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        self.s.cancel.clear()

        def go():
            self.ui.push("busy", True)
            try:
                spk.ask(self.s, self.srv, self.tools, text)
            except Exception as e:
                self.ui.error(f"internal error: {e}")
            finally:
                self.busy = False
                self.ui.push("busy", False)
        threading.Thread(target=go, daemon=True).start()
        return True


def display_items(s):
    """the saved messages as the page shows them: pinned documents as chips, continuation prompts folded away,
    the parts of a continued reply joined"""
    items = []
    for m in s.messages[s.pinned:]:
        r, c = m["role"], m.get("content") or ""
        if r == "user":
            if c == spk.CONTINUE_NOTE or c.startswith("（你刚才的输出开始重复"):
                continue
            items.append({"type": "user", "text": re.sub(r"^\(today is [0-9-]+\)\n", "", c)})
        elif r == "assistant":
            if c and c != "（思考过长被截断）":
                if items and items[-1]["type"] == "assistant" and not items[-1].get("closed"):
                    items[-1]["text"] += spk.drop_overlap(items[-1]["text"][-80:], c)
                else:
                    items.append({"type": "assistant", "text": c})
            for tc in m.get("tool_calls") or []:
                if items and items[-1]["type"] == "assistant":
                    items[-1]["closed"] = True
                f = tc.get("function", {})
                items.append({"type": "tool", "text": f"{f.get('name')}({(f.get('arguments') or '')[:160]})"})
        elif r == "tool":
            items.append({"type": "tool_result", "text": (c or "")[:600]})
    return items


def list_sessions():
    out = []
    if not os.path.isdir(spk.SESS):
        return out
    for f in sorted(os.listdir(spk.SESS), reverse=True)[:200]:
        if not f.endswith(".json"):
            continue
        p = os.path.join(spk.SESS, f)
        title = ""
        try:
            j = json.load(open(p))
            for m in j["messages"][j["pinned"]:]:
                if m["role"] == "user" and m.get("content") and not m["content"].startswith("Summary of our conversation"):
                    title = re.sub(r"^\(today is [0-9-]+\)\n", "", m["content"])[:60]; break
            docs = [os.path.basename(d) for d in j.get("docs", [])]
            if not title and docs:
                title = "📄 " + ", ".join(docs)[:60]
        except Exception:
            continue
        out.append({"id": f[:-5], "title": title or "(empty)", "mtime": os.path.getmtime(p)})
    return out


class App:
    def __init__(self, c):
        self.c, self.sessions, self.tok = c, {}, token()
        self.ops_cache = {}

    def get(self, sid):
        if sid not in self.sessions:
            path = os.path.join(spk.SESS, sid + ".json")
            if not re.fullmatch(r"[0-9-]+", sid or "") or not os.path.exists(path):
                return None
            self.sessions[sid] = WebSession(self.c, spk.Session.load(self.c, path))
        return self.sessions[sid]

    def new(self, profile=None):
        s = spk.Session(self.c)
        while os.path.exists(s.path):
            time.sleep(1); s = spk.Session(self.c)
        s.profile = profile if profile in self.c["profiles"] else self.c["profile"]
        s.save()
        ws = WebSession(self.c, s)
        self.sessions[ws.id] = ws
        return ws


def make_handler(app):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body, ctype="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False)
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _authed(self):
            ck = self.headers.get("Cookie") or ""
            m = re.search(r"(?:^|;\s*)spk_t=([^;]+)", ck)
            return bool(m) and secrets.compare_digest(m.group(1), app.tok)

        def _qs(self):
            return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                raise ValueError("too large")
            return self.rfile.read(n) if n else b""

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            q = self._qs()
            if path == "/" and q.get("t"):
                if secrets.compare_digest(q["t"], app.tok):
                    return self._send(302, "", "text/plain", {"Location": "/", "Set-Cookie":
                                      f"spk_t={app.tok}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000"})
                return self._send(403, "wrong token", "text/plain")
            if not self._authed():
                return self._send(403, "<p>spk: open the link printed by <code>spk web</code> (it carries the access token).</p>", "text/html")
            if path == "/":
                return self._send(200, open(os.path.join(WEB, "index.html"), "rb").read(), "text/html")
            if path == "/api/sessions":
                return self._send(200, list_sessions())
            if path == "/api/profiles":
                return self._send(200, [{"name": k, "model": v["model"], "note": v.get("note", ""), "context": v["context"]}
                                        for k, v in app.c["profiles"].items()])
            if path == "/api/session":
                ws = app.get(q.get("id"))
                if not ws:
                    return self._send(404, {"error": "no such session"})
                return self._send(200, {"id": ws.id, "items": display_items(ws.s), "busy": ws.busy,
                                        "seq": len(ws.ui.events), "docs": [os.path.basename(d) for d in ws.s.docs],
                                        "profile": ws.s.profile, "think": ws.s.think, "web": ws.s.web})
            if path == "/api/events":
                return self.events(q)
            if path == "/api/ops":
                what = q.get("what")
                fn = {"status": spk.spk_ops.status, "updates": spk.spk_ops.updates, "eco": spk.spk_ops.eco}.get(what)
                if not fn:
                    return self._send(400, {"error": "what?"})
                try:
                    return self._send(200, {"text": fn()})
                except Exception as e:
                    return self._send(500, {"error": str(e)})
            return self._send(404, {"error": "not found"})

        def events(self, q):
            ws = app.get(q.get("id"))
            if not ws:
                return self._send(404, {"error": "no such session"})
            since = int(q.get("since") or 0)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True
            ui = ws.ui
            try:
                while True:
                    with ui.cond:
                        if len(ui.events) <= since:
                            ui.cond.wait(15)
                        batch = ui.events[since:]
                    if batch:
                        for e in batch:
                            self.wfile.write(f"id: {e['seq']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n".encode())
                        since = batch[-1]["seq"] + 1
                    else:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def do_POST(self):
            if not self._authed():
                return self._send(403, {"error": "no access"})
            if (self.headers.get("Origin") or "").split("://")[-1] not in ("", self.headers.get("Host")):
                return self._send(403, {"error": "cross-site request"})
            path = urllib.parse.urlsplit(self.path).path
            q = self._qs()
            try:
                if path == "/api/upload":
                    return self.upload(q)
                body = json.loads(self._body() or b"{}")
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            if path == "/api/new":
                ws = app.new(body.get("profile"))
                threading.Thread(target=ws.status, daemon=True).start()       # counting can take a moment
                return self._send(200, {"id": ws.id})
            ws = app.get(body.get("id"))
            if not ws:
                return self._send(404, {"error": "no such session"})
            if path == "/api/send":
                text = (body.get("text") or "").strip()
                if not text:
                    return self._send(400, {"error": "empty"})
                if not ws.run_turn(text):
                    return self._send(409, {"error": "a reply is still running"})
                return self._send(200, {"ok": True})
            if path == "/api/stop":
                ws.s.cancel.set()
                return self._send(200, {"ok": True})
            if path == "/api/confirm":
                a = ws.ui.answers.get(body.get("cid"))
                if a:
                    a[1] = bool(body.get("ok")); a[0].set()
                return self._send(200, {"ok": True})
            if ws.busy and path in ("/api/settings", "/api/pin", "/api/compact"):
                return self._send(409, {"error": "wait for the reply to finish (or stop it)"})
            if path == "/api/settings":
                if body.get("think") in spk.THINK_LEVELS:
                    ws.s.think = body["think"]
                if "web" in body:
                    ws.s.web = ws.tools.web = bool(body["web"])
                if body.get("profile") in app.c["profiles"]:
                    ws.set_profile(body["profile"])
                ws.s.save()
                return self._send(200, ws.status() or {})
            if path == "/api/pin":
                try:
                    ws.s.pin([p for p in body.get("paths", []) if p])
                except OSError as e:
                    return self._send(400, {"error": str(e)})
                ws.s.save()
                return self._send(200, ws.status() or {})
            if path == "/api/compact":
                def go():
                    ws.busy = True; ws.ui.push("busy", True)
                    try:
                        spk.compact(ws.s, ws.srv, ws.tools, "asked"); ws.status()
                    except Exception as e:
                        ws.ui.error(str(e))
                    finally:
                        ws.busy = False; ws.ui.push("busy", False)
                threading.Thread(target=go, daemon=True).start()
                return self._send(200, {"ok": True})
            if path == "/api/status":
                return self._send(200, ws.status() or {})
            return self._send(404, {"error": "not found"})

        def upload(self, q):
            ws = app.get(q.get("id"))
            if not ws:
                return self._send(404, {"error": "no such session"})
            if ws.busy:
                return self._send(409, {"error": "wait for the reply to finish (or stop it)"})
            name = os.path.basename(q.get("name") or "")
            if not name or name.startswith("."):
                return self._send(400, {"error": "bad file name"})
            d = os.path.join(UPLOADS, ws.id)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, name)
            with open(p, "wb") as f:
                f.write(self._body())
            first_doc = not ws.s.docs
            try:
                ws.s.pin([p])
            except Exception as e:
                return self._send(400, {"error": f"could not read {name}: {e}"})
            if first_doc and ws.s.web:                     # as in `spk doc`: documents may be private, no web by default
                ws.s.web = ws.tools.web = False
                ws.ui.warn("web tools are now off for this session (documents pinned); turn them back on if you need them")
            ws.s.save()
            return self._send(200, ws.status() or {})

    return H


def addresses(port, tok):
    hosts = ["127.0.0.1"]
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=5).stdout
        hosts += [ip for ip in re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", out) if not ip.startswith(("127.", "198.18.", "169.254."))]
    except Exception:
        pass
    return [f"http://{h}:{port}/?t={tok}" for h in hosts]


def main(c, host="0.0.0.0", port=8790):
    app = App(c)
    srv = http.server.ThreadingHTTPServer((host, port), make_handler(app))
    srv.daemon_threads = True
    print(f"spk web on {host}:{port}. Open one of these once per device (the link carries the access token):")
    for u in addresses(port, app.tok) if host in ("0.0.0.0", "") else [f"http://{host}:{port}/?t={app.tok}"]:
        print("  " + u)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
