#!/usr/bin/env python3
"""
spk: an agent harness for models served on NVIDIA DGX Sparks (or any OpenAI-compatible server that is slow to read
long prompts and has a prefix cache).

What it is built around:
- Reading a prompt is the expensive part (a GB10 pair reads ~500-650 tokens/s cold), and a prefix cache makes a
  repeated prefix nearly free. So the fixed preamble is short (~1.1K tokens) and never changes, the history is
  append-only, and documents are pinned right after the preamble: read once, reused by every later question.
- It does not hang and does not stop for "continue": every request is watched (server queue, reading, down, lost),
  replies cut off by the length limit are continued automatically, the context is compacted before it fills up
  (also in the middle of a tool loop), and a reply that starts repeating itself is stopped.
- One request at a time, and never a second copy of a request the server may still be working on.
- Spark operations (read only): temperatures, memory, services, pending updates, upstream project news.

  spk                                   chat
  spk doc FILE... [-q "question"]       pin documents, then ask (web tools off unless --web)
  spk run "task"                        one task, prints the answer, exits
  spk -c [run "task"]                   continue the last session
  spk spark | updates | eco             the Sparks' state, their updates, upstream news (no model call)
  spk web [--port 8790]                 the same in a browser (phone too); prints a link with the access token

Configuration: ~/.spk/config.json (servers, nodes); see config.example.json.
"""
import argparse, datetime, json, os, queue, re, shutil, subprocess, sys, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import spk_ops  # noqa: E402

HOME = os.path.expanduser("~/.spk")
SESS = os.environ.get("SPK_SESSIONS") or os.path.join(HOME, "sessions")
CONF = os.environ.get("SPK_CONFIG") or os.path.join(HOME, "config.json")
THINK_LEVELS = ["off", "low", "medium", "high"]
# How each kind of server takes a thinking level. "budget": our Inkling engine's engine-enforced token budget;
# "effort": reasoning_effort as most OpenAI-compatible servers take it.
THINK_STYLES = {
    "budget": {"off": {"thinking_budget": 0, "reasoning_effort": 0}, "low": {"thinking_budget": 1024},
               "medium": {"thinking_budget": 4096}, "high": {"thinking_budget": 16000}},
    "effort": {"off": {"reasoning_effort": "none"}, "low": {"reasoning_effort": "low"},
               "medium": {"reasoning_effort": "medium"}, "high": {"reasoning_effort": "high"}},
    "effort-xhigh": {"off": {"reasoning_effort": "none"}, "low": {"reasoning_effort": "low"},
                     "medium": {"reasoning_effort": "medium"}, "high": {"reasoning_effort": "xhigh"}},
    "default-off": {"off": {}, "low": {"reasoning_effort": "low"}, "medium": {"reasoning_effort": "medium"},
                    "high": {"reasoning_effort": "high"}},
}
PROFILE_DEFAULTS = {"context": 131072, "compact_at": None, "max_tokens": 16384, "tokenize": False, "slots": 0,
                    "think_style": "effort", "api_key": "local", "note": ""}
DEFAULTS = {
    "profile": None,
    "profiles": {},
    "nodes": [],
    "site": "",                     # one or two sentences about the machines, added to the (fixed) system prompt
    "default_think": "low",
    "tool_result_tokens": 6000,     # cap per tool result (approximate)
    "tavily": "~/.hermes/bin/tavily.sh",
    "auto_continue": 6,             # how many times a reply cut off by max_tokens is continued without asking
    "stall_seconds": 600,           # no output for this long while the server reports nothing running: resend
    "server_down_wait": 1800,       # how long to wait for a server that went down before giving up
    "max_steps": 40,                # model calls per user turn (tool loops)
}

SYSTEM = """You are spk, a careful assistant running on the user's own servers.{site} Answer in the user's language \
(usually Chinese). Be direct: give the answer or the judgement first, then the reasons.

Documents the user pinned appear in <document> tags at the start of the conversation. Quote them exactly when it \
matters and say where (document name, line or section). If the documents do not say something, say so instead of \
guessing.

Tools: use them only when they help. Read files in pieces with read_file (it returns numbered lines). Search the web \
only for facts you need and cannot get from the documents; each query leaves the user's machine, so never put \
names, case details or other private facts from the documents into a query. The spark_* tools only read the \
machines' state: nothing here installs updates or restarts services, so recommend such steps to the user instead. \
Write files only when asked."""

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file (txt, md, csv, json, pdf, docx). Returns numbered lines.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer", "description": "first line, 1-based"},
                    "lines": {"type": "integer", "description": "how many lines (default 400)"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory (names, sizes).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search text in files under a path (regular expression). Returns file:line: text.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern", "path"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web. The query leaves the machine: no private facts in it.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fetch_url", "description": "Fetch a web page as clean text.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "spark_status", "description": "The Sparks now: temperatures, power, GPU load, free memory, disk, containers, API health. Read only.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "spark_updates", "description": "Driver, kernel, DGX OS and upgradable key packages on each Spark, and where nodes differ. Read only, installs nothing.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "spark_ecosystem", "description": "New commits and releases of the projects the Spark stacks come from (inference engines, NCCL, model repos) since the last check.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a text file (asks the user first).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]
WEB_TOOLS = {"web_search", "fetch_url"}
CONTINUE_NOTE = "（上一条回复被长度上限截断。请从断点处直接接着写，不要重复已经写过的内容，也不要加开场白。）"

TTY = sys.stdout.isatty()
DIM, BOLD, YEL, RED, RST = ("\033[2m", "\033[1m", "\033[33m", "\033[31m", "\033[0m") if TTY else ("",) * 5


def say(msg, color=DIM, end="\n"):
    print(f"{color}{msg}{RST}", end=end, flush=True)


class Cancelled(Exception):
    pass


class CliUI:
    """Where the core's output goes. The web front end (spk_web.py) has its own with the same methods."""
    def __init__(self):
        self.can_confirm = sys.stdin.isatty()
        self.mode, self.waits = None, False

    def _unwait(self):
        if self.waits:
            print("\r" + " " * 90 + "\r", end="", flush=True); self.waits = False

    def info(self, m): self._unwait(); self._break(); say("  " + m)
    def warn(self, m): self._unwait(); self._break(); say("  " + m, YEL)
    def error(self, m): self._unwait(); self._break(); say("  " + m, RED)
    def tool(self, m): self._unwait(); self._break(); say("  → " + m)

    def waiting(self, m):
        print(f"\r{DIM}  … {m}{RST}   ", end="", flush=True); self.waits = True

    def _break(self):
        if self.mode:
            print(RST); self.mode = None

    def think(self, t):
        self._unwait()
        if self.mode != "think":
            self._break(); print(f"{DIM}  thinking: ", end=""); self.mode = "think"
        print(f"{DIM}{t}", end="", flush=True)

    def text(self, t):
        self._unwait()
        if self.mode != "text":
            if self.mode: print(RST + "\n", end="")
            self.mode = "text"
        print(RST + t, end="", flush=True)

    def end_reply(self, stats):
        self._unwait(); self._break(); say(f"  [{stats}]")

    def status(self, st):
        say(f"  [{st['profile']}] context {'' if st['exact'] else '~'}{st['tokens']:,} / {st['context']:,} tokens "
            f"(compacts at {st['compact_at']:,}); docs {st['docs']}; web {'on' if st['web'] else 'off'}; think {st['think']}")

    def confirm(self, q):
        return input(f"{YEL}  {q} [y/N] {RST}").strip().lower() == "y"

    def user(self, text): pass


def conf():
    c = json.loads(json.dumps(DEFAULTS))
    try:
        user = json.load(open(CONF))
    except FileNotFoundError:
        sys.exit(f"no {CONF}: copy {os.path.join(HERE, 'config.example.json')} there and fill in your servers")
    c.update(user)
    for name, p in c["profiles"].items():
        for k, v in PROFILE_DEFAULTS.items():
            p.setdefault(k, v)
        if not p["compact_at"]:
            p["compact_at"] = int(p["context"] * 0.7)
    if not c["profile"]:
        c["profile"] = next(iter(c["profiles"]), None)
    spk_ops.configure(c)
    return c


def approx_tokens(s):
    """rough count (CJK ~1 token per character, other text ~4 characters per token)"""
    cjk = len(re.findall(r"[　-鿿＀-￯]", s))
    return cjk + (len(s) - cjk) // 4


def cap(s, n):
    if approx_tokens(s) <= n:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if approx_tokens(s[:mid]) <= n: lo = mid
        else: hi = mid - 1
    return s[:lo] + f"\n[... cut: {approx_tokens(s) - n} more tokens; read further with start/lines or a narrower search]"


# ---------- documents ----------

def load_text(path):
    p = os.path.expanduser(path)
    ext = os.path.splitext(p)[1].lower()
    if ext == ".pdf":
        r = subprocess.run(["pdftotext", "-layout", p, "-"], capture_output=True, text=True, timeout=300)
        if r.returncode: raise OSError(r.stderr.strip() or "pdftotext failed")
        return r.stdout
    if ext in (".docx", ".doc", ".rtf", ".odt", ".pages"):
        r = subprocess.run(["textutil", "-convert", "txt", "-stdout", p], capture_output=True, text=True, timeout=300)
        if r.returncode: raise OSError(r.stderr.strip() or "textutil failed")
        return r.stdout
    with open(p, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace")


def doc_block(path):
    lines = load_text(path).splitlines()
    body = "\n".join(f"{i + 1}\t{l}" for i, l in enumerate(lines))
    return f'<document name="{os.path.basename(path)}" lines="{len(lines)}">\n{body}\n</document>'


# ---------- tools ----------

class Tools:
    def __init__(self, c, web, ui):
        self.c, self.web, self.ui = c, web, ui

    def specs(self):
        # always the same list: the tools are part of the cached preamble, so switching web off must not change it
        return TOOLS

    def run(self, name, args):
        try:
            if name in WEB_TOOLS and not self.web:
                return "web tools are off in this session (the user can turn them on with /web on)"
            out = getattr(self, "t_" + name)(**args)
        except AttributeError:
            out = f"no tool named {name}"
        except TypeError as e:
            out = f"bad arguments: {e}"
        except subprocess.TimeoutExpired:
            out = "the tool timed out"
        except Exception as e:
            out = f"error: {e}"
        return cap(out, self.c["tool_result_tokens"])

    def t_read_file(self, path, start=1, lines=400):
        all_lines = load_text(path).splitlines()
        s = max(1, int(start)); e = min(len(all_lines), s - 1 + max(1, int(lines)))
        return f"{path}: lines {s}-{e} of {len(all_lines)}\n" + "\n".join(f"{i}\t{all_lines[i - 1]}" for i in range(s, e + 1))

    def t_list_dir(self, path):
        p = os.path.expanduser(path)
        rows = []
        for n in sorted(os.listdir(p))[:500]:
            q = os.path.join(p, n)
            rows.append(f"{n}/" if os.path.isdir(q) else f"{n}\t{os.path.getsize(q)} B")
        return "\n".join(rows) or "(empty)"

    def t_grep(self, pattern, path):
        exe = shutil.which("rg")
        cmd = [exe, "-n", "--no-heading", "-m", "50", "-e", pattern, os.path.expanduser(path)] if exe else \
              ["grep", "-rn", "-m", "50", "-E", pattern, os.path.expanduser(path)]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout or "(no matches)"

    def _tavily(self, flag, arg):
        exe = os.path.expanduser(self.c["tavily"])
        if not os.path.exists(exe):
            return "{}"
        arg = arg.replace('"', " ").replace("\\", " ")          # the helper puts the argument into JSON as is
        self.ui.warn(f"↗ {'search' if not flag else 'fetch'} (leaves this machine): {arg}")
        return subprocess.run([exe] + ([flag] if flag else []) + [arg], capture_output=True, text=True, timeout=120).stdout

    def t_web_search(self, query):
        res = (json.loads(self._tavily("", query) or "{}").get("results")) or []
        return "\n\n".join(f"[{i + 1}] {x.get('title', '')}\n{x.get('url', '')}\n{x.get('content', '')}" for i, x in enumerate(res)) or "(no results)"

    def t_fetch_url(self, url):
        j = json.loads(self._tavily("-x", url) or "{}")
        res = j.get("results") or []
        return res[0].get("raw_content", "") if res else f"(could not fetch: {j.get('failed_results') or j})"

    def t_spark_status(self):
        return spk_ops.status()

    def t_spark_updates(self):
        return spk_ops.updates()

    def t_spark_ecosystem(self):
        return spk_ops.eco()

    def t_write_file(self, path, content):
        p = os.path.abspath(os.path.expanduser(path))
        if not self.ui.can_confirm:
            return "refused: writing files needs an interactive session"
        if not self.ui.confirm(f"write {p} ({len(content)} chars){' — OVERWRITE' if os.path.exists(p) else ''}?"):
            return "the user declined"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
        return f"wrote {p}"


# ---------- server ----------

class ServerDown(Exception): pass
class Lost(Exception): pass
class TooLong(Exception): pass


class Server:
    def __init__(self, c, profile):
        if profile not in c["profiles"]:
            sys.exit(f"no profile {profile!r} in {CONF}")
        self.c, self.name, self.p = c, profile, c["profiles"][profile]
        self.base = self.p["base_url"].rstrip("/")
        self.think = THINK_STYLES[self.p["think_style"]]
        self.last_usage = None
        self.ui, self.cancel = CliUI(), threading.Event()

    def _req(self, path, body=None):
        h = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.p["api_key"])}
        return urllib.request.Request(self.base + path, None if body is None else json.dumps(body).encode(), h)

    def health(self):
        """the server's /health as a dict, {} when it answers without JSON, None when it does not answer"""
        try:
            raw = urllib.request.urlopen(self._req("/health"), timeout=8).read()
            try:
                return json.loads(raw)
            except ValueError:
                return {}
        except Exception:
            return None

    def count(self, messages, tools):
        """(tokens, exact)"""
        if self.p["tokenize"]:
            try:
                r = urllib.request.urlopen(self._req("/tokenize", {"messages": messages, "tools": tools}), timeout=120)
                return json.load(r)["count"], True
            except Exception:
                pass
        est = approx_tokens(json.dumps(tools or [], ensure_ascii=False)) + sum(approx_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)
        return est, False

    def stream(self, messages, tools, think, max_tokens):
        """Yields ("reasoning"|"content", text), ("tool", call), ("finish", reason). The HTTP read runs on its own
        thread, so this side never blocks for long: every 15 s without data it asks the server how it is doing and
        shows it. Raises ServerDown (server gone: whatever it was doing is lost), Lost (no data for stall_seconds
        while the server reports nothing running or waiting) or TooLong (the server refused the context)."""
        body = {"model": self.p["model"], "messages": messages, "stream": True, "max_tokens": max_tokens,
                "temperature": 0.6, "stream_options": {"include_usage": True}}
        body.update(self.think[think])
        if tools:
            body["tools"] = tools
        q, hold, stop = queue.Queue(), {}, threading.Event()

        def reader():
            try:
                resp = urllib.request.urlopen(self._req("/v1/chat/completions", body), timeout=7200)
                hold["resp"] = resp
                for raw in resp:
                    if stop.is_set():
                        break
                    q.put(("line", raw))
                q.put(("eof", None))
            except urllib.error.HTTPError as e:
                q.put(("http", (e.code, e.read().decode("utf-8", "replace")[:800])))
            except Exception as e:
                q.put(("err", e))

        threading.Thread(target=reader, daemon=True).start()
        last, checked = time.time(), time.time()
        try:
            while True:
                if self.cancel.is_set():
                    raise Cancelled()
                try:
                    kind, val = q.get(timeout=1)
                except queue.Empty:
                    if time.time() - checked < 15:
                        continue
                    checked = time.time()
                    h = self.health()
                    idle = time.time() - last
                    if h is None:
                        raise ServerDown(f"{self.base} stopped answering")
                    act, que = h.get("active"), h.get("queued")
                    where = f"server: {act} running, {que} waiting" if act is not None else "server up"
                    self.ui.waiting(f"no output for {idle:.0f} s ({where})")
                    if idle > self.c["stall_seconds"] and act == 0 and not que:
                        raise Lost(f"no output for {idle:.0f} s and the server is idle")
                    continue
                last = checked = time.time()
                if kind == "eof":
                    return
                if kind == "http":
                    code, text = val
                    if code in (400, 413) and re.search(r"context|too long|max_model_len|maximum|exceed", text, re.I):
                        raise TooLong(text)
                    raise RuntimeError(f"server said {code}: {text}")
                if kind == "err":
                    if isinstance(val, (urllib.error.URLError, ConnectionError, OSError)) and self.health() is None:
                        raise ServerDown(f"{self.base}: {getattr(val, 'reason', val)}")
                    raise RuntimeError(f"connection broke: {val}")
                line = val.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                j = json.loads(data)
                if j.get("usage"):
                    self.last_usage = j["usage"]
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content"): yield "reasoning", d["reasoning_content"]
                    if d.get("content"): yield "content", d["content"]
                    for tc in d.get("tool_calls") or []:
                        yield "tool", tc
                    if ch.get("finish_reason"):
                        yield "finish", ch["finish_reason"]
        finally:
            stop.set()
            try:
                hold.get("resp") and hold["resp"].close()
            except Exception:
                pass

    def wait_up(self):
        limit = time.time() + self.c["server_down_wait"]
        while time.time() < limit:
            if self.health() is not None:
                return True
            self.ui.waiting(f"server down, waiting for it to come back ({(limit - time.time()) / 60:.0f} min left)")
            for _ in range(20):
                if self.cancel.is_set():
                    raise Cancelled()
                time.sleep(1)
        return False


class Repeats:
    """stop a reply that loops: the same 120 characters 4 times in the last 3,000, or the same line 6 times running"""
    def __init__(self):
        self.buf, self.n = "", 0

    def feed(self, text):
        self.buf = (self.buf + text)[-3000:]; self.n += len(text)
        if self.n < 400:
            return False
        self.n = 0
        tail = self.buf[-120:]
        if len(tail.strip()) >= 40 and self.buf.count(tail) >= 4:
            return True
        lines = [l.strip() for l in self.buf.splitlines() if l.strip()]
        return len(lines) >= 6 and len(set(lines[-6:])) == 1 and len(lines[-1]) > 3


# ---------- session ----------

class Session:
    def __init__(self, c, path=None):
        self.c = c
        self.path = path or os.path.join(SESS, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
        site = (" " + c["site"].strip()) if c.get("site") else ""
        self.messages = [{"role": "system", "content": SYSTEM.format(site=site)}]
        self.pinned = 1                 # messages[:pinned] are never compacted (system prompt + pinned documents)
        self.web, self.think, self.docs, self.profile = True, c["default_think"], [], None
        self.ui = CliUI()

    def save(self):
        os.makedirs(SESS, exist_ok=True)
        tmp = self.path + ".tmp"
        json.dump({"messages": self.messages, "pinned": self.pinned, "web": self.web, "think": self.think,
                   "docs": self.docs, "profile": self.profile}, open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, self.path)

    @classmethod
    def load(cls, c, path):
        s = cls(c, path)
        j = json.load(open(path))
        s.messages, s.pinned, s.web, s.think = j["messages"], j["pinned"], j["web"], j["think"]
        s.docs, s.profile = j.get("docs", []), j.get("profile")
        return s

    def pin(self, paths):
        """documents go into one user message right after the system prompt (or after earlier pinned ones)"""
        blocks = [doc_block(p) for p in paths]
        self.docs += [os.path.abspath(os.path.expanduser(p)) for p in paths]
        self.messages[self.pinned:self.pinned] = [
            {"role": "user", "content": "Documents for this session:\n\n" + "\n\n".join(blocks)},
            {"role": "assistant", "content": "已读入文档。"}]
        self.pinned += 2


def size(s, srv, tools):
    n, exact = srv.count(s.messages, tools.specs())
    return n, exact


def compact(s, srv, tools, why=""):
    """Fold the earlier turns into a summary. Kept as they are: the system prompt and pinned documents (their cache
    stays valid) and the current turn (from the last user message on); in the current turn, tool results older
    than the last two are cut short if that is what it takes."""
    body = s.messages[s.pinned:]
    last_user = max((i for i, m in enumerate(body) if m["role"] == "user"), default=0)
    old, cur = body[:last_user], body[last_user:]
    s.ui.info(f"compacting{(' (' + why + ')') if why else ''}: {len(old)} earlier messages into a summary ...")
    if old:
        shrunk = [dict(m, content=cap(m["content"], 1500)) if m["role"] == "tool" and m.get("content") else m for m in old]
        req = s.messages[:s.pinned] + shrunk + [{"role": "user", "content":
              "Summarize our conversation so far for your own later use: the user's goals, what was found (document "
              "names and line numbers, numbers, names), decisions made, open questions. Plain text, complete but short."}]
        summary = ""
        for k, v in srv.stream(req, tools.specs(), "low", 8192):
            if k == "content":
                summary += v
        old = [{"role": "user", "content": "Summary of our conversation so far:\n" + summary.strip()},
               {"role": "assistant", "content": "好的，我记住了。"}]
    tools_in_cur = [i for i, m in enumerate(cur) if m["role"] == "tool"]
    for i in tools_in_cur[:-2]:
        cur[i] = dict(cur[i], content=cap(cur[i]["content"] or "", 800))
    s.messages = s.messages[:s.pinned] + old + cur
    s.save()


def ensure_room(s, srv, tools):
    n, exact = size(s, srv, tools)
    # compact only when the conversation itself has grown: pinned documents near the threshold must not make every
    # turn compact again
    if n > srv.p["compact_at"] and n - srv.count(s.messages[:s.pinned], tools.specs())[0] > 6000:
        compact(s, srv, tools, f"{'' if exact else '~'}{n:,} tokens > {srv.p['compact_at']:,}")
        n, exact = size(s, srv, tools)
        if n > srv.p["context"] - srv.p["max_tokens"]:
            s.ui.error(f"still {n:,} tokens after compacting: the pinned documents alone are too long for {srv.name}")
    return n


def drop_overlap(tail, text):
    """the continuation often starts by repeating the last few words before the cut: drop that part"""
    for k in range(min(len(tail), len(text)), 0, -1):
        if tail.endswith(text[:k]) and (k > 1 or re.match(r"[\u3400-\u9fff]", text[0])):
            return text[k:]
    return text


def one_call(s, srv, tools):
    """one model reply, with everything that would otherwise need the user: waiting out a down server, resending a
    lost request, compacting on a context error, continuing after a length cut, stopping a loop. Returns the tool
    calls of the reply (empty when it is a final answer)."""
    calls, cont, retries, prev_tail = [], 0, 0, ""
    while True:
        t0, first, content, fin = time.time(), None, "", None
        ui = s.ui
        hold = ""                     # after a length cut: the start of the continuation, held back until the part
                                      # that repeats the end of the cut-off text can be dropped
        rep = Repeats()
        try:
            for kind, val in srv.stream(s.messages, tools.specs(), s.think, srv.p["max_tokens"]):
                if first is None and kind in ("reasoning", "content", "tool"):
                    first = time.time() - t0
                if kind == "reasoning":
                    ui.think(val)
                    if rep.feed(val):
                        fin = "repeat"; break
                elif kind == "content":
                    content += val
                    if prev_tail and len(content) < 80:
                        hold += val; continue
                    if prev_tail and hold is not None:
                        val = drop_overlap(prev_tail, hold + val); hold = None
                    ui.text(val)
                    if rep.feed(val):
                        fin = "repeat"; break
                elif kind == "tool":
                    calls.append(val)
                elif kind == "finish":
                    fin = val
        except (KeyboardInterrupt, Cancelled):
            fin = "interrupted"
        except ServerDown as e:
            ui.error(str(e))
            if content or calls:
                fin = "server-down"                    # keep what came; the loop below decides what to do
            elif srv.wait_up():
                ui.info("server is back; sending the request again"); continue
            else:
                raise RuntimeError("the server did not come back")
        except Lost as e:
            ui.error(str(e))
            if retries < 1 and not content and not calls:
                retries += 1; ui.info("sending the request again"); continue
            fin = "lost"
        except TooLong:
            if retries < 2:
                retries += 1; compact(s, srv, tools, "the server said the context is too long"); continue
            raise RuntimeError("the context is too long even after compacting")
        if prev_tail and hold:
            ui.text(drop_overlap(prev_tail, hold))
        msg = {"role": "assistant", "content": content or None}
        if calls:
            msg["tool_calls"] = calls
        if content or calls:
            s.messages.append(msg)
        ui.end_reply(f"{first or 0:.1f} s to first token, {time.time() - t0:.0f} s, finish {fin}")
        broke = fin in ("server-down", "lost") and content
        if (fin == "length" or broke) and not calls and cont < s.c["auto_continue"]:
            cont += 1
            if fin == "server-down" and not srv.wait_up():
                raise RuntimeError("the server did not come back")
            ui.info(f"{'cut off by the length limit' if fin == 'length' else 'reply broke off'}: continuing ({cont}/{s.c['auto_continue']})")
            if not content:
                s.messages.append({"role": "assistant", "content": "（思考过长被截断）"})
            s.messages.append({"role": "user", "content": CONTINUE_NOTE})
            prev_tail = (content or "")[-80:]
            s.save()
            ensure_room(s, srv, tools)
            continue
        if fin == "repeat":
            ui.error("stopped: the reply was repeating itself")
            s.messages.append({"role": "user", "content": "（你刚才的输出开始重复，已被截停。请换一种方式，简洁地给出结论。）"})
            if retries < 1:
                retries += 1; continue
        if fin == "interrupted":
            raise Cancelled()
        if fin == "lost" and not content and not calls:
            raise RuntimeError("the server lost the request twice; nothing was answered")
        return calls


def turn(s, srv, tools):
    """a user turn: model replies and tool calls until a final answer"""
    for step in range(s.c["max_steps"]):
        ensure_room(s, srv, tools)
        calls = one_call(s, srv, tools)
        if not calls:
            s.save(); return
        for tc in calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            s.ui.tool(f"{fn}({', '.join(f'{k}={str(v)[:60]!r}' for k, v in args.items())})")
            out = tools.run(fn, args)
            s.messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "name": fn, "content": out})
        s.save()
    s.ui.error(f"stopped after {s.c['max_steps']} steps; say 'continue' to go on")


def status(s, srv, tools):
    n, exact = size(s, srv, tools)
    st = {"profile": srv.name, "model": srv.p["model"], "tokens": n, "exact": exact, "context": srv.p["context"],
          "compact_at": srv.p["compact_at"], "docs": len(s.docs), "doc_names": [os.path.basename(d) for d in s.docs],
          "web": s.web, "think": s.think}
    s.ui.status(st)
    return st


HELP = """commands: /doc FILE...  pin documents   /web on|off   /think off|low|medium|high   /model [name]
          /spark  /updates  /eco  (Spark state, no model call)   /status   /compact   /new   /save   /quit"""


def repl(s, srv, tools):
    print(f"{BOLD}spk{RST} → {srv.name}: {srv.p['model']} at {srv.base}\n{HELP}")
    status(s, srv, tools)
    while True:
        try:
            line = input(f"\n{BOLD}› {RST}").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            rest = rest.strip()
            try:
                if cmd in ("/quit", "/exit", "/q"): break
                elif cmd == "/doc": s.pin(rest.split()); s.save(); status(s, srv, tools)
                elif cmd == "/web": s.web = tools.web = rest == "on"; status(s, srv, tools)
                elif cmd == "/think" and rest in THINK_LEVELS: s.think = rest; status(s, srv, tools)
                elif cmd == "/model":
                    if rest in s.c["profiles"]:
                        srv = Server(s.c, rest); s.profile = srv.name; status(s, srv, tools)
                    else:
                        for k, v in s.c["profiles"].items(): print(f"  {k:10s} {v['model']}  {v['note']}")
                elif cmd == "/status": status(s, srv, tools)
                elif cmd in ("/spark", "/updates", "/eco"):
                    print({"/spark": spk_ops.status, "/updates": spk_ops.updates, "/eco": spk_ops.eco}[cmd]())
                elif cmd == "/compact": compact(s, srv, tools, "asked"); status(s, srv, tools)
                elif cmd == "/new":
                    web = s.web; s = Session(s.c); s.web = web; s.profile = srv.name; print("  new session")
                elif cmd == "/save": s.save(); print(f"  {s.path}")
                else: print(HELP)
            except (OSError, RuntimeError) as e:
                s.ui.error(str(e))
            continue
        ask(s, srv, tools, line)


def ask(s, srv, tools, text):
    if len(s.messages) == s.pinned:
        text = f"(today is {datetime.date.today().isoformat()})\n" + text
    s.messages.append({"role": "user", "content": text})
    s.ui.user(text)
    srv.ui, tools.ui, srv.cancel = s.ui, s.ui, getattr(s, "cancel", None) or srv.cancel
    try:
        turn(s, srv, tools)
    except (KeyboardInterrupt, Cancelled):
        s.ui.error("stopped (what was written is kept)")
    except RuntimeError as e:
        s.ui.error(str(e))
    s.save()
    status(s, srv, tools)


def main():
    ap = argparse.ArgumentParser(prog="spk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["doc", "run", "spark", "updates", "eco", "web"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("-q", "--question")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true")
    ap.add_argument("--web", action="store_true", help="allow web tools in doc mode")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--think", choices=THINK_LEVELS)
    ap.add_argument("-m", "--model", help="profile name from the config")
    ap.add_argument("--max-tokens", type=int, help="override the profile's max_tokens (per reply)")
    ap.add_argument("--compact-at", type=int, help="override the profile's compaction threshold (tokens)")
    ap.add_argument("--host", default="0.0.0.0", help="spk web: address to listen on")
    ap.add_argument("--port", type=int, default=8790, help="spk web: port")
    ap.add_argument("--link", action="store_true", help="spk web: print the links (with the access token) and exit")
    a = ap.parse_args()
    c = conf()
    if a.cmd in ("spark", "updates", "eco"):
        print({"spark": spk_ops.status, "updates": spk_ops.updates, "eco": spk_ops.eco}[a.cmd]()); return
    if a.cmd == "web":
        import spk_web
        if a.link:
            return print("\n".join(spk_web.addresses(a.port, spk_web.token())))
        return spk_web.main(c, a.host, a.port)
    if a.cont:
        files = sorted(f for f in os.listdir(SESS) if f.endswith(".json")) if os.path.isdir(SESS) else []
        if not files:
            sys.exit("no saved session")
        s = Session.load(c, os.path.join(SESS, files[-1]))
        say(f"  continuing {files[-1]}")
    else:
        s = Session(c)
        if a.cmd == "doc":
            s.web = a.web
            s.pin(a.args)
    srv = Server(c, a.model or s.profile or c["profile"])
    s.profile = srv.name
    if a.max_tokens: srv.p["max_tokens"] = a.max_tokens
    if a.compact_at: srv.p["compact_at"] = a.compact_at
    if a.no_web: s.web = False
    if a.think: s.think = a.think
    interactive = a.cmd != "run" and sys.stdin.isatty()
    s.ui.can_confirm = interactive
    tools = Tools(c, s.web, s.ui)
    if a.cmd == "run" or a.question:
        ask(s, srv, tools, a.question or " ".join(a.args))
        if not interactive:
            return
    repl(s, srv, tools)


if __name__ == "__main__":
    main()
