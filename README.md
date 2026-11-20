# spk — an agent harness for models on DGX Sparks

`spk` is a small command-line agent for models served on NVIDIA DGX Sparks (GB10), or on any OpenAI-compatible
server that reads long prompts slowly and has a prefix cache. It is one Python file plus one for machine operations,
standard library only.

It is built for long documents and long tasks, everyday questions with web search, and keeping an eye on the Sparks
themselves.

中文简介见文末。

## What it does differently

**It is shaped around how a Spark spends its time.** On two GB10s, reading a prompt costs about 1.5 ms per token
cold, while a prefix the server has cached is nearly free. So:

- the fixed preamble (system prompt and nine tools) is about 1,100 tokens, and it never changes: not per session,
  not per day (the date goes into the first message), not when web tools are switched off;
- the history is append-only, so every turn reuses the previous one's cache;
- documents are **pinned** right after the preamble: read once, and every later question only reads its own new
  tokens. Measured: two documents of 11,039 tokens pinned, the first question read them in 15.7 s, the follow-up
  read 521 new tokens in 1.5 s.

**It does not hang, and it does not stop to ask you to type "continue".**

- Every request is read on its own thread. While nothing arrives, spk asks the server every 15 s and shows what it
  is doing (`… no output for 45 s (server: 3 running, 2 waiting)`).
- Server down: spk says so, waits for it to come back (30 minutes by default) and sends the request again. It only
  re-sends when the server lost the request, never a second copy of one still being worked on.
- No output for 10 minutes while the server reports nothing running: the request was lost; spk sends it once more,
  then stops with a message.
- A reply cut off by the length limit is continued automatically (up to 6 times), and the words it repeats at the
  join are dropped from the screen.
- A reply that starts repeating itself is stopped, and the model is asked to answer differently.
- Ctrl-C stops the current reply and keeps what was written.

**It compacts on its own.** Before every model call, in the middle of a tool loop too, it counts the context
(exactly, with the server's `/tokenize`, where the engine has it) and compacts past the threshold: the earlier turns
become a summary, while the system prompt, the pinned documents (so their cache stays valid) and the current turn
stay as they are. A server that refuses the context as too long also triggers compaction.

**It looks after the Sparks (read only).** `spk spark` shows each node's GPU and SoC temperature, power, load, free
memory, disk, containers and API health, with warnings. `spk updates` lists driver, kernel, DGX OS and the key
upgradable packages per node and points out nodes that differ. `spk eco` reports new commits and releases of the
projects Spark inference stacks come from (TensorFold, exllamav3, NCCL, Inkling and Flash Next builds, model repos)
since the last check, and flags where upstream has moved past the version you run. The model can call all three as
tools. Nothing in spk installs, restarts or changes anything on the machines.

**Privacy.** In document mode web tools are off unless you pass `--web`. Every query that leaves the machine is
printed. The system prompt tells the model never to put private facts from the documents into a search.

## Install

```bash
git clone https://github.com/oracle9i88/spark-harness
ln -s "$PWD/spark-harness/spk.py" ~/.local/bin/spk
mkdir -p ~/.spk && cp spark-harness/config.example.json ~/.spk/config.json   # then edit it
```

Python 3.10 or newer. Optional: `rg` (search), `pdftotext` (PDF), `textutil` (macOS: docx), a Tavily helper script
for web search (`tavily` in the config: a script taking `QUERY`, or `-x URL`, and printing Tavily's JSON).

## Configure

`~/.spk/config.json` (see `config.example.json`):

- `profiles`: one per model server: `base_url`, `model`, `context`, `max_tokens`, `compact_at` (default 70% of the
  context), `tokenize` (the server has `/tokenize`), `slots`, `think_style`:
  - `budget`: `thinking_budget` in tokens (the [Inkling engine for Sparks](https://github.com/oracle9i88/inkling2-exl3));
  - `effort`, `effort-xhigh`, `default-off`: `reasoning_effort`, with "high" sent as `high` or `xhigh`, and "off"
    sent as `none` or not at all.
- `nodes`: the Sparks for `spk spark` / `spk updates`: `name`, `ssh` (user@host), optional `via` (a jump host),
  `role`, optional `health` (URL). spk uses `ssh -o BatchMode=yes`, so keys must already work.
- `site`: a sentence about your machines, added to the (fixed) system prompt.
- `eco_pins`: `{"repo or model": "the commit or revision you run"}`; `eco_repos` / `eco_models`: more to watch.
- `auto_continue`, `stall_seconds`, `server_down_wait`, `max_steps`, `tool_result_tokens`, `default_think`.

## Use

```bash
spk                                    # chat
spk doc contract.pdf notes.docx        # pin documents, then ask (web tools off)
spk doc report.md -q "what changed between versions?"
spk run "summarize ~/notes/today.md"   # one task, prints the answer, exits
spk -c                                 # continue the last session; spk -c run "..." for one more question
spk -m flashnext                       # use another profile
spk spark | spk updates | spk eco      # the machines, without the model
```

In a session: `/doc FILE...`, `/web on|off`, `/think off|low|medium|high`, `/model [name]`, `/spark`, `/updates`,
`/eco`, `/status`, `/compact`, `/new`, `/save`, `/quit`. Sessions are saved in `~/.spk/sessions/`.

Tools the model has: `read_file` (numbered lines; txt, md, csv, json, pdf, docx), `list_dir`, `grep`, `web_search`,
`fetch_url`, `spark_status`, `spark_updates`, `spark_ecosystem`, `write_file` (asks you first). There is no shell
tool.

## Web

```bash
spk web               # serves on all interfaces, port 8790; prints links that carry the access token
spk web --link        # print the links again
```

The same harness in a browser, phone included: sessions, streaming replies (thinking folded), documents by upload or
drag and drop, model / thinking / web switches, a context meter, a stop button, and the Spark panel (status, updates,
upstream news). A turn runs on the server, so closing or reloading the page loses nothing: it reconnects and replays
what it missed.

Access needs the token in `~/.spk/web-token` (made on first start); open a printed link once per device and it sets
a cookie. Requests without it get 403. The tools can read files on the machine running `spk web`, so treat the link
like a password, and prefer a private network (Tailscale, LAN) over exposing the port. Uploaded files go to
`~/.spk/uploads/`. Uploading the first document turns web tools off for that session, as in `spk doc`.

## Tests

`tests/run_tests.sh` runs spk against `tests/mock_server.py`, a fake server that misbehaves on purpose: a reply cut
by the length limit three times, a reply that loops, a server that accepts the request and never answers, a server
that is down. Each case must end on its own with the right message.

## 中文简介

`spk` 是给 DGX Spark 上的模型用的命令行助手。它的设计都围绕 Spark 的特点：读提示慢、前缀缓存几乎免费。

- **固定开头短而且永远不变**：约 1,100 个 token。历史只往后追加。文档钉在开头之后，只读一遍，之后每个问题只读新增的字。
- **不会卡死**：服务器排队、读提示、宕机、丢请求，都会显示出来并自动处理。宕机就等它回来再重发；丢了请求就重发一次。
- **不用手动输入"继续"**：被长度上限截断会自动续写，最多 6 次；开始复读就截停。
- **自动压缩**：每次调用模型前都会计数。超过阈值就把早先的对话压成摘要，钉住的文档和当前这轮不动。
- **网页版**：`spk web`，手机也能用；页面关掉或刷新，对话照样在服务器上跑完，重新打开就接上。要带口令的链接才能进。
- **Spark 运维（只读）**：温度、内存、服务状态、各节点的待装更新和版本差异，以及上游项目（TensorFold、exllamav3、NCCL、模型仓库）的新动态。不会安装或重启任何东西。
- **隐私**：文档模式默认关掉联网搜索，每条外发的查询都会显示出来。

## License

MIT
