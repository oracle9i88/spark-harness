"""
spk_ops: the Spark side of spk. Read-only: it looks at the five DGX Sparks and at the projects their stacks come from,
and changes nothing (installing updates stays the owner's call).

  status()    temperature, power, GPU load, memory, disk, containers, each serving endpoint's health
  updates()   driver / kernel / DGX packages per node, what apt already knows is upgradable, nodes that differ
  eco()       new commits / releases of the upstream projects and model repos since the last check
"""
import concurrent.futures as cf, datetime, json, os, subprocess, urllib.request

# Set from ~/.spk/config.json by configure(): "nodes": [{"name", "ssh", "via" (optional jump host), "role",
# "health" (optional URL)}], "eco_repos" / "eco_models": extra entries [name, what, pinned-or-null].
NODES = []
PROBE = r'''export LC_ALL=C
echo "up=$(uptime -p | sed 's/^up //')"
echo "gpu=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)"
t=0; for z in /sys/class/thermal/thermal_zone*/temp; do v=$(( $(cat $z) / 1000 )); [ $v -gt $t ] && t=$v; done; echo "soc_max=$t"
echo "mem_avail_gib=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)"
echo "disk_free=$(df -h / | awk 'NR==2{print $4}')"
echo "containers=$(docker ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')"
echo "load=$(cut -d' ' -f1-3 /proc/loadavg)"
'''
UPD = r'''export LC_ALL=C
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "kernel=$(uname -r)"
echo "os=$(. /etc/os-release; echo $VERSION)"
echo "apt_cache=$(date -r /var/cache/apt/pkgcache.bin +%F 2>/dev/null)"
echo "upgradable=$(apt list --upgradable 2>/dev/null | grep -c upgradable)"
apt list --upgradable 2>/dev/null | grep upgradable | sed 's/^/pkg=/'
'''

# upstream projects of Spark inference stacks; [repo, what, pinned (what we run) or None]
REPOS = [
    ["ashhart/TensorFold", "TensorFold inference engine", None],
    ["bertholomus/TensorFold", "TensorFold fork (DeepSeek on two Sparks)", None],
    ["MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold", "Qwen3.8 Flash Next on one Spark", None],
    ["0xSero/inkling2-exl3", "Inkling-2 exllamav3 overlay", None],
    ["oracle9i88/inkling2-exl3", "Inkling-2-Small on two or three Sparks (branch spark)", None],
    ["turboderp-org/exllamav3", "exllamav3", None],
    ["ajensenwaud/inkspark-2dgx", "Inkling on two DGX Sparks", None],
    ["NVIDIA/nccl", "NCCL", None],
    ["zyang-dev/nccl", "NCCL for 3 Sparks in a ring (branch dgxspark-3node-ring)", None],
    ["NVIDIA/dgx-spark-playbooks", "NVIDIA's DGX Spark playbooks", None],
]
BRANCH = {"oracle9i88/inkling2-exl3": "spark", "zyang-dev/nccl": "dgxspark-3node-ring"}
HF = [
    ["thinkingmachines/Inkling-2-Small", "Inkling-2-Small original", None],
    ["nvidia/Qwen3.8-Flash-Next-FP4-Local", "Qwen3.8 Flash Next FP4", None],
]


def configure(c):
    global NODES, REPOS, HF
    NODES = [(n["name"], n["ssh"], n.get("via"), n.get("role", ""), n.get("health")) for n in c.get("nodes", [])]
    pins = c.get("eco_pins", {})
    have = {r[0] for r in REPOS}
    REPOS = [[r, w, pins.get(r, p)] for r, w, p in REPOS] + [r for r in c.get("eco_repos", []) if r[0] not in have]
    have = {m[0] for m in HF}
    HF = [[m, w, pins.get(m, p)] for m, w, p in HF] + [m for m in c.get("eco_models", []) if m[0] not in have]


STATE = os.path.expanduser("~/.spk/eco-state.json")


def ssh(node, script, timeout=40):
    _, target, via, _, _ = node
    inner = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target, "bash -s"]
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", via, " ".join(inner[:-1]) + " 'bash -s'"] if via else inner
    try:
        r = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout)
        return r.stdout, (r.stderr.strip().splitlines() or [""])[-1] if r.returncode else ""
    except subprocess.TimeoutExpired:
        return "", "timed out"


def kv(text):
    out, pkgs = {}, []
    for line in text.splitlines():
        k, _, v = line.partition("=")
        if k == "pkg": pkgs.append(v)
        elif k: out[k] = v.strip()
    out["pkgs"] = pkgs
    return out


def http_health(url):
    if not url:
        return ""
    try:
        j = json.load(urllib.request.urlopen(url, timeout=5))
        a, q = j.get("active"), j.get("queued")
        return "ok" + (f" ({a} running, {q} waiting)" if a is not None else "")
    except Exception as e:
        return f"DOWN ({type(e).__name__})"


def status():
    if not NODES:
        return "no nodes in the config"
    with cf.ThreadPoolExecutor(len(NODES)) as ex:
        res = list(ex.map(lambda n: (n, ssh(n, PROBE), http_health(n[4])), NODES))
    rows, warn = [], []
    for n, (out, err), hl in res:
        name, role = n[0], n[3]
        if not out:
            rows.append(f"{name}  UNREACHABLE ({err})  {role}"); warn.append(f"{name} unreachable"); continue
        d = kv(out)
        g = [x.strip() for x in d.get("gpu", "").split(",")] + ["?"] * 4
        soc = int(d.get("soc_max", "0") or 0); gt = int(float(g[0])) if g[0] not in ("?", "", "[N/A]") else 0
        mem = float(d.get("mem_avail_gib", "0") or 0)
        rows.append(f"{name}  GPU {g[0]}°C {g[1]} W {g[2]}%  SoC max {soc}°C  mem free {mem:.1f} GiB  disk free {d.get('disk_free')}  "
                    f"load {d.get('load')}  up {d.get('up')}\n      {role}; containers: {d.get('containers') or '-'}"
                    + (f"; API {hl}" if hl else ""))
        if max(soc, gt) >= 85: warn.append(f"{name} hot ({max(soc, gt)}°C)")
        if mem < 2.0: warn.append(f"{name} low on memory ({mem:.1f} GiB free)")
        if hl.startswith("DOWN"): warn.append(f"{name} API {hl}")
    t = datetime.datetime.now().strftime("%F %T")
    return f"Sparks at {t}\n" + "\n".join(rows) + ("\nWARNINGS: " + "; ".join(warn) if warn else "\nno warnings")


KEY_PKGS = ("dgx-spark-ota", "dgx-release", "nvidia-driver", "libnvidia-compute", "nvidia-kernel", "linux-image",
            "linux-nvidia", "nvidia-firmware", "fwupd", "nvidia-container-toolkit")


def summarize_pkgs(pkgs):
    """'name/repo new arch [upgradable from: old]' -> the packages that matter one by one, the rest as counts"""
    key, groups = [], {}
    for p in pkgs:
        name = p.split("/")[0]; parts = p.split()
        new = parts[1] if len(parts) > 1 else "?"; old = p.rsplit(" ", 1)[-1].rstrip("]") if "from:" in p else "?"
        if name.startswith(KEY_PKGS):
            key.append(f"{name}: {old} -> {new}")
        else:
            g = "cuda" if name.startswith(("cuda", "libcu", "libnv", "nsight")) else "docker" if "docker" in name or name.startswith("containerd") \
                else "nvidia/dgx other" if ("nvidia" in name or "dgx" in name) else "other"
            groups[g] = groups.get(g, 0) + 1
    out = key[:12] + ([f"... {len(key) - 12} more key packages"] if len(key) > 12 else [])
    if groups:
        out.append("also: " + ", ".join(f"{n} {g}" for g, n in sorted(groups.items(), key=lambda x: -x[1])))
    return out


def updates():
    if not NODES:
        return "no nodes in the config"
    with cf.ThreadPoolExecutor(len(NODES)) as ex:
        res = list(ex.map(lambda n: (n, ssh(n, UPD, 60)), NODES))
    rows, seen = [], {}
    for n, (out, err) in res:
        if not out:
            rows.append(f"{n[0]}  UNREACHABLE ({err})"); continue
        d = kv(out)
        for k in ("driver", "kernel", "os"):
            seen.setdefault(k, {}).setdefault(d.get(k), []).append(n[0])
        rows.append(f"{n[0]}  driver {d.get('driver')}  kernel {d.get('kernel')}  DGX OS {d.get('os')}  "
                    f"apt cache {d.get('apt_cache')}  upgradable {d.get('upgradable')}")
        rows += [f"      {line}" for line in summarize_pkgs(d["pkgs"])]
    diff = [f"{k}: " + " vs ".join(f"{v} ({', '.join(ns)})" for v, ns in vals.items()) for k, vals in seen.items() if len(vals) > 1]
    note = ("\nNodes differ:\n  " + "\n  ".join(diff)) if diff else "\nall nodes on the same driver, kernel and OS"
    return ("Updates (read only; apt lists are as of each node's last 'apt update', shown as apt cache)\n" + "\n".join(rows) + note)


def _gh(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=30)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[:200])
    return json.loads(r.stdout)


def eco(mark_seen=True):
    try:
        state = json.load(open(STATE))
    except FileNotFoundError:
        state = {}
    first = not state
    out, new_state = [], dict(state)

    def one_repo(item):
        repo, what, pinned = item
        try:
            info = _gh(f"repos/{repo}")
            br = BRANCH.get(repo, info["default_branch"])
            commits = _gh(f"repos/{repo}/commits?sha={br}&per_page=8")
            try:
                rel = _gh(f"repos/{repo}/releases/latest").get("tag_name")
            except Exception:
                rel = None
            return repo, what, pinned, br, commits, rel, None
        except Exception as e:
            return repo, what, pinned, None, [], None, str(e)

    with cf.ThreadPoolExecutor(6) as ex:
        repos = list(ex.map(one_repo, REPOS))
    for repo, what, pinned, br, commits, rel, err in repos:
        if err:
            out.append(f"{repo}: could not read ({err})"); continue
        last = state.get(repo, {}).get("sha")
        head = commits[0]["sha"] if commits else None
        fresh = []
        for c in commits:
            if c["sha"] == last: break
            fresh.append(c)
        when = commits[0]["commit"]["committer"]["date"][:10] if commits else "?"
        tag = f", release {rel}" if rel else ""
        pin = ""
        if pinned and head and not head.startswith(pinned):
            pin = f"  [we run {pinned}]"
        if first or not last:
            out.append(f"{repo} ({what}): last commit {when} {head[:7] if head else ''}{tag}{pin}\n    " +
                       (commits[0]["commit"]["message"].splitlines()[0][:110] if commits else ""))
        elif fresh:
            out.append(f"{repo} ({what}): {len(fresh)}{'+' if len(fresh) == len(commits) else ''} new since last check{tag}{pin}\n" +
                       "\n".join(f"    {c['commit']['committer']['date'][:10]} {c['sha'][:7]} {c['commit']['message'].splitlines()[0][:100]}" for c in fresh))
        new_state[repo] = {"sha": head, "release": rel}
    for mid, what, pinned in HF:
        try:
            j = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{mid}", timeout=15))
            sha, mod = j.get("sha", ""), (j.get("lastModified") or "")[:10]
            last = state.get("hf:" + mid, {}).get("sha")
            pin = f"  [we use {pinned}]" if pinned and not sha.startswith(pinned) else ""
            if first or not last or last != sha:
                out.append(f"hf {mid} ({what}): {'changed, ' if last and last != sha else ''}revision {sha[:8]} of {mod}{pin}")
            new_state["hf:" + mid] = {"sha": sha}
        except Exception as e:
            out.append(f"hf {mid}: could not read ({type(e).__name__})")
    if mark_seen:
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        new_state["_checked"] = datetime.datetime.now().isoformat(timespec="minutes")
        json.dump(new_state, open(STATE, "w"), indent=1)
    head = "Ecosystem (first check: latest of each)" if first else f"Ecosystem: new since {state.get('_checked', '?')}"
    return head + "\n" + ("\n".join(out) if out else "nothing new")


if __name__ == "__main__":
    import sys
    configure(json.load(open(os.path.expanduser("~/.spk/config.json"))))
    print({"status": status, "updates": updates, "eco": eco}[sys.argv[1] if len(sys.argv) > 1 else "status"]())
