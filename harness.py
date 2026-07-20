"""
shared inference harness for kaggle-hosted gguf models via llama.cpp.

handles: llama-server binary bootstrap (per-repo, since some models need a
fork), cloudflared bootstrap, weight fetching, server lifecycle, tunnel
exposure. one model runs at a time -- calling run() with a new model_key
always stops whatever's currently up first (displace, not coexist).

import this from a thin notebook, don't copy it into one. keep one copy of
this file as the source of truth and let every notebook pull it fresh.

logs: llama-server and cloudflared both write to files under WORK_DIR
(llama-server.log / cloudflared.log) -- `!tail -50 /kaggle/tmp/llama-server.log`
is your post-mortem tool.
"""

import glob
import os
import re
import shutil
import signal
import subprocess
import time
from collections import Counter

import requests
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

# ---- configurable paths -------------------------------------------------
# cache dataset: leave None and the harness AUTO-DISCOVERS any attached
# kaggle dataset that contains llama-server-* binaries or cloudflared (build
# one with harvest_cache() at the end of a session). set a path only to pin
# a specific dataset when several are attached.
CACHE_DATASET_DIR = None  # e.g. "/kaggle/input/llm-inference-cache"

_discovered_cache = None  # memo: None = not searched, False = none found


def _cache_dir():
    """explicit CACHE_DATASET_DIR wins; else scan /kaggle/input once for a
    dataset that looks like our cache."""
    global _discovered_cache
    if CACHE_DATASET_DIR:
        return CACHE_DATASET_DIR
    if _discovered_cache is None:
        _discovered_cache = False
        try:
            for d in sorted(os.listdir("/kaggle/input")):
                p = f"/kaggle/input/{d}"
                if os.path.isdir(p) and any(
                        f == "cloudflared" or f.startswith("llama-server")
                        for f in os.listdir(p)):
                    _discovered_cache = p
                    print(f"cache dataset auto-discovered: {p}")
                    break
        except OSError:
            pass
    return _discovered_cache or None

# everything big lives on the ephemeral scratch volume, NOT /kaggle/working.
# /kaggle/working is the ~20GB persistent-output volume -- a 20GB+ gguf
# won't even finish downloading there. /kaggle/tmp sits on the big (~60GB)
# scratch disk that doesn't persist, which is fine here: persistence is
# what CACHE_DATASET_DIR is for.
WORK_DIR = "/kaggle/tmp"

CLOUDFLARED_BIN = f"{WORK_DIR}/cloudflared"
SERVER_LOG = f"{WORK_DIR}/llama-server.log"
TUNNEL_LOG = f"{WORK_DIR}/cloudflared.log"

# models can override this per-entry with "llama_cpp_repo" (e.g. bonsai
# needs the prismml fork for its ternary kernels). each repo gets its own
# build dir and its own cached-binary name.
DEFAULT_LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp"

_current = {"server": None, "tunnel": None, "port": None, "log_fhs": []}


# ---- small helpers ------------------------------------------------------

def _ensure_workdir():
    os.makedirs(WORK_DIR, exist_ok=True)


def _tail(path, n=25):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no log yet)"


def _exec_copy(src, dst):
    """kaggle dataset mounts drop the executable bit, so copy out + chmod first"""
    shutil.copy(src, dst)
    os.chmod(dst, 0o755)
    return dst


def _repo_slug(repo_url):
    tail = repo_url.rstrip("/").split("github.com/")[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "-", tail).strip("-").lower()


# ---- launch progress (studio bar + eta) ---------------------------------
# one shared mechanism for all three stacks. the download phase gets a real
# bar+eta from bytes-on-disk vs the known total; build/load/pip are honestly
# indeterminate (no clean signal) so they show a phase label + elapsed, never
# a fake percentage.
_progress = {"phase": "idle", "total": 0.0, "watch": None, "base": 0.0, "t0": 0.0}

_PHASE_LABEL = {
    "build": "building llama.cpp from source — one-time this session (~15 min)",
    "load": "loading weights into vram",
    "install": "pip installing dependencies",
    "tunnel": "opening the tunnel",
}


def _dirsize(path):
    if not path or not os.path.isdir(path):
        return 0
    total = 0
    for root, dirs, files in os.walk(path):
        if "llama.cpp-" in root:  # build trees don't grow during downloads -- skip
            dirs[:] = []
            continue
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def set_progress(phase, total=0.0, watch=None):
    """studio progress hook. phase in {build,download,load,install,tunnel,idle}.
    download phases pass total (bytes) + watch (a dir that grows as bytes land)
    to enable a real bar; progress is measured as bytes-appeared-since-now, so
    pre-existing files in watch don't count."""
    _progress.update(phase=phase, total=float(total or 0), watch=watch,
                     base=float(_dirsize(watch)) if watch else 0.0, t0=time.time())


def _remote_size(cfg):
    try:
        info = HfApi().model_info(cfg["hf_repo"], files_metadata=True)
        return next((s.size for s in info.siblings
                     if s.rfilename == cfg["hf_file"]), 0) or 0
    except Exception:
        return 0


def _fmt_eta(s):
    s = int(max(s, 0))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def progress_line():
    """human-readable progress for the current phase; '' when idle."""
    ph = _progress["phase"]
    if ph == "idle":
        return ""
    elapsed = time.time() - _progress["t0"]
    if ph == "download" and _progress["total"] and _progress["watch"]:
        done = max(_dirsize(_progress["watch"]) - _progress["base"], 0)
        total = _progress["total"]
        if done > 50e6:  # a real download is in flight
            frac = min(done / total, 0.999)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            filled = int(frac * 24)
            bar = "█" * filled + "░" * (24 - filled)
            return (f"downloading  {bar}  {frac * 100:.0f}%  ·  "
                    f"{done / 1e9:.1f}/{total / 1e9:.1f} GB  ·  "
                    f"{rate / 1e6:.0f} MB/s  ·  eta {_fmt_eta(eta)}")
        return f"preparing download… ({_fmt_eta(elapsed)})"
    return f"{_PHASE_LABEL.get(ph, ph)} … ({_fmt_eta(elapsed)})"


# ---- binary bootstrap ---------------------------------------------------

def ensure_llamacpp_binary(repo_url=None):
    """returns a working llama-server path for the given llama.cpp repo,
    building from source if nothing's cached. mainline and forks coexist in
    separate build dirs, so swapping between them never rebuilds."""
    _ensure_workdir()
    repo_url = repo_url or DEFAULT_LLAMACPP_REPO
    slug = _repo_slug(repo_url)
    build_dir = f"{WORK_DIR}/llama.cpp-{slug}"
    built_bin = f"{build_dir}/build/bin/llama-server"

    # cached binary in the dataset? look for a per-repo name first, and let
    # a plain "llama-server" satisfy the mainline repo only.
    cache = _cache_dir()
    if cache:
        for name in ([f"llama-server-{slug}"] +
                     (["llama-server"] if repo_url == DEFAULT_LLAMACPP_REPO else [])):
            cached = f"{cache}/{name}"
            if os.path.exists(cached):
                local = f"{WORK_DIR}/{name}"
                if not os.path.exists(local):
                    print(f"copying cached binary out of dataset: {name}")
                    _exec_copy(cached, local)
                return local

    if os.path.exists(built_bin):
        print(f"using already-built llama-server: {built_bin}")
        return built_bin

    print(f"no cached binary for {slug}, building from source (one-time cost this session)")
    set_progress("build")
    # skip the clone if a previous (possibly failed) attempt left one behind --
    # git refuses to clone into a non-empty dir, which used to wedge retries
    if not os.path.exists(build_dir):
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, build_dir],
            check=True,
        )
    # a failed earlier configure poisons build/CMakeCache.txt (find_library
    # NOTFOUND results are cached and re-trusted, e.g. the libcuda.so probe),
    # so scrub the cache before re-configuring. compiled objects survive, so
    # an interrupted compile still resumes instead of starting over.
    shutil.rmtree(f"{build_dir}/build/CMakeFiles", ignore_errors=True)
    try:
        os.remove(f"{build_dir}/build/CMakeCache.txt")
    except OSError:
        pass
    subprocess.run(
        [
            "cmake", "-B", "build",
            "-DGGML_CUDA=ON",
            # kaggle's t4s are compute capability 7.5 -- building for just
            # that arch instead of every arch nvcc knows cuts compile time
            # by a large factor
            "-DCMAKE_CUDA_ARCHITECTURES=75",
            # llama.cpp wants libcurl for its built-in downloader and errors
            # out if the dev headers are missing; we fetch weights ourselves,
            # so switch it off rather than apt-get extra packages
            "-DLLAMA_CURL=OFF",
            # kaggle's image has no linkable libcuda.so anywhere cmake looks
            # (driver ships as libcuda.so.1 only, toolkit stubs are trimmed),
            # so the CUDA::cuda_driver target never exists and configure dies.
            # NO_VMM drops that link entirely (ggml's cmake: "no need to link
            # directly with the cuda driver lib") at the cost of the VMM pool
            # allocator -- negligible for single-model serving on t4s.
            "-DGGML_CUDA_NO_VMM=ON",
            # static build -> llama-server is one self-contained file, so
            # saving just that file to a Kaggle Dataset actually works (the
            # default shared build would also need the libggml/libllama .so's)
            "-DBUILD_SHARED_LIBS=OFF",
        ],
        cwd=build_dir, check=True,
    )
    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release",
         "--target", "llama-server", "-j", str(os.cpu_count() or 4)],
        cwd=build_dir, check=True,
    )
    print(f"tip: save {built_bin} to a Kaggle Dataset as 'llama-server-{slug}' "
          "so future sessions skip this build")
    return built_bin


def ensure_cloudflared():
    """returns a runnable cloudflared path, downloading it if needed
    (cloudflared is NOT preinstalled on kaggle)"""
    _ensure_workdir()
    found = shutil.which("cloudflared")
    if found:
        return found
    cache = _cache_dir()
    if cache and os.path.exists(f"{cache}/cloudflared"):
        if not os.path.exists(CLOUDFLARED_BIN):
            _exec_copy(f"{cache}/cloudflared", CLOUDFLARED_BIN)
        return CLOUDFLARED_BIN
    if os.path.exists(CLOUDFLARED_BIN):
        return CLOUDFLARED_BIN

    print("downloading cloudflared")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(CLOUDFLARED_BIN, "wb") as f:
        f.write(r.content)
    os.chmod(CLOUDFLARED_BIN, 0o755)
    return CLOUDFLARED_BIN


# ---- weight bootstrap ---------------------------------------------------

def _cached_weight_path(model_cfg):
    cache = _cache_dir()
    if cache:
        # subfoldered hf_files are stored flat in the dataset, so try both
        for name in (model_cfg["hf_file"], os.path.basename(model_cfg["hf_file"])):
            candidate = f"{cache}/{name}"
            if os.path.exists(candidate):
                return candidate
    return None


def ensure_weights(model_cfg):
    """returns a local path to the model's gguf, downloading from hf if not
    cached. hf_hub_download resumes partial downloads and no-ops if the file
    is already present. gated/private repos: put a token in kaggle secrets
    and export it as HF_TOKEN before calling run()."""
    _ensure_workdir()
    cached = _cached_weight_path(model_cfg)
    if cached:
        print(f"using cached weights: {cached}")
        return cached

    print(f"fetching {model_cfg['hf_file']} from {model_cfg['hf_repo']}")
    set_progress("download", total=_remote_size(model_cfg), watch=WORK_DIR)
    return hf_hub_download(
        repo_id=model_cfg["hf_repo"],
        filename=model_cfg["hf_file"],
        local_dir=WORK_DIR,
    )


# ---- quant switching + budget check --------------------------------------

# gguf sidecar files that are never the model itself
_AUX_GGUF = ("mmproj", "vae", "encoder")


def resolve_quant(repo, quant):
    """map a quant name ('Q3_K_M', 'MTP-Q4_K_M', 'UD-Q4_K_XL', ...) to the
    actual gguf filename in the repo, so nobody has to memorize filenames.
    prefers an exact -<quant>.gguf suffix, then the shortest matching name."""
    q = quant.lower()
    ggufs = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    cands = [f for f in ggufs
             if q in os.path.basename(f).lower()
             and not any(t in os.path.basename(f).lower() for t in _AUX_GGUF)]
    if not cands:
        avail = sorted({m.group(0) for f in ggufs for m in re.finditer(
            r"(?:UD-)?(?:I?Q\d[A-Za-z0-9_]*|BF16|F16|F32)", os.path.basename(f))})
        raise ValueError(
            f"no gguf matching quant {quant!r} in {repo}. "
            f"quants actually available: {', '.join(avail) or '(none?)'}"
        )
    exact = [f for f in cands if f.lower().endswith(f"-{q}.gguf")]
    return min(exact or cands, key=len)


def list_quants(model_key, registry):
    """prints every gguf in the model's repo with its size in GB -- the
    'what can i switch to' command. returns [(filename, size_gb), ...]."""
    repo = registry[model_key]["hf_repo"]
    info = HfApi().model_info(repo, files_metadata=True)
    rows = sorted((s.rfilename, round((s.size or 0) / 1e9, 2))
                  for s in info.siblings if s.rfilename.lower().endswith(".gguf"))
    for name, gb in rows:
        print(f"{gb:8.2f} GB  {name}")
    return rows


# card-scan: sampling key aliases -> llama-server/openai names
_CANON_SAMPLE = {
    "temperature": "temperature", "temp": "temperature",
    "topp": "top_p", "topk": "top_k", "minp": "min_p",
    "presencepenalty": "presence_penalty",
    "repetitionpenalty": "repeat_penalty", "repeatpenalty": "repeat_penalty",
    "frequencypenalty": "frequency_penalty",
}


def read_card_flags(repo, filename=None):
    """best-effort scan of a model card for its recommended sampling params
    and speculative-decoding flags -> {sampling, extra_args, notes}. this is
    regex heuristics, NOT an llm: cards list several modes and phrase things
    freely, so treat the result as SUGGESTIONS to verify. first value per key
    wins (cards usually lead with the general/default mode). {} on failure.

    filename gates the mtp flags: a card can discuss MTP because the repo ships
    separate -MTP- variants, but those flags only work on a gguf that actually
    has the drafter head -- so only suggest them when the chosen file (or the
    repo) is itself MTP."""
    try:
        p = hf_hub_download(repo, "README.md", token=os.environ.get("HF_TOKEN"))
        text = open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return {}
    low = text.lower()
    sampling = {}
    # separator between key and value tolerates the forms cards actually use:
    #   `temperature`: 0.85   **temp** = 1.0   --temp 0.6   | top_p | 0.95 |
    # bounded to <=6 chars of space/backtick/bold/pipe/=/: so it can't bridge
    # across prose words (letters aren't in the class, which stops false hits)
    pat = re.compile(
        r"\b(temp(?:erature)?|top[_\- ]?p|top[_\- ]?k|min[_\- ]?p|"
        r"presence[_\- ]penalty|repe(?:tition|at)[_\- ]penalty|frequency[_\- ]penalty)"
        r"[\s`*|=:]{1,6}([0-9]+(?:\.[0-9]+)?)", re.I)
    # neutral no-ops -- omit so suggestions stay clean (matches how the
    # registry entries were hand-authored)
    _noop = {"min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
             "repeat_penalty": 1.0, "top_k": 0}
    buckets = {}
    for m in pat.finditer(text):
        canon = re.sub(r"[_\- ]", "", m.group(1).lower())
        key = _CANON_SAMPLE.get(canon)
        if not key:
            continue
        v = m.group(2)
        buckets.setdefault(key, []).append(
            int(v) if key == "top_k" and "." not in v else float(v))
    # most frequent value per key: the real recommended value is repeated
    # across the card's mode recipes, while stray mentions appear once
    for key, vals in buckets.items():
        val = Counter(vals).most_common(1)[0][0]
        if _noop.get(key) != val:
            sampling[key] = val
    extra, notes = [], []
    if "--jinja" in low or "chat_template" in low or "tool call" in low or "tool-call" in low:
        extra.append("--jinja")
    is_mtp = "mtp" in (filename or "").lower() or "mtp" in repo.lower()
    if "draft-mtp" in low and is_mtp:  # speculative-decoding drafter
        if "--jinja" not in extra:
            extra.append("--jinja")
        if re.search(r"-fa\s+on|flash[\- ]?attn", low):
            extra += ["-fa", "on"]
        extra += ["--spec-type", "draft-mtp"]
        mm = re.search(r"spec-draft-n-max\s+(\d+)", low)
        extra += ["--spec-draft-n-max", mm.group(1) if mm else "2"]
        notes.append("mtp drafter detected")
    out = {}
    if sampling:
        out["sampling"] = sampling
    if extra:
        out["extra_args"] = extra
    if notes:
        out["notes"] = "; ".join(notes)
    return out


def detect_entry(repo, quant=None, read_card=False):
    """build a registry-style entry for an arbitrary gguf repo, auto-detecting
    the file and gpu placement from real file sizes: prefers ~Q4_K_M among
    files fitting the dual-t4 budget (<=26GB), then decides single vs dual t4
    at the 12GB line. lets you serve unlisted models without editing the
    registry:
        MODELS["my-model"] = detect_entry("bartowski/SomeModel-GGUF")
        run("my-model", MODELS)
    defaults are guesses -- check the model card for sampling params, MTP
    flags, and ctx quirks, then promote a proven entry into model_registry.py."""
    info = HfApi().model_info(repo, files_metadata=True)
    ggufs = [(s.rfilename, round((s.size or 0) / 1e9, 2)) for s in info.siblings
             if s.rfilename.lower().endswith(".gguf")
             and not any(t in s.rfilename.lower() for t in _AUX_GGUF)
             and "-of-" not in s.rfilename.lower()]  # sharded ggufs need every part
    if not ggufs:
        raise ValueError(f"no single-file gguf in {repo} -- is it a gguf repo?")
    if quant:
        name = resolve_quant(repo, quant)
        size = dict(ggufs).get(name, 0)
    else:
        fitting = [g for g in ggufs if g[1] <= 26] or [min(ggufs, key=lambda g: g[1])]
        name, size = (next((g for g in fitting if "q4_k_m" in g[0].lower()), None)
                      or next((g for g in fitting if "q4" in g[0].lower()), None)
                      or max(fitting, key=lambda g: g[1]))
    single = size <= 12
    entry = {
        "hf_repo": repo,
        "hf_file": name,
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None if single else "1,1",
        "n_cpu_moe": None,
        "gpu_devices": [0] if single else [0, 1],
        # --jinja applies the chat template shipped in the gguf -- the right
        # default for modern instruct models
        "extra_args": ["--jinja"],
        "est_vram_gb": round(size + 2, 1),  # file + rough kv/buffers at 8k ctx
    }
    if read_card:
        # pull the card's recommended sampling + mtp flags over the file-only
        # defaults (best-effort; the card overrides the blanket --jinja).
        # pass the chosen file so mtp flags only apply to an mtp gguf.
        card = read_card_flags(repo, filename=name)
        if card.get("sampling"):
            entry["sampling"] = card["sampling"]
        if card.get("extra_args"):
            entry["extra_args"] = card["extra_args"]
    return entry


def _warn_if_over_budget(cfg):
    """soft warning only, never a block -- the user may know better
    (e.g. n_cpu_moe parks expert weights in system ram)"""
    dual = bool(cfg.get("tensor_split")) or len(cfg.get("gpu_devices") or [0]) > 1
    budget = 26 if dual else 12
    try:
        info = HfApi().model_info(cfg["hf_repo"], files_metadata=True)
        size = next((s.size for s in info.siblings
                     if s.rfilename == cfg["hf_file"]), None)
    except Exception:
        return  # metadata fetch is best-effort; the download itself will tell
    if size and size / 1e9 > budget:
        print(f"\n{'!' * 60}\n"
              f"WARNING: {cfg['hf_file']} is {size / 1e9:.1f}GB, over the "
              f"~{budget}GB {'dual' if dual else 'single'}-gpu budget for this "
              f"config. proceeding anyway -- expect OOM unless you know better.\n"
              f"{'!' * 60}\n")


def harvest_cache():
    """stage everything cacheable into /kaggle/working so the notebook's
    Output tab -> "New Dataset" turns it into the boot cache: built
    llama-server binaries (per-repo names), cloudflared, and every gguf
    downloaded this session. skips files that would blow the ~19.5GB
    /kaggle/working quota. next session: attach the dataset -- the harness
    auto-discovers it, no config needed. returns the staged filenames."""
    out = "/kaggle/working"
    if not os.path.isdir(out):
        print("not on kaggle (/kaggle/working missing) -- nothing to do")
        return []
    budget = 19.0e9 - sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(out) for f in fs)
    staged, skipped = [], []

    def stage(src, name):
        nonlocal budget
        size = os.path.getsize(src)
        if size > budget:
            skipped.append((name, size))
            return
        shutil.copy(src, os.path.join(out, name))
        budget -= size
        staged.append(name)

    for build in glob.glob(f"{WORK_DIR}/llama.cpp-*/build/bin/llama-server"):
        slug = build.split("llama.cpp-", 1)[1].split("/", 1)[0]
        stage(build, f"llama-server-{slug}")
    if os.path.exists(CLOUDFLARED_BIN):
        stage(CLOUDFLARED_BIN, "cloudflared")
    for gguf in sorted(glob.glob(f"{WORK_DIR}/*.gguf")):
        stage(gguf, os.path.basename(gguf))

    print(f"staged {len(staged)} files in {out}: {', '.join(staged) or '(none)'}")
    for name, size in skipped:
        print(f"skipped {name} ({size / 1e9:.1f}GB): /kaggle/working quota")
    if staged:
        print("next: Save Version (quick save) -> notebook Output tab -> New Dataset. "
              "attach it to future notebooks and the harness finds it automatically.")
    return staged


# ---- server + tunnel lifecycle ------------------------------------------

def stop():
    """stops whatever's currently running (server + tunnel), if anything"""
    was_running = _current["port"] is not None
    for key in ("server", "tunnel"):
        proc = _current[key]
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        _current[key] = None
    for fh in _current["log_fhs"]:
        try:
            fh.close()
        except OSError:
            pass
    _current["log_fhs"] = []
    _current["port"] = None
    print("stopped current server + tunnel" if was_running else "nothing was running")


def _wait_for_health(proc, port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup. tail of {SERVER_LOG}:\n{_tail(SERVER_LOG)}"
            )
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def start_server(model_key, model_cfg, port=8080, api_key=None, health_timeout=600):
    """stops any running model, boots the requested one. always displaces, never
    coexists. model_cfg is an effective config dict (registry entry + overrides)."""
    stop()

    binary = ensure_llamacpp_binary(model_cfg.get("llama_cpp_repo"))
    weights = ensure_weights(model_cfg)

    cmd = [
        binary,
        "--model", weights,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--ctx-size", str(model_cfg.get("ctx", 8192)),
        "--n-gpu-layers", str(model_cfg.get("ngl", 99)),
    ]
    if model_cfg.get("tensor_split"):
        cmd += ["--tensor-split", model_cfg["tensor_split"], "--split-mode", "layer"]
    if model_cfg.get("n_cpu_moe") is not None:
        cmd += ["--n-cpu-moe", str(model_cfg["n_cpu_moe"])]
    if model_cfg.get("extra_args"):
        cmd += list(model_cfg["extra_args"])
    if api_key:
        cmd += ["--api-key", api_key]

    env = os.environ.copy()
    if model_cfg.get("gpu_devices"):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in model_cfg["gpu_devices"])

    # logs go to a file, NOT subprocess.PIPE: an unread PIPE fills its ~64KB
    # buffer and then silently blocks the server mid-session. a file also
    # survives for post-mortem.
    log_fh = open(SERVER_LOG, "w")
    _current["log_fhs"].append(log_fh)
    print(f"starting {model_key}: {' '.join(cmd)}")
    set_progress("load")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    _current["server"] = proc
    _current["port"] = port

    if not _wait_for_health(proc, port, health_timeout):
        raise RuntimeError(
            f"{model_key} didn't come up healthy within {health_timeout}s. "
            f"tail of {SERVER_LOG}:\n{_tail(SERVER_LOG)}"
        )
    print(f"{model_key} is up on port {port}")
    return proc


def _tunnel_cmd(binary, port, token):
    """named tunnel only for the default llm port -- the cloudflare ingress
    maps one hostname to one local service; other ports get quick tunnels"""
    if token and port == 8080:
        return [binary, "tunnel", "run", "--token", token]
    return [binary, "tunnel", "--url", f"http://localhost:{port}"]


def start_tunnel(port=8080):
    """launches a cloudflared tunnel, returns the public url.

    with a CF_TUNNEL_TOKEN env var (kaggle secret) this runs your NAMED
    tunnel instead of a throwaway quick tunnel -- the same hostname every
    session. one-time setup: cloudflare dashboard -> zero trust -> networks
    -> tunnels -> create, point its public hostname at http://localhost:8080,
    save the token as the CF_TUNNEL_TOKEN secret (and the hostname as
    CF_TUNNEL_HOSTNAME so the printed url is the real one)."""
    set_progress("tunnel")
    binary = ensure_cloudflared()
    token = os.environ.get("CF_TUNNEL_TOKEN")
    cmd = _tunnel_cmd(binary, port, token)
    log_fh = open(TUNNEL_LOG, "w")
    _current["log_fhs"].append(log_fh)
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    _current["tunnel"] = proc

    if "--token" in cmd:
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"cloudflared (named tunnel) exited. tail of {TUNNEL_LOG}:\n{_tail(TUNNEL_LOG)}")
            if "Registered tunnel connection" in _tail(TUNNEL_LOG, 200):
                host = os.environ.get("CF_TUNNEL_HOSTNAME")
                url = f"https://{host}" if host else "https://<your-tunnel-hostname>"
                print("named tunnel connected -- same hostname every session")
                return url
            time.sleep(0.5)
        raise RuntimeError(
            f"named tunnel didn't connect in time. tail of {TUNNEL_LOG}:\n{_tail(TUNNEL_LOG)}")

    url_pattern = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"cloudflared exited. tail of {TUNNEL_LOG}:\n{_tail(TUNNEL_LOG)}")
        match = url_pattern.search(_tail(TUNNEL_LOG, 200))
        if match:
            return match.group(0)
        time.sleep(0.5)
    raise RuntimeError(f"cloudflared didn't print a url in time. tail of {TUNNEL_LOG}:\n{_tail(TUNNEL_LOG)}")


def run(model_key, registry, *, port=8080, api_key=None, health_timeout=600, **overrides):
    """the one call the notebook actually makes: swap to model_key, expose it,
    return the url. any registry field (ctx, ngl, tensor_split, n_cpu_moe,
    gpu_devices, extra_args, hf_file, quant, ...) can be overridden per-call:
        run("bartowski/Qwen_Qwen3.6-35B-A3B-GGUF", MODELS, quant="Q3_K_M", ctx=16384)
    model_key is the full hf repo id (author/name) -- same as the registry key.
    quant= resolves to a real filename via resolve_quant, so switching quants
    never requires knowing the repo's naming scheme."""
    cfg = {**registry[model_key], **overrides}
    # resolve + sanity-check BEFORE stop(): a typo'd quant name should raise
    # while the currently-running model is still up, not after killing it
    if cfg.get("quant"):
        cfg["hf_file"] = resolve_quant(cfg["hf_repo"], cfg["quant"])
        print(f"quant {cfg['quant']!r} -> {cfg['hf_file']}")
    print(f"effective config for {model_key}: {cfg}")
    _warn_if_over_budget(cfg)
    start_server(model_key, cfg, port=port, api_key=api_key, health_timeout=health_timeout)
    url = start_tunnel(port)
    print(f"\n{model_key} live at {url}/v1  (openai-compatible)")
    if not api_key:
        print("note: this url is publicly reachable -- pass api_key='...' to run() to gate it")
    return url
