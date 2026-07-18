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

import os
import re
import shutil
import signal
import subprocess
import time

import requests
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

# ---- configurable paths -------------------------------------------------
# if you've got a Kaggle Dataset with prebuilt binaries and/or cached gguf
# weights, point this at it (attach the dataset to the notebook first, then
# set the slug below). leave it as None and the harness bootstraps
# everything fresh -- works with zero prior setup, just costs several
# minutes of quota the first time each session.
CACHE_DATASET_DIR = None  # e.g. "/kaggle/input/llm-inference-cache"

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
    if CACHE_DATASET_DIR:
        for name in ([f"llama-server-{slug}"] +
                     (["llama-server"] if repo_url == DEFAULT_LLAMACPP_REPO else [])):
            cached = f"{CACHE_DATASET_DIR}/{name}"
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
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, build_dir],
        check=True,
    )
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
    if CACHE_DATASET_DIR and os.path.exists(f"{CACHE_DATASET_DIR}/cloudflared"):
        if not os.path.exists(CLOUDFLARED_BIN):
            _exec_copy(f"{CACHE_DATASET_DIR}/cloudflared", CLOUDFLARED_BIN)
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
    if CACHE_DATASET_DIR:
        candidate = f"{CACHE_DATASET_DIR}/{model_cfg['hf_file']}"
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
    return hf_hub_download(
        repo_id=model_cfg["hf_repo"],
        filename=model_cfg["hf_file"],
        local_dir=WORK_DIR,
    )


# ---- quant switching + budget check --------------------------------------

def resolve_quant(repo, quant):
    """map a quant name ('Q3_K_M', 'MTP-Q4_K_M', 'UD-Q4_K_XL', ...) to the
    actual gguf filename in the repo, so nobody has to memorize filenames.
    prefers an exact -<quant>.gguf suffix, then the shortest matching name."""
    q = quant.lower()
    ggufs = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    cands = [f for f in ggufs
             if q in os.path.basename(f).lower()
             and not any(t in os.path.basename(f).lower()
                         for t in ("mmproj", "vae", "encoder"))]
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


def start_tunnel(port=8080):
    """launches a cloudflared quick tunnel, returns the public url.
    swap the command for your named-tunnel invocation if you're not on quick tunnels."""
    binary = ensure_cloudflared()
    log_fh = open(TUNNEL_LOG, "w")
    _current["log_fhs"].append(log_fh)
    proc = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://localhost:{port}"],
        stdout=log_fh, stderr=subprocess.STDOUT,
    )
    _current["tunnel"] = proc

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
        run("qwen3.6-35b-a3b-hotswap", MODELS, quant="Q3_K_M", ctx=16384)
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
