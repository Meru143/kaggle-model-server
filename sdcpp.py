"""stable-diffusion.cpp as a third image backend, for the models comfy can't load.

ComfyUI-GGUF only loads archs in its IMG_ARCH_LIST (flux, sd1, sdxl, sd3, aura,
hidream, cosmos, ltxv, hyvid, wan, lumina2, qwen_image). Two models we want are
NOT in it -- ideogram4 and flux2 -- so their ggufs are unloadable there no matter
the quant. sd.cpp supports both natively (Ideogram4 landed 2026/06/04), and the
community ggufs for them are in fact sd.cpp bundles: rectangleworm's repo lays
out cond/ uncond/ text_encoder/ vae/, which maps 1:1 onto sd-cli's flags.

    import sdcpp
    sdcpp.install()                       # clone + cuda build, ~10-20 min once
    sdcpp.fetch("rectangleworm/ideogram-4-gguf")
    png = sdcpp.generate("rectangleworm/ideogram-4-gguf", "a fox in the snow")

logs: /kaggle/tmp/sdcpp.log -- `!tail -50 /kaggle/tmp/sdcpp.log`

ideogram4 wants a STRUCTURED JSON prompt (high_level_description /
style_description / ...), not a sentence -- comfy's official template builds one
too. generate() wraps a plain string into the minimal shape automatically.
"""

import json
import os
import shutil
import subprocess
import time

if os.path.isdir("/kaggle"):
    os.environ.setdefault("HF_HOME", "/kaggle/tmp/hf-home")

from huggingface_hub import hf_hub_download

from harness import WORK_DIR, _tail, set_progress

SD_REPO = "https://github.com/leejet/stable-diffusion.cpp"
SD_DIR = f"{WORK_DIR}/stable-diffusion.cpp"
SD_BIN = f"{SD_DIR}/build/bin/sd-cli"
SD_LOG = f"{WORK_DIR}/sdcpp.log"
OUT_DIR = "/kaggle/tmp/outputs"

# role -> (repo, path). roles map straight onto sd-cli flags.
# quants chosen for a 15GB t4 with --offload-to-cpu; the author's card suggests
# a bigger cond (Q6_K/Q8_0) if you have room -- override via fetch(quants=...).
SD_MODELS = {
    "rectangleworm/ideogram-4-gguf": {
        "diffusion": ("rectangleworm/ideogram-4-gguf", "diffusion/cond/ideogram4-Q4_K.gguf"),
        "uncond": ("rectangleworm/ideogram-4-gguf",
                   "diffusion/uncond/ideogram4_unconditional_Q4_K.gguf"),
        "llm": ("rectangleworm/ideogram-4-gguf", "text_encoder/Qwen3-VL-8B-Q4_K_M.gguf"),
        "vae": ("rectangleworm/ideogram-4-gguf", "vae/flux2-vae.safetensors"),
        "steps": 25, "cfg": 7.0, "json_prompt": True,
    },
}

_ROLE_FLAG = {"diffusion": "--diffusion-model", "uncond": "--uncond-diffusion-model",
              "llm": "--llm", "vae": "--vae"}

_paths = {}  # key -> {role: local path}, filled by fetch()


def install(jobs=None):
    """clone (with submodules) and build sd-cli with cuda. idempotent."""
    set_progress("install")
    if os.path.exists(SD_BIN):
        print(f"sd-cli already built: {SD_BIN}")
        return SD_BIN
    if not os.path.isdir(SD_DIR):
        # --recursive: sd.cpp vendors ggml as a submodule and won't configure without it
        subprocess.run(["git", "clone", "--recursive", "--depth", "1", SD_REPO, SD_DIR],
                       check=True)
    else:  # existing clone from a failed run -- make sure submodules are present
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"],
                       cwd=SD_DIR, check=False)

    build = f"{SD_DIR}/build"
    # a failed configure caches find_library NOTFOUND results and every retry
    # replays them -- same trap as llama.cpp, so scrub before configuring
    for stale in (f"{build}/CMakeCache.txt", f"{build}/CMakeFiles"):
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)
        elif os.path.exists(stale):
            os.remove(stale)
    os.makedirs(build, exist_ok=True)

    set_progress("build")
    with open(SD_LOG, "w") as log:
        try:
            cfg = ["cmake", "-B", build, "-S", SD_DIR,
                   "-DCMAKE_BUILD_TYPE=Release",
                   "-DSD_CUDA=ON",
                   "-DCMAKE_CUDA_ARCHITECTURES=75",   # t4 is sm75
                   # kaggle's image has no linkable libcuda.so, so cmake never creates
                   # CUDA::cuda_driver and the link step dies. same fix as llama.cpp.
                   "-DGGML_CUDA_NO_VMM=ON",
                   # skip libwebp/libwebm -- we only ever want png out
                   "-DSD_WEBP=OFF", "-DSD_WEBM=OFF"]
            subprocess.run(cfg, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=600)
            subprocess.run(["cmake", "--build", build, "--config", "Release",
                            "-j", str(jobs or os.cpu_count() or 4)],
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=3600)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"sd.cpp build failed. tail of {SD_LOG}:\n{_tail(SD_LOG)}") from e
    if not os.path.exists(SD_BIN):
        raise RuntimeError(f"build finished but {SD_BIN} is missing. tail of {SD_LOG}:\n"
                           f"{_tail(SD_LOG)}")
    os.chmod(SD_BIN, 0o755)
    print(f"sd-cli built: {SD_BIN}")
    return SD_BIN


def fetch(key, quants=None):
    """download the model's four parts. quants={'diffusion': 'path/in/repo', ...}
    overrides any role's file (e.g. a bigger cond quant)."""
    cfg = SD_MODELS[key]
    set_progress("download", total=_total(key, quants), watch=os.environ.get("HF_HOME"))
    got = {}
    for role in ("diffusion", "uncond", "llm", "vae"):
        if role not in cfg:
            continue
        repo, path = cfg[role]
        path = (quants or {}).get(role, path)
        print(f"fetching {role}: {repo} :: {path}")
        got[role] = hf_hub_download(repo_id=repo, filename=path)
    _paths[key] = got
    print(f"{key!r} ready ({len(got)} files)")
    return got


def _total(key, quants=None):
    from huggingface_hub import HfApi
    cfg, api, sizes, total = SD_MODELS[key], HfApi(), {}, 0
    for role in ("diffusion", "uncond", "llm", "vae"):
        if role not in cfg:
            continue
        repo, path = cfg[role]
        path = (quants or {}).get(role, path)
        if repo not in sizes:
            try:
                sizes[repo] = {s.rfilename: (s.size or 0)
                               for s in api.model_info(repo, files_metadata=True).siblings}
            except Exception:
                sizes[repo] = {}
        total += sizes[repo].get(path, 0)
    return total


def list_quants(key, role="diffusion"):
    """[(path, gb)] of ggufs available for one role, smallest first -- so the
    studio can offer a quant picker for sd.cpp models too. scoped to that
    role's own folder, so the cond model isn't offered the uncond weights."""
    cfg = SD_MODELS.get(key) or {}
    if role not in cfg:
        return []
    repo, path = cfg[role]
    folder = path.rsplit("/", 1)[0] + "/" if "/" in path else ""
    from huggingface_hub import HfApi
    rows = [(s.rfilename, (s.size or 0) / 1e9)
            for s in HfApi().model_info(repo, files_metadata=True).siblings
            if s.rfilename.lower().endswith(".gguf") and s.rfilename.startswith(folder)]
    return sorted(rows, key=lambda r: r[1])


def _gpu_count():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return sum(1 for l in out.stdout.splitlines() if l.startswith("GPU "))
    except Exception:
        return 1


def _vram_args(gpus=None):
    """how to fit ~16.5GB of weights.

    two cards -> pin modules to devices and keep everything RESIDENT in vram:
    both denoisers + vae on cuda0 (~11.4GB), the text encoder alone on cuda1
    (~5GB). that beats --offload-to-cpu, which parks weights in RAM and drags
    them back over pcie every time they're needed -- placement only moves
    activations at module boundaries, which are far smaller than weights.
    (no nvlink on t4, but that's irrelevant here: nothing is split mid-module.)

    one card -> fall back to cpu offload, since 16.5GB can't sit in 15GB.
    """
    gpus = _gpu_count() if gpus is None else gpus
    if gpus >= 2:
        # module key for the text encoder is 'llm' (it's loaded via --llm, and
        # 'llm' is what the source parses) -- NOT 'te'/'clip', which are the
        # keys for the older CLIP/T5 encoders. Getting this wrong doesn't error:
        # the assignment is just ignored and everything lands on device 0, which
        # is exactly how 5.0+5.5+5.5GB ooms a 14.6GB card.
        return ["--backend", "diffusion=cuda0,vae=cuda0,llm=cuda1"]
    return ["--offload-to-cpu"]


def _as_prompt(prompt, want_json):
    """ideogram4 expects a structured json prompt; a bare sentence gets wrapped
    into the minimal valid shape rather than confusing the model."""
    if not want_json:
        return prompt
    s = (prompt or "").strip()
    if s.startswith("{"):
        try:
            json.loads(s)
            return s          # already a json prompt, pass through untouched
        except ValueError:
            pass
    return json.dumps({"high_level_description": s})


def generate(key, prompt, width=1024, height=1024, steps=None, cfg_scale=None,
             seed=-1, init_image=None, strength=0.75, vram="auto", timeout=3600):
    """run sd-cli once and return the png path."""
    if not os.path.exists(SD_BIN):
        raise RuntimeError("sd-cli isn't built yet -- run sdcpp.install() first")
    if key not in _paths:
        raise RuntimeError(f"{key!r} not fetched yet -- run sdcpp.fetch({key!r}) first")
    cfg, files = SD_MODELS[key], _paths[key]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"sd_{int(time.time())}.png")

    cmd = [SD_BIN]
    for role, path in files.items():
        cmd += [_ROLE_FLAG[role], path]
    cmd += ["--prompt", _as_prompt(prompt, cfg.get("json_prompt")),
            "--width", str(int(width)), "--height", str(int(height)),
            "--steps", str(int(steps or cfg.get("steps", 20))),
            "--cfg-scale", str(float(cfg_scale if cfg_scale is not None else cfg.get("cfg", 7.0))),
            "--seed", str(int(seed)), "--output", out]
    if vram == "auto":
        cmd += _vram_args()
    elif vram == "offload":
        cmd += ["--offload-to-cpu"]
    elif vram == "split":
        cmd += _vram_args(gpus=2)
    if init_image:
        cmd += ["--init-img", init_image, "--strength", str(float(strength))]

    set_progress("load")
    with open(SD_LOG, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    set_progress("idle")
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"sd-cli failed (exit {proc.returncode}). tail of {SD_LOG}:\n"
                           f"{_tail(SD_LOG)}")
    return out


if __name__ == "__main__":
    # self-check: the command is assembled correctly without needing gpu/weights
    _paths["rectangleworm/ideogram-4-gguf"] = {
        "diffusion": "/w/cond.gguf", "uncond": "/w/uncond.gguf",
        "llm": "/w/llm.gguf", "vae": "/w/vae.safetensors"}
    assert set(_ROLE_FLAG) == {"diffusion", "uncond", "llm", "vae"}
    for flag in _ROLE_FLAG.values():
        assert flag.startswith("--")
    # plain text must become valid json for ideogram, json must pass through
    wrapped = _as_prompt("a fox in the snow", True)
    assert json.loads(wrapped)["high_level_description"] == "a fox in the snow"
    passthru = _as_prompt('{"high_level_description": "x"}', True)
    assert json.loads(passthru)["high_level_description"] == "x"
    assert _as_prompt("plain", False) == "plain"
    # broken json must be wrapped rather than sent as-is
    assert json.loads(_as_prompt('{"oops"', True))["high_level_description"] == '{"oops"'
    print("ok: sdcpp flags + json prompt wrapping")
