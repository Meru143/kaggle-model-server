"""headless comfyui as a managed backend for the video stacks, mirroring
harness.py's lifecycle style: install once per session, fetch a named model
stack, start the server with logs going to a file, expose it through the
same cloudflared tunnel helper, drive it over the http api.

usage (from any notebook that clones the repo, or the studio's video tab):
    import comfy_bootstrap as comfy
    comfy.install()
    comfy.fetch_stack("ltx-2.3")
    url = comfy.start()          # full comfyui node gui at this public url
    # ... queue_workflow(...) or click around the gui, then:
    comfy.stop()

logs: /kaggle/tmp/comfyui.log -- `!tail -50 /kaggle/tmp/comfyui.log`
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import requests

# big downloads must land on the ~60GB scratch disk, not the root volume
if os.path.isdir("/kaggle"):
    os.environ.setdefault("HF_HOME", "/kaggle/tmp/hf-home")

from huggingface_hub import hf_hub_download, list_repo_files

from harness import WORK_DIR, _tail, start_tunnel
from harness import _current as _harness_state

COMFY_DIR = f"{WORK_DIR}/ComfyUI"
COMFY_LOG = f"{WORK_DIR}/comfyui.log"

_current = {"proc": None, "log_fh": None, "port": None,
            "tunnel": None, "tunnel_fh": None}

# model stacks: (hf_repo, filename_in_repo, ComfyUI/models subdir).
# filenames verified against the repos (and the scail-2 / ltx cards' own
# placement recipes) in july 2026.
STACKS = {
    # ltx-2.3 22b audio+video gen. dev unet (20+ steps, better output) with
    # the distilled lora for refinement, per the unsloth card's workflow.
    # default Q3_K_M for t4 headroom; fetch_stack(..., unet="ltx-2.3-22b-dev-Q4_0.gguf")
    # if you want the bigger quant. text encoder is gemma-3-12b (gguf + mmproj).
    "ltx-2.3": [
        ("unsloth/LTX-2.3-GGUF", "ltx-2.3-22b-dev-Q3_K_M.gguf", "unet"),
        ("unsloth/LTX-2.3-GGUF", "vae/ltx-2.3-22b-dev_video_vae.safetensors", "vae"),
        ("unsloth/LTX-2.3-GGUF", "vae/ltx-2.3-22b-dev_audio_vae.safetensors", "vae"),
        ("unsloth/LTX-2.3-GGUF", "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors", "text_encoders"),
        ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-lora-384.safetensors", "loras"),
        ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "latent_upscale_models"),
        ("unsloth/gemma-3-12b-it-qat-GGUF", "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf", "text_encoders"),
        ("unsloth/gemma-3-12b-it-qat-GGUF", "mmproj-BF16.gguf", "text_encoders"),
    ],
    # scail-2 character animation / motion transfer (wan 2.1 14b backbone).
    # companion files exactly as the gguf card's "required files" table.
    "scail-2": [
        ("realrebelai/SCAIL-2_GGUF", "SCAIL-2-Q4_K_M.gguf", "unet"),
        ("Kijai/WanVideo_comfy", "umt5-xxl-enc-fp8_e4m3fn.safetensors", "text_encoders"),
        ("lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v",
         "loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors", "loras"),
        ("Comfy-Org/sam3.1", "checkpoints/sam3.1_multiplex_fp16.safetensors", "sam"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", "vae"),
    ],
    # EXPERIMENTAL: lingbot 30b-a3b moe text-to-video via the rebels node
    # pack. Q3_K_M streams from disk / page cache, fits 30GB ram comfortably.
    # needs structured json captions (use the pack's Structured Prompt node,
    # always set lighting) -- see the workflow json shipped in the gguf repo.
    "lingbot-30b": [
        ("realrebelai/LingBot-30B-3B_GGUF_ComfyUI", "LingBot-Video-30B-A3B-Q3_K_M.gguf", "unet"),
        ("realrebelai/LingBot_ComfyUI", "LingBot_text-encoder.safetensors", "text_encoders"),
        ("realrebelai/LingBot_ComfyUI", "LingBot_vae.safetensors", "vae"),
    ],
}

_LINGBOT_NODE_REPO = "https://github.com/RealRebelAI/ComfyUI_Rebels_LingBot"

# comfyui model subdirs we can map hf paths onto by name
_COMFY_DIRS = ("checkpoints", "clip", "clip_vision", "controlnet",
               "diffusion_models", "latent_upscale_models", "loras", "sam",
               "text_encoders", "unet", "upscale_models", "vae")


def detect_stack(repo, quant=None):
    """best-effort single-repo stack for STACKS: maps model files into comfyui
    dirs via the repo's own path segments (comfy-org repackaged repos and gguf
    packs follow this convention), keeping ONE unet gguf (~q4, <=13GB for a
    t4; quant= picks by name instead). returns (files, skipped) where files is
    STACKS-shaped [(repo, filename, subdir), ...].

    honest limit: multi-repo recipes (encoders/vaes hosted elsewhere) can't be
    machine-discovered -- the model card knows, the machine doesn't. those
    still deserve a curated STACKS entry."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    placed, denoisers, skipped = [], [], []
    for s in info.siblings:
        f = s.rfilename
        if not f.lower().endswith((".gguf", ".safetensors", ".sft", ".pt")):
            continue
        gb = round((s.size or 0) / 1e9, 2)
        segs = [p.lower() for p in f.split("/")[:-1]]
        sub = next((d for d in _COMFY_DIRS if d in segs), None)
        if sub in ("unet", "diffusion_models", "checkpoints"):
            denoisers.append((f, gb, sub))
        elif sub is None and f.lower().endswith(".gguf"):
            # bare ggufs in video packs are unet quants by convention
            denoisers.append((f, gb, "unet"))
        elif sub:
            placed.append((repo, f, sub))
        else:
            skipped.append(f)
    if not denoisers:
        raise ValueError(
            f"{repo} has no comfy-mappable denoiser (unet/diffusion_models/"
            f"checkpoints) -- not a video model pack? curated recipes live in STACKS.")
    # exactly ONE denoiser: repos ship every quant/precision variant, and
    # fetching them all is a 100GB mistake. companions (vae/clip/loras) are
    # small and all kept.
    if quant:
        q = quant.lower()
        cands = [d for d in denoisers if q in d[0].lower()] or denoisers
    else:
        cands = [d for d in denoisers if d[1] <= 13] or \
                [min(denoisers, key=lambda d: d[1])]
    pick = (next((d for d in cands if "q4_k_m" in d[0].lower()), None)
            or next((d for d in cands if "q4" in d[0].lower()), None)
            or next((d for d in cands if "fp8" in d[0].lower()), None)
            or min(cands, key=lambda d: d[1]))
    placed.insert(0, (repo, pick[0], pick[2]))
    skipped += [d[0] for d in denoisers if d[0] != pick[0]]
    return placed, skipped


def _clone(url, dst):
    if os.path.exists(dst):
        return
    subprocess.run(["git", "clone", "--depth", "1", url, dst], check=True)


def _pip_requirements(pkg_dir):
    req = os.path.join(pkg_dir, "requirements.txt")
    if os.path.exists(req):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req], check=True)


def install():
    """clone comfyui + the gguf/kjnodes packs, pip install each requirements.txt.
    idempotent -- re-running skips anything already present."""
    for url, dst in [
        ("https://github.com/comfyanonymous/ComfyUI", COMFY_DIR),
        ("https://github.com/city96/ComfyUI-GGUF", f"{COMFY_DIR}/custom_nodes/ComfyUI-GGUF"),
        ("https://github.com/kijai/ComfyUI-KJNodes", f"{COMFY_DIR}/custom_nodes/ComfyUI-KJNodes"),
    ]:
        _clone(url, dst)
        _pip_requirements(dst)
    print(f"comfyui ready at {COMFY_DIR}")


def _place(repo, filename, subdir):
    """hf-download onto the scratch disk, symlink into ComfyUI/models/<subdir>/"""
    local = hf_hub_download(repo_id=repo, filename=filename, local_dir=WORK_DIR)
    dst_dir = f"{COMFY_DIR}/models/{subdir}"
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(filename))
    if not os.path.lexists(dst):
        os.symlink(local, dst)
    return dst


def _place_workflows(repos):
    """any workflow json shipped in a stack's repos lands in the gui's
    workflow browser, so the first video isn't 'build a graph from scratch'"""
    wf_dir = f"{COMFY_DIR}/user/default/workflows"
    for repo in sorted(repos):
        try:
            for f in list_repo_files(repo):
                if f.lower().endswith(".json") and "workflow" in f.lower():
                    local = hf_hub_download(repo_id=repo, filename=f, local_dir=WORK_DIR)
                    os.makedirs(wf_dir, exist_ok=True)
                    shutil.copy(local, os.path.join(wf_dir, os.path.basename(f)))
                    print(f"workflow -> gui browser: {os.path.basename(f)}")
        except Exception as e:  # workflows are a bonus, never a blocker
            print(f"workflow scan skipped for {repo}: {e}")


def fetch_stack(key, unet=None):
    """downloads a named model set and symlinks it into comfyui's model dirs.
    unet= overrides just the unet gguf filename (e.g. a different quant)."""
    files = STACKS[key]
    if key == "lingbot-30b":
        # the rebels loader node + the 30b transformer config it expects
        node_dir = f"{COMFY_DIR}/custom_nodes/ComfyUI_Rebels_LingBot"
        _clone(_LINGBOT_NODE_REPO, node_dir)
        _pip_requirements(node_dir)
        cfg = hf_hub_download("robbyant/lingbot-video-moe-30b-a3b", "transformer/config.json")
        os.makedirs(f"{node_dir}/model_assets", exist_ok=True)
        shutil.copy(cfg, f"{node_dir}/model_assets/transformer_config_30b.json")
    for repo, filename, subdir in files:
        if unet and subdir == "unet":
            filename = unet
        print(f"fetching {repo} :: {filename} -> models/{subdir}/")
        _place(repo, filename, subdir)
    _place_workflows({repo for repo, _, _ in files})
    print(f"stack {key!r} in place")


def start(port=8188):
    """launches headless comfyui (logs to a file, never PIPE), waits for the
    http api, then exposes it through cloudflared and returns the public url --
    the full node gui is served at that url."""
    stop()
    log_fh = open(COMFY_LOG, "w")
    _current["log_fh"] = log_fh
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", str(port), "--force-fp16"],
        cwd=COMFY_DIR, stdout=log_fh, stderr=subprocess.STDOUT,
    )
    _current["proc"] = proc
    _current["port"] = port

    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"comfyui exited during startup. tail of {COMFY_LOG}:\n{_tail(COMFY_LOG)}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/", timeout=3).status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"comfyui not up within 180s. tail of {COMFY_LOG}:\n{_tail(COMFY_LOG)}")

    url = start_tunnel(port)
    # take ownership of the tunnel proc: an llm relaunch calls harness.stop(),
    # which must not tear down the video tunnel
    _current["tunnel"] = _harness_state["tunnel"]
    _harness_state["tunnel"] = None
    if _harness_state["log_fhs"]:
        _current["tunnel_fh"] = _harness_state["log_fhs"].pop()
    print(f"comfyui live at {url} (open it in a browser for the node gui)")
    return url


def queue_workflow(workflow, timeout=3600):
    """POSTs an API-format workflow, polls /history until done, returns output paths.

    export from the gui with "Export (API format)", then:
        wf = json.load(open("my_workflow_api.json"))
        paths = comfy.queue_workflow(wf)
    """
    port = _current["port"] or 8188
    r = requests.post(f"http://127.0.0.1:{port}/prompt", json={"prompt": workflow}, timeout=30)
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        entry = requests.get(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=30).json().get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"workflow failed. tail of {COMFY_LOG}:\n{_tail(COMFY_LOG)}")
            if status.get("completed"):
                paths = []
                for node_out in entry.get("outputs", {}).values():
                    for kind in ("images", "gifs", "videos", "audio"):
                        for f in node_out.get(kind, []):
                            paths.append(os.path.join(
                                COMFY_DIR, "output", f.get("subfolder", ""), f["filename"]))
                return paths
        time.sleep(3)
    raise TimeoutError(f"workflow still running after {timeout}s -- check {COMFY_LOG}")


def stop():
    """terminates the comfyui process (and its tunnel) cleanly, if up"""
    for key in ("proc", "tunnel"):
        proc = _current[key]
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        _current[key] = None
    for key in ("log_fh", "tunnel_fh"):
        if _current[key]:
            try:
                _current[key].close()
            except OSError:
                pass
        _current[key] = None
    _current["port"] = None
    print("stopped comfyui")
