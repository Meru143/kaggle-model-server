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
import random
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

from harness import WORK_DIR, _tail, set_progress, start_tunnel
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
    # lingbot dense 1.3b, same rebels node pack -- the fast small sibling.
    # (the ALX fp8 variant is skipped on purpose: its W8A8 fp8 matmul needs
    # sm89+, t4 is sm75. this bf16 repack runs everywhere.)
    "lingbot-1.3b": [
        ("realrebelai/LingBot_ComfyUI", "LingBot_1.3b_DiT.safetensors", "unet"),
        ("realrebelai/LingBot_ComfyUI", "LingBot_text-encoder.safetensors", "text_encoders"),
        ("realrebelai/LingBot_ComfyUI", "LingBot_vae.safetensors", "vae"),
    ],
    # sulphur-2: ltx-2.3 finetune (video). unet from the abiray gguf mirror,
    # standard ltx companions, plus sulphur's own distill lora; its t2v/i2v
    # workflow jsons land in the gui browser automatically.
    "sulphur-2": [
        ("Abiray/Sulphur-2-base-GGUF", "sulphur_dev-Q3_K_M.gguf", "unet"),
        ("unsloth/LTX-2.3-GGUF", "vae/ltx-2.3-22b-dev_video_vae.safetensors", "vae"),
        ("unsloth/LTX-2.3-GGUF", "vae/ltx-2.3-22b-dev_audio_vae.safetensors", "vae"),
        ("unsloth/LTX-2.3-GGUF", "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors", "text_encoders"),
        ("unsloth/gemma-3-12b-it-qat-GGUF", "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf", "text_encoders"),
        ("unsloth/gemma-3-12b-it-qat-GGUF", "mmproj-BF16.gguf", "text_encoders"),
        ("SulphurAI/Sulphur-2-base", "distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors", "loras"),
        ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "latent_upscale_models"),
    ],
    # 10eros 1.4: explicit-content ltx-2.3 finetune (nsfw output violates
    # kaggle tos -- your account, your risk). same companions as ltx-2.3.
    "ltx-10eros": [
        ("vantagewithai/LTX2.3-10Eros-1.4-GGUF", "10Eros_v1.4-Q3_K_M.gguf", "unet"),
        ("unsloth/LTX-2.3-GGUF", "vae/ltx-2.3-22b-dev_video_vae.safetensors", "vae"),
        ("unsloth/LTX-2.3-GGUF", "vae/ltx-2.3-22b-dev_audio_vae.safetensors", "vae"),
        ("unsloth/LTX-2.3-GGUF", "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors", "text_encoders"),
        ("unsloth/gemma-3-12b-it-qat-GGUF", "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf", "text_encoders"),
        ("unsloth/gemma-3-12b-it-qat-GGUF", "mmproj-BF16.gguf", "text_encoders"),
        ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-lora-384.safetensors", "loras"),
        ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "latent_upscale_models"),
    ],
    # z-image turbo IMAGE gen in comfy -- the reliable path on t4: comfy's
    # fp16 numerics work on bf16-less cards where the diffusers pipeline
    # NaNs to black frames. Q8_0 is near-lossless at 7.2GB. use comfy's
    # built-in z-image workflow template in the gui.
    "z-image": [
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q8_0.gguf", "unet"),
        ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors", "text_encoders"),
        ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", "vae"),
    ],
    # krea 2 turbo IMAGE gen in comfy (image ggufs run here, not llama.cpp).
    # companions per the vantage workflow: qwen3-vl encoder + qwen-image vae.
    "krea2-turbo": [
        ("vantagewithai/Krea-2-Turbo-GGUF", "krea2_turbo-Q4_K_M.gguf", "unet"),
        ("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "text_encoders"),
        ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "vae"),
    ],
    # krea 2 raw (image, 52-step quality tier): comfy-org's single-repo
    # repack, fp8 storage dequanted on t4. comfy fallback for the diffusers
    # entry in case fp16 blackframes it.
    "krea2-raw": [
        ("Comfy-Org/Krea-2", "diffusion_models/krea2_raw_fp8_scaled.safetensors", "diffusion_models"),
        ("Comfy-Org/Krea-2", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "text_encoders"),
        ("Comfy-Org/Krea-2", "vae/qwen_image_vae.safetensors", "vae"),
    ],
    # flux.1-dev (image): the classic low-vram comfy recipe -- city96 gguf
    # unet + fp8 t5 + clip_l. the vae comes from the GATED bfl repo: needs
    # HF_TOKEN + accepted flux license (same as running flux at all).
    # Q8_0 is the community quality pick; unet="flux1-dev-Q4_K_S.gguf" for
    # more headroom.
    "flux1": [
        ("city96/FLUX.1-dev-gguf", "flux1-dev-Q8_0.gguf", "unet"),
        ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "text_encoders"),
        ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors", "text_encoders"),
        ("black-forest-labs/FLUX.1-dev", "ae.safetensors", "vae"),
    ],
    # ideogram 4 (image): comfy-org's complete official repack, fp8 storage.
    # both transformers (conditional + cfg branch) + the 8b qwen3-vl encoder.
    # GATED lineage upstream but this repack is open. comfy fallback for the
    # diffusers ideogram entries.
    "ideogram4": [
        ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_fp8_scaled.safetensors", "diffusion_models"),
        ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors", "diffusion_models"),
        ("Comfy-Org/Ideogram-4", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors", "text_encoders"),
        ("Comfy-Org/Ideogram-4", "vae/flux2-vae.safetensors", "vae"),
    ],
    # krea 2 turbo HD finetune (image): ships its own hd-tuned vae; same
    # qwen3-vl encoder. Q6_K (10.9GB) is the quality pick if vram allows.
    "krea2-hd": [
        ("wikeeyang/Krea2-Turbo-HD-V1", "Krea2-Turbo-HD-V1-Q4_K_S.gguf", "unet"),
        ("wikeeyang/Krea2-Turbo-HD-V1", "Krea2-HD-vae.safetensors", "vae"),
        ("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "text_encoders"),
    ],
    # flux2-klein 9b finetune (image): comfy-org publishes the matching
    # encoder (fp8 storage, fine on t4) + vae; example workflow ships in-repo.
    "flux2-klein-v3": [
        ("wikeeyang/Flux2-Klein-9B-True-V3", "Flux2-Klein-9B-True-V3-Q4_K.gguf", "unet"),
        ("Comfy-Org/flux2-klein-9b", "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors", "text_encoders"),
        ("Comfy-Org/flux2-klein-9b", "split_files/vae/flux2-vae.safetensors", "vae"),
    ],
}

# which stacks render stills vs motion. comfy is the reliable both-t4 image
# path on t4 (diffusers NaNs to black frames), but image is still a different
# job from video -- the studio lists them in separate dropdowns so the video
# picker isn't polluted with image models.
IMAGE_STACKS = frozenset({
    "z-image", "krea2-turbo", "krea2-raw", "krea2-hd",
    "flux1", "flux2-klein-v3", "ideogram4",
})
assert IMAGE_STACKS <= STACKS.keys(), \
    f"IMAGE_STACKS names not in STACKS: {IMAGE_STACKS - STACKS.keys()}"


def video_stacks():
    """stack keys that produce video (+ any imported packs, which default here)"""
    return sorted(k for k in STACKS if k not in IMAGE_STACKS)


def image_stacks():
    """stack keys that produce stills -- the comfy image path"""
    return sorted(k for k in STACKS if k in IMAGE_STACKS)


def stack_repo(key):
    """the stack's primary (first/denoiser) hf repo -- lets the studio label its
    dropdowns with a real author/name instead of the short internal alias."""
    files = STACKS.get(key)
    return files[0][0] if files else key


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


def _stack_total(files):
    """sum bytes of a stack's specific files, for the studio download bar"""
    from huggingface_hub import HfApi
    api, sizes, total = HfApi(), {}, 0
    for repo, fn, _ in files:
        if repo not in sizes:
            try:
                sizes[repo] = {s.rfilename: (s.size or 0)
                               for s in api.model_info(repo, files_metadata=True).siblings}
            except Exception:
                sizes[repo] = {}
        total += sizes[repo].get(fn, 0)
    return total


def install():
    """clone comfyui + the gguf/kjnodes packs, pip install each requirements.txt.
    idempotent -- re-running skips anything already present."""
    set_progress("install")
    for url, dst in [
        ("https://github.com/comfyanonymous/ComfyUI", COMFY_DIR),
        ("https://github.com/city96/ComfyUI-GGUF", f"{COMFY_DIR}/custom_nodes/ComfyUI-GGUF"),
        ("https://github.com/kijai/ComfyUI-KJNodes", f"{COMFY_DIR}/custom_nodes/ComfyUI-KJNodes"),
        # lets a workflow pin each model to a specific card. comfy is otherwise
        # single-gpu: on t4x2 the text encoder alone is 5-9GB, so parking it on
        # gpu1 is what stops the big stacks spilling to cpu ram (= minutes/step)
        ("https://github.com/pollockjj/ComfyUI-MultiGPU",
         f"{COMFY_DIR}/custom_nodes/ComfyUI-MultiGPU"),
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


def _looks_like_workflow(f):
    """'workflow' in the name, or a root-level json that isn't a config
    (some packs ship the workflow as e.g. Vantage_Krea-2-Turbo.json)"""
    n = f.lower()
    if not n.endswith(".json"):
        return False
    if "workflow" in n:
        return True
    return "/" not in f and not any(x in n for x in ("config", "index", "tokenizer"))


def _place_workflows(repos):
    """any workflow json shipped in a stack's repos lands in the gui's
    workflow browser, so the first video isn't 'build a graph from scratch'"""
    wf_dir = f"{COMFY_DIR}/user/default/workflows"
    for repo in sorted(repos):
        try:
            for f in list_repo_files(repo):
                if _looks_like_workflow(f):
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
    set_progress("download", total=_stack_total(files), watch=WORK_DIR)
    if key.startswith("lingbot"):
        # every lingbot variant loads through the rebels node pack
        node_dir = f"{COMFY_DIR}/custom_nodes/ComfyUI_Rebels_LingBot"
        _clone(_LINGBOT_NODE_REPO, node_dir)
        _pip_requirements(node_dir)
        if key == "lingbot-30b":
            # plus the 30b transformer config the moe loader expects
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
    set_progress("load")
    log_fh = open(COMFY_LOG, "w")
    _current["log_fh"] = log_fh
    proc = subprocess.Popen(
        # --enable-cors-header '*' swaps comfy's origin-only middleware -- which
        # 403s any request whose Host/Origin don't match, i.e. EVERY request
        # arriving through a cloudflared tunnel hostname -- for permissive cors,
        # so the public url actually loads the gui instead of "403 not authorized".
        # (the url is the only secret anyway; comfy has no auth either way.)
        [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", str(port),
         "--force-fp16", "--enable-cors-header", "*"],
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


# ---- headless image generation ------------------------------------------
# flat /prompt (api-format) graphs per image stack so the studio can do
# prompt->png WITHOUT anyone opening the node gui. every node type + sampler
# recipe below is transcribed from the official comfy templates
# (github.com/Comfy-Org/workflow_templates) -- not invented -- and the loaders
# point at exactly the files fetch_stack() downloads. this is the reliable
# image path on t4: comfy's fp16 works where the diffusers pipelines black-frame.
#
# families: 'ksampler' (z-image / krea2 / flux1 -- classic KSampler), 'flux2'
# (SamplerCustomAdvanced + Flux2Scheduler), 'ideogram4' (dual-transformer;
# EXPERIMENTAL -- its DualModelGuider/Ideogram4Scheduler nodes are newer and
# unverified on this build, so it may still need the gui).
_IMAGE_RECIPE = {
    "z-image": dict(family="ksampler", unet=("gguf", "z-image-turbo-Q8_0.gguf"),
                    clip=("qwen_3_4b_fp8_mixed.safetensors", "lumina2"),
                    vae="ae.safetensors", latent="EmptySD3LatentImage",
                    shift=3.0, steps=8, cfg=1.0, sampler="res_multistep", scheduler="simple"),
    "krea2-turbo": dict(family="ksampler", unet=("gguf", "krea2_turbo-Q4_K_M.gguf"),
                    clip=("qwen3vl_4b_fp8_scaled.safetensors", "krea2"),
                    vae="qwen_image_vae.safetensors", latent="EmptyLatentImage",
                    steps=8, cfg=1.0, sampler="euler", scheduler="simple"),
    "krea2-hd": dict(family="ksampler", unet=("gguf", "Krea2-Turbo-HD-V1-Q4_K_S.gguf"),
                    clip=("qwen3vl_4b_fp8_scaled.safetensors", "krea2"),
                    vae="Krea2-HD-vae.safetensors", latent="EmptyLatentImage",
                    steps=8, cfg=1.0, sampler="euler", scheduler="simple"),
    "krea2-raw": dict(family="ksampler", unet=("safetensors", "krea2_raw_fp8_scaled.safetensors"),
                    clip=("qwen3vl_4b_fp8_scaled.safetensors", "krea2"),
                    vae="qwen_image_vae.safetensors", latent="EmptyLatentImage",
                    steps=40, cfg=4.0, sampler="euler", scheduler="simple"),
    "flux1": dict(family="ksampler", unet=("gguf", "flux1-dev-Q8_0.gguf"),
                    clip=("clip_l.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "flux"),
                    vae="ae.safetensors", latent="EmptySD3LatentImage",
                    steps=20, cfg=1.0, sampler="euler", scheduler="simple"),
    "flux2-klein-v3": dict(family="flux2", unet=("gguf", "Flux2-Klein-9B-True-V3-Q4_K.gguf"),
                    clip=("qwen_3_8b_fp8mixed.safetensors", "flux2"),
                    vae="flux2-vae.safetensors", steps=20, cfg=5.0),
    "ideogram4": dict(family="ideogram4", unet=("safetensors", "ideogram4_fp8_scaled.safetensors"),
                    unet_uncond="ideogram4_unconditional_fp8_scaled.safetensors",
                    clip=("qwen3vl_8b_fp8_scaled.safetensors", "ideogram4"),
                    vae="flux2-vae.safetensors", steps=25, cfg=7.0, mu=0.5, std=1.75),
}
assert set(_IMAGE_RECIPE) == set(IMAGE_STACKS), \
    f"image recipe / IMAGE_STACKS mismatch: {set(_IMAGE_RECIPE) ^ set(IMAGE_STACKS)}"


_NODE_CACHE = {}


def _has_node(cls):
    """does the RUNNING comfy actually expose this node class? asked once via
    /object_info. lets the builder use the ComfyUI-MultiGPU loaders when the
    pack is there and silently fall back to core nodes when it isn't -- a
    missing class_type would otherwise 400 the whole prompt."""
    if "nodes" not in _NODE_CACHE:
        try:
            r = requests.get(f"http://127.0.0.1:{_current['port'] or 8188}/object_info",
                             timeout=10)
            r.raise_for_status()
            _NODE_CACHE["nodes"] = set(r.json())
        except Exception:
            _NODE_CACHE["nodes"] = set()  # comfy down / old build -> core only
    return cls in _NODE_CACHE["nodes"]


def _second_gpu():
    """'cuda:1' only when the box really has two cards -- otherwise placement is
    pointless and pinning a nonexistent device would fail the load"""
    try:
        import torch
        return "cuda:1" if torch.cuda.device_count() > 1 else None
    except Exception:
        return None


def _loader_node(spec, device=None):
    """model loader: gguf -> UnetLoaderGGUF (ComfyUI-GGUF), else UNETLoader.
    UNETLoader reads models/unet AND models/diffusion_models, so fp8 safetensors
    placed in diffusion_models/ load fine. device= pins it to one card."""
    kind, fn = spec
    if kind == "gguf":
        base, mg = {"unet_name": fn}, "UnetLoaderGGUFMultiGPU"
        core = "UnetLoaderGGUF"
    else:
        base, mg = {"unet_name": fn, "weight_dtype": "default"}, "UNETLoaderMultiGPU"
        core = "UNETLoader"
    if device and _has_node(mg):
        return {"class_type": mg, "inputs": {**base, "device": device}}
    return {"class_type": core, "inputs": base}


def _clip_node(clip, device=None):
    if len(clip) == 3:  # dual (flux1): clip_l + t5
        base = {"clip_name1": clip[0], "clip_name2": clip[1], "type": clip[2]}
        core, mg = "DualCLIPLoader", "DualCLIPLoaderMultiGPU"
    else:
        base = {"clip_name": clip[0], "type": clip[1]}
        core, mg = "CLIPLoader", "CLIPLoaderMultiGPU"
    if device and _has_node(mg):
        return {"class_type": mg, "inputs": {**base, "device": device}}
    return {"class_type": core, "inputs": base}


def _vae_node(vae, device=None):
    if device and _has_node("VAELoaderMultiGPU"):
        return {"class_type": "VAELoaderMultiGPU",
                "inputs": {"vae_name": vae, "device": device}}
    return {"class_type": "VAELoader", "inputs": {"vae_name": vae}}


def _round16(px, default):
    px = int(px) if px else default
    return max(256, (px // 16) * 16)  # 16-multiple keeps every latent packer happy


def build_image_workflow(stack, prompt, width=None, height=None, steps=None, seed=None):
    """flat api-format graph for one image stack, prompt/size/steps/seed injected"""
    r = _IMAGE_RECIPE[stack]
    w, h = _round16(width, 1024), _round16(height, 1024)
    steps = int(steps) if steps else r["steps"]
    seed = random.randint(0, 2**63 - 1) if seed is None else int(seed)
    fam = r["family"]
    # the whole speed story on t4x2: the text encoder (5-9GB) goes on the idle
    # second card so the denoiser gets gpu0 to itself. without this comfy is
    # single-gpu, the big stacks overflow 15GB and stream weights from cpu ram
    # every step. falls back to plain single-card loaders on a 1-gpu box.
    enc_dev = _second_gpu()
    main_dev = "cuda:0" if enc_dev else None

    if fam == "ksampler":
        g = {"1": _loader_node(r["unet"], main_dev), "2": _clip_node(r["clip"], enc_dev),
             "3": _vae_node(r["vae"], main_dev)}
        model_ref = ["1", 0]
        if r.get("shift") is not None:  # z-image needs AuraFlow model-sampling
            g["4"] = {"class_type": "ModelSamplingAuraFlow",
                      "inputs": {"shift": r["shift"], "model": ["1", 0]}}
            model_ref = ["4", 0]
        g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}}
        g["6"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}}
        g["7"] = {"class_type": r["latent"],
                  "inputs": {"width": w, "height": h, "batch_size": 1}}
        g["8"] = {"class_type": "KSampler",
                  "inputs": {"seed": seed, "steps": steps, "cfg": r["cfg"],
                             "sampler_name": r["sampler"], "scheduler": r["scheduler"],
                             "denoise": 1.0, "model": model_ref,
                             "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}}
        g["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}}
        g["10"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "km", "images": ["9", 0]}}
        return g

    if fam == "flux2":  # SamplerCustomAdvanced pipeline, empty-prompt negative
        g = {"1": _loader_node(r["unet"], main_dev), "2": _clip_node(r["clip"], enc_dev),
             "3": _vae_node(r["vae"], main_dev),
             "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
             "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
             "7": {"class_type": "CFGGuider",
                   "inputs": {"cfg": r["cfg"], "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0]}},
             "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": w, "height": h}},
             "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
             "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
             "11": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
             "12": {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["10", 0], "guider": ["7", 0], "sampler": ["9", 0],
                               "sigmas": ["8", 0], "latent_image": ["11", 0]}},
             "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
             "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "km", "images": ["13", 0]}}}
        return g

    if fam == "ideogram4":  # dual transformer (main + unconditional), best-effort
        # BOTH transformers on gpu0 (~11GB together) and the 8b encoder alone on
        # gpu1 (~9GB): that's the split that fits, and it's why this one went
        # from 20 min of cpu-swapping to something usable.
        g = {"1": _loader_node(r["unet"], main_dev),
             "1u": _loader_node(("safetensors", r["unet_uncond"]), main_dev),
             "2": _clip_node(r["clip"], enc_dev),
             "3": _vae_node(r["vae"], main_dev),
             "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
             "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
             "7": {"class_type": "DualModelGuider",
                   "inputs": {"cfg": r["cfg"], "model": ["1", 0], "positive": ["5", 0],
                              "model_negative": ["1u", 0], "negative": ["6", 0]}},
             "8": {"class_type": "Ideogram4Scheduler",
                   "inputs": {"steps": steps, "width": w, "height": h, "mu": r["mu"], "std": r["std"]}},
             "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
             "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
             "11": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
             "12": {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["10", 0], "guider": ["7", 0], "sampler": ["9", 0],
                               "sigmas": ["8", 0], "latent_image": ["11", 0]}},
             "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
             "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "km", "images": ["13", 0]}}}
        return g

    raise ValueError(f"unknown image family {fam!r} for stack {stack!r}")


def generate_image(stack, prompt, width=None, height=None, steps=None, seed=None, timeout=1800):
    """headless prompt->png: builds the stack's workflow, queues it on the
    already-running comfy server, returns the saved png path. call start()
    (via the studio's Install+load) once for the stack before generating."""
    if stack not in _IMAGE_RECIPE:
        raise ValueError(f"{stack!r} has no headless image recipe (video stack?)")
    if not (_current["proc"] and _current["proc"].poll() is None):
        raise RuntimeError("comfy isn't running -- press Install + load for this image stack first")
    wf = build_image_workflow(stack, prompt, width, height, steps, seed)
    paths = queue_workflow(wf, timeout=timeout)
    imgs = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    if not imgs:
        raise RuntimeError(f"workflow produced no image -- check {COMFY_LOG}")
    return imgs[0]


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


if __name__ == "__main__":
    # graph-integrity self-check: every input that references another node
    # ([node_id, slot]) must point at a node that exists in the same graph.
    # catches a mistyped node id in build_image_workflow before it 400s on comfy.
    for _stack in _IMAGE_RECIPE:
        _g = build_image_workflow(_stack, "a test prompt", width=1000, height=800, steps=7)
        for _nid, _node in _g.items():
            assert isinstance(_node.get("class_type"), str) and _node["class_type"], _nid
            for _k, _v in _node["inputs"].items():
                if isinstance(_v, list) and len(_v) == 2 and isinstance(_v[0], str):
                    assert _v[0] in _g, f"{_stack}: node {_nid}.{_k} -> missing {_v[0]}"
        assert any(n["class_type"] == "SaveImage" for n in _g.values()), _stack
        assert _round16(1000, 1024) == 992 and _round16(None, 768) == 768
    print(f"ok: {len(_IMAGE_RECIPE)} image workflows build, all refs resolve")
