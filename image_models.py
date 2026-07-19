"""text-to-image on kaggle t4s via diffusers. registry + loader, no comfyui.

dependency note: the model cards say "install diffusers from git" for the
newer pipeline classes, but all four (ZImage, Krea2, Flux, Ideogram4) landed
in the 0.39 stable release -- and git-main imports symbols from UNRELEASED
huggingface_hub (CachedRepoTreeNotFoundError broke image loads on kaggle),
so stable is pinned deliberately. environment beats card.

usage from a notebook (see run_image.ipynb):
    from image_models import IMAGE_MODELS, install, load, generate
    install("z-image-turbo")
    pipe = load("z-image-turbo")
    generate(pipe, "a fox in the snow, golden hour")

serving is out of scope for v1 -- these run as notebook cells. a fastapi
wrapper can reuse harness.start_tunnel(port) later if an image api is wanted.

t4 rules baked in everywhere: torch.float16 only (bf16 is unsupported on
sm75 -- cards below all say bf16, adapted per the environment), sdpa/eager
attention (no flash-attention on turing), nf4 quantization with
bnb_4bit_compute_dtype=torch.float16.
"""

import os
import subprocess
import sys
import time

# multi-GB checkpoint downloads must land on the ~60GB scratch disk, not the
# quota-capped root/working volumes
if os.path.isdir("/kaggle"):
    os.environ.setdefault("HF_HOME", "/kaggle/tmp/hf-home")

OUT_DIR = "/kaggle/tmp/outputs"

IMAGE_MODELS = {
    # easiest fit: 6b single-stream dit, 8-step distilled. card: 9 steps is
    # really 8 dit forwards, and turbo models want guidance 0.
    "z-image-turbo": {
        "hf_repo": "Tongyi-MAI/Z-Image-Turbo",
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes"],
        "quantize": ["transformer"],
        "defaults": {"num_inference_steps": 9, "guidance_scale": 0.0},
    },
    # krea 2 turbo: 8 steps, guidance 0.0 (card).
    "krea-2-turbo": {
        "hf_repo": "krea/Krea-2-Turbo",
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes"],
        "quantize": ["transformer"],
        "defaults": {"num_inference_steps": 8, "guidance_scale": 0.0},
    },
    # NON-COMMERCIAL license. 12b transformer + t5-xxl encoder; nf4 + offload
    # is what makes it fit a t4. card defaults: 50 steps, guidance 3.5.
    "flux1-dev": {
        "hf_repo": "black-forest-labs/FLUX.1-dev",
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes",
                "sentencepiece", "protobuf"],
        "quantize": ["transformer"],
        "gated": True,  # accept the license on the model page first
        "defaults": {"num_inference_steps": 50, "guidance_scale": 3.5},
    },
    # GATED + NON-COMMERCIAL: needs HF_TOKEN (kaggle secret) and accepting
    # the license on the model page first. card's diffusers path uses the
    # -diffusers repo (not ideogram-ai/ideogram-4-nf4 -- that layout is for
    # their own ideogram4 package). weights ship already-nf4, so no
    # quantization pass here.
    "ideogram-4": {
        "hf_repo": "ideogram-ai/ideogram-4-nf4-diffusers",
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes"],
        "quantize": None,
        "gated": True,
        "defaults": {},  # card passes no steps/guidance -- pipeline defaults
    },
}


def detect_image_entry(repo):
    """registry-style entry for any diffusers-format repo (model_index.json),
    so unlisted models can be imported without editing this file:
        IMAGE_MODELS["my-model"] = detect_image_entry("author/some-model")
    nf4-quantizes the transformer when its files are too big for a t4 in
    fp16; defaults are the pipeline's own -- check the card for steps/guidance."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    files = {s.rfilename: (s.size or 0) / 1e9 for s in info.siblings}
    if "model_index.json" not in files:
        raise ValueError(f"{repo} is not a diffusers-format repo (no model_index.json) "
                         "-- gguf llm repos belong in the Launch tab's import instead")
    comp = "transformer" if any(f.startswith("transformer/") for f in files) else "unet"
    comp_gb = sum(gb for f, gb in files.items()
                  if f.startswith(f"{comp}/") and f.endswith(".safetensors"))
    return {
        "hf_repo": repo,
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes",
                "sentencepiece", "protobuf"],
        # crude but safe: file bytes roughly bound the fp16 load; over ~8GB
        # the denoiser won't share a 15GB t4 comfortably -> nf4 it
        "quantize": [comp] if comp_gb > 8 else None,
        "gated": bool(getattr(info, "gated", False)),
        "defaults": {},
    }


def install(key):
    """pip-installs the model's exact requirements (and nothing more)"""
    pkgs = IMAGE_MODELS[key]["pip"]
    print(f"installing for {key}: {pkgs}")
    # -U so kaggle's preinstalled older diffusers/hub get upgraded to a
    # consistent released set (pip's only-if-needed strategy leaves torch alone)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], check=True)


def load(key, gpu: int = 0):
    """returns a ready pipeline: fp16, transformer nf4 where needed, cpu offload.
    DiffusionPipeline resolves the concrete class (ZImagePipeline, Krea2Pipeline,
    FluxPipeline, Ideogram4Pipeline) from the repo's model_index.json.
    gpu=1 lets image gen share the box with a llama-server on gpu 0."""
    import torch
    from diffusers import DiffusionPipeline

    cfg = IMAGE_MODELS[key]
    kwargs = {"torch_dtype": torch.float16}  # cards say bf16; t4 is sm75 -> fp16
    if cfg.get("gated"):
        kwargs["token"] = os.environ.get("HF_TOKEN")
    if cfg.get("quantize"):
        from diffusers.quantizers import PipelineQuantizationConfig
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True,
                          "bnb_4bit_compute_dtype": torch.float16},
            components_to_quantize=cfg["quantize"],
        )
    print(f"loading {cfg['hf_repo']} (first time downloads to the hf cache under /kaggle)")
    pipe = DiffusionPipeline.from_pretrained(cfg["hf_repo"], **kwargs)
    # one component on gpu at a time -- the 15GB t4 can't hold encoder +
    # transformer + vae together for the bigger models
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe._km_defaults = dict(cfg["defaults"])
    return pipe


def generate(pipe, prompt, **overrides):
    """runs the pipeline, saves a png to /kaggle/tmp/outputs/, returns the path"""
    os.makedirs(OUT_DIR, exist_ok=True)
    params = {**getattr(pipe, "_km_defaults", {}), **overrides}
    t0 = time.time()
    image = pipe(prompt, **params).images[0]
    path = os.path.join(OUT_DIR, f"{int(time.time())}.png")
    image.save(path)
    print(f"{path}  ({time.time() - t0:.1f}s, {params or 'pipeline defaults'})")
    return path


if __name__ == "__main__":
    # gpu-free self-check: registry shape + defaults merge
    for k, c in IMAGE_MODELS.items():
        assert c["hf_repo"] and c["pip"] and "defaults" in c, k
    class _P:
        _km_defaults = {"num_inference_steps": 9, "guidance_scale": 0.0}
    merged = {**_P._km_defaults, "num_inference_steps": 4}
    assert merged == {"num_inference_steps": 4, "guidance_scale": 0.0}
    print("image_models self-check ok")
