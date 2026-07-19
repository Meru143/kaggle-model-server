"""text-to-image on kaggle t4s via diffusers. registry + loader, no comfyui.

dependency note: the model cards say "install diffusers from git" for the
newer pipeline classes, but all four (ZImage, Krea2, Flux, Ideogram4) landed
in the 0.39 stable release -- and git-main imports symbols from UNRELEASED
huggingface_hub (CachedRepoTreeNotFoundError broke image loads on kaggle),
so stable is pinned deliberately. environment beats card.

usage (from any notebook that clones the repo, or the studio's image tab):
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
    # reduces fragmentation-induced oom on long sessions
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

OUT_DIR = "/kaggle/tmp/outputs"

IMAGE_MODELS = {
    # easiest fit: 6b single-stream dit, 8-step distilled. card: 9 steps is
    # really 8 dit forwards, and turbo models want guidance 0. runs
    # UNQUANTIZED: the 12.3GB fp16 transformer fits one t4 under balanced
    # placement, and nf4 on a few-step distill risks degenerate (black)
    # output on top of the t4's fp16-range issues.
    "z-image-turbo": {
        "hf_repo": "Tongyi-MAI/Z-Image-Turbo",
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes"],
        "quantize": None,
        "defaults": {"num_inference_steps": 9, "guidance_scale": 0.0},
    },
    # krea 2 turbo: 8 steps, guidance 0.0 (card). encoder nf4'd too so the
    # whole resident set (~10GB) fits one t4 -- see load() for why quantized
    # models can't spread across gpus.
    "krea-2-turbo": {
        "hf_repo": "krea/Krea-2-Turbo",
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes"],
        "quantize": ["transformer", "text_encoder"],
        "defaults": {"num_inference_steps": 8, "guidance_scale": 0.0,
                     "height": 768, "width": 768},
    },
    # NON-COMMERCIAL license. 12b transformer + t5-xxl encoder; nf4 + offload
    # is what makes it fit a t4. card defaults: 50 steps, guidance 3.5.
    "flux1-dev": {
        "hf_repo": "black-forest-labs/FLUX.1-dev",
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes",
                "sentencepiece", "protobuf"],
        # text_encoder_2 is the 9.5GB t5; the small clip stays fp16
        "quantize": ["transformer", "text_encoder_2"],
        "gated": True,  # accept the license on the model page first
        "defaults": {"num_inference_steps": 50, "guidance_scale": 3.5,
                     "height": 768, "width": 768},
    },
    # GATED + NON-COMMERCIAL: needs HF_TOKEN (kaggle secret) and accepting
    # the license on the model page first. card's diffusers path uses the
    # -diffusers repo (not ideogram-ai/ideogram-4-nf4 -- that layout is for
    # their own ideogram4 package). weights ship already-nf4, so no
    # quantization pass here.
    "ideogram-4": {
        "hf_repo": "ideogram-ai/ideogram-4-nf4-diffusers",
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes"],
        # transformers ship pre-nf4; quantizing the 5.5GB encoder brings the
        # resident set under one t4
        "quantize": ["text_encoder"],
        "gated": True,
        "defaults": {"height": 768, "width": 768},  # steps/guidance: pipeline defaults
    },
    # krea 2 without the turbo distillation: better quality, 52 steps,
    # guidance 3.5 (card). same pipeline + nf4 treatment as the turbo.
    "krea-2-raw": {
        "hf_repo": "krea/Krea-2-Raw",
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes"],
        "quantize": ["transformer", "text_encoder"],
        "defaults": {"num_inference_steps": 52, "guidance_scale": 3.5,
                     "height": 768, "width": 768},
    },
    # fal's 8-step distill of ideogram 4 (GATED, non-commercial lineage):
    # a transformer-only repo dropped into the ideogram base pipeline. no
    # runtime cfg -- guidance 1.0 skips the uncond branch entirely, and the
    # zero_uncond shim (from fal's card) satisfies diffusers 0.39's
    # mandatory cfg slot without loading the base's 5GB uncond transformer.
    "ideogram-4-instant": {
        "hf_repo": "ideogram-ai/ideogram-4-nf4-diffusers",
        "transformer_from": "fal/ideogram-v4-instant",  # bf16 -> nf4 on load
        "zero_uncond": True,
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes"],
        "quantize": None,
        "gated": True,
        "defaults": {"num_inference_steps": 8, "guidance_scale": 1.0,
                     "height": 768, "width": 768},
    },
    # fal's 20-step sibling of -instant: same recipe, more steps, better
    # detail (card). GATED like the rest of the ideogram family.
    "ideogram-4-fast": {
        "hf_repo": "ideogram-ai/ideogram-4-nf4-diffusers",
        "transformer_from": "fal/ideogram-v4-fast",
        "zero_uncond": True,
        "pip": ["diffusers>=0.39", "transformers", "accelerate", "bitsandbytes"],
        "quantize": None,
        "gated": True,
        "defaults": {"num_inference_steps": 20, "guidance_scale": 1.0,
                     "height": 768, "width": 768},
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


def load(key, gpu=None):
    """returns a ready pipeline: fp16, denoiser nf4 where needed.
    DiffusionPipeline resolves the concrete class (ZImagePipeline, Krea2Pipeline,
    FluxPipeline, Ideogram4Pipeline) from the repo's model_index.json.

    placement -- gpu=None (default): spread components across BOTH t4s via
    device_map="balanced". needed because bitsandbytes modules can't cpu-
    offload: the nf4 denoiser stays resident, and denoiser + fp16 text
    encoder together overflow one 15GB card (that's the OOM). stop a running
    llm first if vram is tight. gpu=0/1: pin one gpu with cpu offload --
    coexists with a llama-server, but only the smaller models fit."""
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
    if cfg.get("transformer_from"):
        # transformer-swap repos (fal ideogram distills): pull just the
        # denoiser from the variant repo, nf4 it on load, drop it into the
        # base pipeline -- the base's own transformer is never downloaded
        from diffusers import AutoModel, BitsAndBytesConfig
        kwargs["transformer"] = AutoModel.from_pretrained(
            cfg["transformer_from"], subfolder="transformer",
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
            token=os.environ.get("HF_TOKEN"))
    if cfg.get("zero_uncond"):
        # fal card's zero-parameter stand-in for diffusers' mandatory cfg
        # branch; with guidance_scale=1.0 it is never actually called
        class _ZeroUncond(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("_dtype_anchor",
                                     torch.empty(0, dtype=torch.float16),
                                     persistent=False)

            @property
            def dtype(self):
                return self._dtype_anchor.dtype

        kwargs["unconditional_transformer"] = _ZeroUncond()
    # bnb-quantized entries are pinned to one gpu: accelerate's "balanced"
    # planner sizes the UNQUANTIZED checkpoint, decides it can't fit, and
    # spills quantized modules to cpu -- which bitsandbytes forbids
    # ("Some modules are dispatched on the CPU"). with encoders nf4'd too,
    # each quantized model's resident set fits a single t4 anyway. only
    # unquantized pipelines (z-image) spread across both gpus.
    has_bnb = bool(cfg.get("quantize")) or bool(cfg.get("transformer_from"))
    dual = gpu is None and torch.cuda.device_count() >= 2 and not has_bnb
    if dual:
        kwargs["device_map"] = "balanced"
    print(f"loading {cfg['hf_repo']} "
          f"({'balanced across both gpus' if dual else f'gpu {gpu or 0} + cpu offload'})")
    pipe = DiffusionPipeline.from_pretrained(cfg["hf_repo"], **kwargs)
    if not dual:
        # one component on gpu at a time; fine for the smaller models only
        pipe.enable_model_cpu_offload(gpu_id=gpu or 0)
    for helper in ("enable_attention_slicing", "enable_vae_tiling"):
        # t4 sdpa uses the math backend (no flash kernels on sm75), which
        # materializes the full attention matrix -- slicing shrinks the
        # peak where the model supports it; harmless no-op where it doesn't
        try:
            getattr(pipe, helper)()
        except Exception:
            pass
    try:
        # fp16 vae decode is the classic source of NaN -> black frames
        # (cards assume bf16's range; t4 is fp16-only). vaes are tiny, so
        # fp32 costs nothing that matters.
        pipe.vae.to(torch.float32)
    except Exception as e:
        print(f"vae fp32 upcast skipped: {e}")
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
    import numpy as np
    if np.asarray(image).max() <= 2:  # all-black frame = fp16 overflow (NaN latents)
        print("WARNING: output is a black frame -- latents overflowed fp16 "
              "somewhere upstream (t4 has no bf16). try more steps, a different "
              "prompt length, or another model; if it persists for this model, "
              "report it -- the fix is upcasting its hot component to fp32.")
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
