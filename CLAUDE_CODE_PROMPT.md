# Build my kaggle-model-server repo

You are working in a folder that will become a GitHub repo called `kaggle-model-server`.
It serves AI models from Kaggle's free tier and exposes them through cloudflared tunnels.
The repo is the single source of truth; thin Kaggle notebooks clone it at session start.

## Existing files — do NOT rewrite these

This folder should already contain three debugged, working files:

- `harness.py` — llama.cpp lifecycle: per-repo binary builds (mainline + forks), cloudflared bootstrap, weight fetching via hf_hub_download, file-based logging, health checks, tunnel. Read it fully before writing anything else — every new module must reuse its patterns (especially `ensure_cloudflared`, `start_tunnel`, `_tail`, the log-to-file rule, and `WORK_DIR = "/kaggle/tmp"`).
- `model_registry.py` — 3 verified LLM entries (bonsai ternary, gemma4-12b, qwen3.6-35b) with a field reference and a VRAM budget cheat sheet in the docstring.
- `run_model.ipynb` — the thin launcher notebook pattern: clone cell, config cell with a single MODEL_KEY, run cell, stop/log-tips markdown.

If any of the three is missing, STOP and ask me to add them first (they come from a Claude chat).
Extend these, match their style (lowercase comments, thin notebooks, all logic in .py modules).
ONE sanctioned modification to harness.py is allowed — Step 1.5 below (per-call setting overrides + quant resolution). Everything else in harness.py stays as-is, and while implementing Step 1.5 you must preserve these invariants exactly (each is a hard-won production fix, not clutter): WORK_DIR=/kaggle/tmp; logs to files never subprocess.PIPE; the cmake flags `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF`; `_exec_copy` for cached binaries; per-repo build dirs for llama.cpp forks; the health check raising with the log tail. `run()`/`stop()` must remain backward compatible — the existing notebook's calls keep working unchanged.

## Target environment — hard constraints for ALL code you write

- Kaggle free notebooks: 2× Tesla T4 (Turing, SM75, ~15GB usable each), 30GB system RAM, 4 vCPUs.
- T4 dtype rules: torch.float16 ONLY. Never torch.bfloat16 (unsupported — silently slow or breaks). No fp8 compute. No flash-attention-2 (needs Ampere+); use `attn_implementation="eager"` or sdpa. For bitsandbytes NF4: `bnb_4bit_compute_dtype=torch.float16`.
- Disk: `/kaggle/working` is quota-capped (~20GB, persists as output). All large downloads, builds, and model files go to `/kaggle/tmp` (~60GB ephemeral). harness.py already does this — follow it.
- Subprocess rule: never `stdout=subprocess.PIPE` without a continuous reader — an unread pipe fills its ~64KB buffer and blocks the child mid-run. Always log to files under /kaggle/tmp like harness.py does.
- Kaggle dataset mounts (`/kaggle/input/...`) drop the executable bit — binaries must be copied out and chmod'd (harness has `_exec_copy`).
- Sessions are ephemeral (max ~12h, disk wiped). Persistence = GitHub for code, Kaggle Datasets for caches. Nothing in the repo may assume prior state.
- Internet is available in-session (user enables it).
- Secrets: read `HF_TOKEN` from the environment only (Kaggle secrets). Never hardcode tokens.

## Non-negotiable rules

1. NEVER invent a Hugging Face filename. Before writing any `hf_file` value, verify it exists by listing the repo:
   ```python
   from huggingface_hub import list_repo_files
   print(sorted(list_repo_files("REPO_ID")))
   ```
   (pip install huggingface_hub first; pass token=os.environ.get("HF_TOKEN") for gated repos.)
   If a repo is gated or unreachable from here, write the entry anyway with the field set to your best candidate and a `# TODO: verify filename on kaggle` comment — clearly marked, never silent.
2. READ THE MODEL CARD before writing any registry entry or loader. For every repo you touch, fetch its card:
   ```python
   from huggingface_hub import hf_hub_download
   print(open(hf_hub_download("REPO_ID", "README.md")).read())
   ```
   and extract the model-specific facts from it: exact companion files (mmproj, VAEs, text encoders, LoRAs, configs), pinned dependency versions, `trust_remote_code` / custom pipeline classes, recommended inference and sampling params, gating and license terms, and any llama.cpp-fork or custom-node requirements. The card is the authority on those facts. Any per-model details written into this prompt (version pins, file lists, step counts) are a July 2026 snapshot — if the card you fetch disagrees, the card is fresher and wins.
3. PRECEDENCE RULE — the one exception to rule 2: where a model card's recipe conflicts with the Target Environment section above, the environment WINS and you adapt the recipe. Cards routinely assume newer GPUs than a T4: they will say `torch.bfloat16` → you write `torch.float16`; `attn_implementation="flash_attention_2"` → `"eager"` (or sdpa); fp8 compute → fp8-as-storage with fp16 compute, or a non-fp8 variant; SGLang/fa3/CUDA-13 serving stacks → the CUDA-12 vLLM or transformers path; "requires RTX 4090" → quantize + offload. Every such adaptation gets a one-line comment stating what the card said and why it was changed (e.g. `# card says bf16; t4 is sm75 -> fp16`), so future sessions can audit the deltas.
4. You cannot run GPU code here. Validation is static only: `python -m py_compile` every .py, `json.load` every .ipynb, and dry-run any pure-python helpers. Design for on-Kaggle debuggability: clear error messages that include log-file paths.
5. Commit only code, notebooks, and docs. Add a `.gitignore` covering `*.gguf`, `*.safetensors`, `*.bin`, `__pycache__/`, `.ipynb_checkpoints/`, `outputs/`.
6. Keep each module under ~300 lines. Prefer fewer, denser helpers over frameworks.

## Step 1 — expand model_registry.py (LLM stack)

Keep the 3 existing entries untouched. Add these, each with verified `hf_file`, sensible `est_vram_gb`, and a one-line comment on what it's for:

Single-T4 tier (`gpu_devices: [0]`, no tensor_split) — file must be ≤ ~12GB:
- `prism-ml/Bonsai-27B-gguf` — the 1-bit companion to the existing ternary entry; same `llama_cpp_repo: "https://github.com/PrismML-Eng/llama.cpp"`.
- `empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF` — pick a ~Q4_K_M quant; keep `ctx: 8192` with a comment that the advertised 1M context is not feasible in this VRAM.
- `empero-ai/Qwythos-9B-v2-GGUF` — ~Q4–Q6 quant.
- `yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF` — Q4 tier, `extra_args: ["--jinja"]`.
- `bartowski/LLaMA-Mesh-GGUF` — Q4_K_M; comment: text-to-3D-mesh LLM.
- `huihui-ai/Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-GGUF` — abliterated twin of the Qwythos entry above; same ctx caveat (default 8192, the 1M is not feasible here).
- `huihui-ai/Huihui-gemma-4-12B-it-qat-q4_0-unquantized-abliterated-GGUF` — abliterated gemma-4-12B QAT; pick the ~Q4 file, `extra_args: ["--jinja"]`.
- `mradermacher/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated-GGUF` — abliterated gemma-4 coder finetune; mradermacher filenames follow `<model>.Q4_K_M.gguf` and there may be a separate `-i1-` imatrix repo — list files, pick a Q4; `--jinja`.

Dual-T4 tier (`tensor_split: "1,1"`, `gpu_devices: [0, 1]`) — file ≤ ~26GB:
- `bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF` — Q4 tier.
- `unsloth/Qwen3.6-27B-MTP-GGUF` — Q4 tier; comment that llama.cpp's MTP support gives a big decode speedup.
- `deepreinforce-ai/Ornith-1.0-35B-GGUF` — Q4 tier.
- `InternScience/Agents-A1-Q4_K_M-GGUF` — the OFFICIAL quant: single file `Agents-A1-Q4_K_M.gguf`, 21.2GB, qwen35moe arch. Never download the bf16 `InternScience/Agents-A1` repo (70GB). Put the card's recommended sampling (temp 0.85, top_p 0.95, top_k 20, presence_penalty 1.1) in the entry comment. Their collection also lists an official 4B released 2026-07-14 — if it has a gguf, add it as a bonus single-T4 entry.
- `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` — this repo IS a GGUF release (imatrix quants incl. custom `Q4_K_P`, which trades ~5-15% extra size for 1-2 quant levels of quality). Pick Q4_K_M or Q4_K_P after checking sizes fit ≤ ~24GB. `extra_args: ["--jinja"]`. Entry notes: HF's hardware-compatibility widget doesn't recognize K_P files, so trust `list_repo_files` over the widget; an optional `mmproj-*-f16.gguf` adds vision; the card asks for ≥128K ctx to preserve thinking mode — that KV budget doesn't exist on this hardware, so default ctx 8192-16384 with a comment that thinking mode is degraded and non-thinking is preferred.
- `huihui-ai/Huihui-Agents-A1-abliterated-GGUF` — abliterated Agents-A1; Q4 tier (~21GB), same sampling-params comment as the official entry.
- `huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF` — abliterated 27B with the MTP drafter; Q4 tier.
- `huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF` — abliteration of lordx64's Claude-4.7-Opus reasoning distill of Qwen3.6-35B-A3B; 36B qwen35moe, multimodal base (grab the mmproj if the repo has one). VERIFIED quirks: filenames use a `...-abliterated-ggml-model-<QUANT>.gguf` infix (another reason list_repo_files beats guessing), and the compatibility widget misses the Q4_K file the card itself references. Default to Q4_K (~21GB, dual-T4); note Q2_K is 13.2GB and squeezes onto ONE t4 as a fallback entry, while Q6_K (29.2GB) is over budget — skip it.

MTP entries (all three): the card's reference invocation enables the drafter with `--spec-type draft-mtp --spec-draft-n-max 6` plus `-fa on` — put those in `extra_args` after cross-checking each card per rule 2. Registry keys for all abliterated variants get an `-abl` suffix (e.g. `agents-a1-abl`) so censored/uncensored twins are unambiguous at a glance, and each entry comment cross-references its base entry.

Sanity-check every entry against the docstring's budget cheat sheet before finalizing.
Add a note to the registry docstring making the philosophy explicit: entries are DEFAULTS — every field can be overridden per-call via `run()` kwargs (Step 1.5), and a proven override gets promoted back into the registry later.

## Step 1.5 — settings overrides + quant switching (the ONE harness.py modification)

Goal: changing quant, context, or any other setting must never require editing the registry or knowing exact gguf filenames. Implement:

1. **Per-call overrides.** `run(model_key, registry, *, port=8080, api_key=None, health_timeout=600, **overrides)` — any registry field (`ctx`, `ngl`, `tensor_split`, `n_cpu_moe`, `gpu_devices`, `extra_args`, `hf_file`, plus the new `quant`) may be passed as a kwarg. Build the effective config as `{**registry[model_key], **overrides}` (only keys actually provided). `start_server` takes the effective config. Print the effective config at launch so the log shows exactly what ran.

2. **Quant resolution by name, not filename.** New kwarg `quant="Q3_K_M"` (or a `quant` field in a registry entry): resolve it at runtime with `huggingface_hub.list_repo_files(repo)` — filter to `.gguf` files whose name contains the quant string (case-insensitive), exclude `mmproj`/vae/encoder files, prefer an exact `-{quant}.gguf` suffix match, and if several remain pick the shortest name. On no match, raise an error that LISTS the quants actually available in the repo (parse the `Q\d_\w+`/`IQ\d_\w+`/`UD-` patterns out of the filenames). When `quant` is absent, use `hf_file` exactly as today — full backward compatibility.

3. **`list_quants(model_key, registry)` helper** — prints every gguf in the repo with its size in GB (use `HfApi().model_info(repo, files_metadata=True)` for sizes). This is both the human-facing "what can I switch to" command and the data source for the UI in Step 4.5.

4. **Soft budget warning.** Before downloading, fetch the chosen file's size; if it exceeds ~12GB on a single-GPU config or ~26GB on a dual config, print a loud warning with the numbers (don't block — the user may know better, e.g. n_cpu_moe setups).

Update `run_model.ipynb`'s config cell to showcase the new UX, e.g.:

```python
MODEL_KEY = "qwen3.6-35b-a3b-hotswap"
url = run(MODEL_KEY, MODELS,
          quant="Q3_K_M",     # try a smaller quant — no filename needed
          ctx=16384)          # more context; omit anything to keep registry defaults
# list_quants(MODEL_KEY, MODELS)   # <- see what quants this repo offers, with sizes
```

## Step 2 — create image_models.py (diffusers stack, no ComfyUI)

A registry + loader for text-to-image on T4s. Public API:

- `IMAGE_MODELS` dict with keys: `"z-image-turbo"` (Tongyi-MAI/Z-Image-Turbo — easiest fit, 8-step), `"krea-2-turbo"` (krea/Krea-2-Turbo — needs `pip install git+https://github.com/huggingface/diffusers.git` for Krea2Pipeline; 8 steps, guidance 0.0), `"flux1-dev"` (black-forest-labs/FLUX.1-dev — comment: non-commercial license), `"ideogram-4"` (ideogram-ai/ideogram-4-nf4 — comment: gated + non-commercial; requires HF_TOKEN and accepting the license on the model page first).
- `install(key)` — pip-installs that model's exact requirements (each key lists its own; keep them minimal).
- `load(key)` — returns a ready pipeline: transformer quantized NF4 via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)` where the pipeline supports component-level quantization, `torch_dtype=torch.float16` everywhere, then `pipe.enable_model_cpu_offload()`. Per-key default step counts / guidance in the registry.
- `generate(pipe, prompt, **overrides)` — runs, saves PNG to `/kaggle/tmp/outputs/` (create it), returns the path, prints elapsed time.

Keep serving out of scope for now — usage is notebook cells. Add a docstring note that a FastAPI wrapper can reuse `harness.start_tunnel` later.

## Step 3 — create comfy_bootstrap.py (video stack)

Headless ComfyUI as a managed backend, mirroring harness.py's lifecycle style:

- `install()` — clone ComfyUI, `city96/ComfyUI-GGUF`, and `kijai/ComfyUI-KJNodes` into `/kaggle/tmp/ComfyUI`, pip install each requirements.txt. Idempotent (skip if present).
- `fetch_stack(key)` — downloads a named model set via `hf_hub_download` into /kaggle/tmp and symlinks files into the right `ComfyUI/models/<subdir>/`. Implement these keys with these exact files (verified from the model cards — still re-verify with list_repo_files):
  - `"ltx-2.3"`: from `unsloth/LTX-2.3-GGUF`: a dev unet gguf (default `ltx-2.3-22b-dev-Q3_K_M.gguf` for T4 headroom; also allow Q4_0) → `unet/`; `vae/ltx-2.3-22b-dev_video_vae.safetensors` and `vae/ltx-2.3-22b-dev_audio_vae.safetensors` → `vae/`; `text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors` → `text_encoders/`. From `Lightricks/LTX-2.3`: `ltx-2.3-22b-distilled-lora-384.safetensors` → `loras/`; `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` → `latent_upscale_models/`. From `unsloth/gemma-3-12b-it-qat-GGUF`: `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` and `mmproj-BF16.gguf` → `text_encoders/`.
  - `"scail-2"`: from `realrebelai/SCAIL-2_GGUF`: the Q4_K_M unet gguf → `unet/`. From `Kijai/WanVideo_comfy`: `umt5-xxl-enc-fp8_e4m3fn.safetensors` → `text_encoders/`. From `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v`: `loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors` → `loras/`. From `Comfy-Org/sam3.1`: `checkpoints/sam3.1_multiplex_fp16.safetensors` → `sam/`. From `Comfy-Org/Wan_2.1_ComfyUI_repackaged`: `split_files/clip_vision/clip_vision_h.safetensors` → `clip_vision/` and `split_files/vae/wan_2.1_vae.safetensors` → `vae/`.
  - `"lingbot-30b"` (mark EXPERIMENTAL in comments): clone `https://github.com/RealRebelAI/ComfyUI_Rebels_LingBot` into custom_nodes; download the Q3_K_M gguf from `realrebelai/LingBot-30B-3B_GGUF_ComfyUI` (Q3 fits 30GB RAM comfortably); encoder + `LingBot_vae.safetensors` from `realrebelai/LingBot_ComfyUI`; the 30B `transformer/config.json` into the node pack's `model_assets/transformer_config_30b.json`.
- `start(port=8188)` — launch `python main.py --listen 0.0.0.0 --port <port> --force-fp16` with stdout/stderr to `/kaggle/tmp/comfyui.log` (never PIPE), poll `http://127.0.0.1:<port>/` until up (timeout ~180s, raise with log tail on failure), then `from harness import start_tunnel` and return the public URL for that port.
- `queue_workflow(workflow: dict, timeout=3600)` — POST to `/prompt`, poll `/history/<prompt_id>` until done, return output file paths. Include one tiny doc example of loading a workflow JSON exported from the ComfyUI GUI.
- `stop()` — terminate the ComfyUI process cleanly (track it in a module-level dict like harness `_current`).

## Step 4 — create tasks.py (small in-notebook models)

Lazy-import helpers, each ≤ ~40 lines, each accepting `gpu: int = 1` and setting `CUDA_VISIBLE_DEVICES` before touching torch so they can share the box with a running LLM on GPU 0:

- `ocr_pages(images)` — `ATH-MaaS/OvisOCR2` via vLLM (its card pins `vllm==0.22.1` and `gdn_prefill_backend="triton"`); page images → markdown list.
- `parse_pdf(path)` — `baidu/Unlimited-OCR` via transformers (`trust_remote_code=True`, `torch_dtype=torch.float16`), using its `infer_multi` for multi-page.
- `transcribe(audio_path)` — `OpenMOSS-Team/MOSS-Transcribe-Diarize` via vLLM (CUDA-12 path per its card); returns speaker-labelled transcript.
- `embed(texts)` — `nvidia/Nemotron-3-Embed-1B-BF16`, loaded fp16.
- `tts(text, out_path)` — `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, fp16.

Every helper: clear ImportError message naming the pip install needed; comment the T4 rules (fp16, eager attention) where relevant.

## Step 4.5 — create control_panel.py (Gradio settings UI)

A point-and-click alternative to editing the config cell, for the LLM stack. Build with Gradio Blocks (`pip install gradio`), ~80–120 lines:

- Controls: Model dropdown (keys of MODELS); Quant dropdown that repopulates from `list_quants` when the model changes (show "name — size GB" labels); ctx number input and n_cpu_moe input pre-filled from the registry entry; optional api_key textbox.
- Buttons: **Launch** — calls `run(model_key, MODELS, quant=..., ctx=..., ...)` in a background thread (a launch takes minutes: build/download/load — never block the click handler); **Stop** — calls `stop()`; **Refresh status** — shows current state plus the last ~40 lines of the server log via harness `_tail(SERVER_LOG)`. When the model is up, display its tunnel URL prominently.
- Serving: `launch_panel(auth=None)` runs `demo.launch(share=True, auth=auth, server_port=7860)`. Gradio's share link is the panel's own public URL (independent of the model's cloudflared URL — both coexist fine). Strongly recommend `auth=("user", "pass")` in the docstring since share links are public.
- Keep it honest and simple: no websockets, no auto-polling — a manual Refresh button is fine for v1.

## Step 5 — launcher notebooks

Create `run_image.ipynb` and `run_video.ipynb` cloned from `run_model.ipynb`'s exact structure (markdown intro → setup/clone cell with `rm -rf` guard → config cell with one KEY variable → run cell → stop/log-tips markdown). run_image: pick IMAGE_KEY, install+load+generate in cells. run_video: install() → fetch_stack(VIDEO_KEY) → start() → print URL → queue_workflow example. In run_model.ipynb, add one optional cell after the config cell: "prefer clicking to editing? run this instead:" → `from control_panel import launch_panel; launch_panel(auth=("joy", "CHANGE-ME"))`. Ask me for my GitHub username/repo name and fill the real clone URL into ALL three notebooks (including the existing run_model.ipynb — that edit is allowed).

## Step 6 — README.md

Write it for future-me on a phone: what this is (one paragraph), Kaggle quickstart (accelerator T4×2, internet on, import notebook, Run All, copy tunnel URL, `stop()` + end session), the three stacks and when to open which notebook, the cache-Dataset recipe (save `llama-server-<slug>` binaries, `cloudflared`, and hot ggufs to a Dataset; attach it; set `CACHE_DATASET_DIR` in harness.py — cuts cold boot from ~15–20 min to ~2), security (`run(..., api_key=...)` because trycloudflare URLs are public), a "changing settings" section (per-call overrides + `quant=` + `list_quants` — registry edits only needed to change defaults), a "UIs you get for free" section (1. llama-server ships a built-in chat web UI at the tunnel URL root — open it in any browser; 2. the Gradio control panel via `launch_panel`, with the auth warning; 3. ComfyUI's full node GUI at the video-stack tunnel URL), troubleshooting (the exact log paths), and a license/caveat table: FLUX.1-dev and Ideogram 4 non-commercial, Ideogram gated (HF_TOKEN), Bonsai needs the PrismML fork and is unverified on T4 kernels, and a note that NSFW generation violates Kaggle ToS.

## Step 7 — validate, commit, publish

1. `python -m py_compile` every .py; `python -c "import json; json.load(open(f))"` every .ipynb.
2. Run your filename-verification script one final time across every `hf_repo`/`hf_file` in model_registry.py, image_models.py, and comfy_bootstrap.py; print a table of verified vs TODO.
3. `git init`, add `.gitignore`, commit as "initial scaffold: llm + image + video + tasks stacks".
4. If the `gh` CLI is installed and authenticated: `gh repo create kaggle-model-server --public --source=. --push`. Otherwise print the exact manual commands for me to push.
5. Finish with: the final file tree, the verified/TODO table, and a short "first session checklist" of what to test on Kaggle in order (1. run_model with gemma4-12b — mainline path; 2. bonsai — fork + T4 kernel smoke test; 3. a dual-GPU model; 4. run_image with z-image-turbo; 5. run_video with ltx-2.3).

Work through the steps in order. Ask me only when genuinely blocked (missing base files, GitHub name, gated-repo confirmation) — otherwise proceed and mark uncertainties with clear TODO comments.
