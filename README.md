# kaggle-model-server

Serve real AI models from Kaggle's free tier (2× Tesla T4) and reach them from anywhere through cloudflared tunnels. This repo is the single source of truth: thin Kaggle notebooks clone it at session start, all logic lives in the `.py` modules, and nothing assumes a previous session ever existed. Three stacks: **LLMs** via llama.cpp (`harness.py` + `model_registry.py`), **text-to-image** via diffusers (`image_models.py`), **video** via headless ComfyUI (`comfy_bootstrap.py`), plus small side-task helpers (`tasks.py`).

## Kaggle quickstart

1. New notebook → Accelerator **GPU T4 ×2** → Settings → **Internet ON**.
2. File → Import notebook → pick `run_model.ipynb` (or `run_image` / `run_video`) from this repo.
3. **Run All**. First boot builds/downloads for ~15–20 min (see cache recipe below to cut that to ~2).
4. Copy the printed `https://…trycloudflare.com` URL — that's your server, from any device.
5. When done: run `stop()` (or `comfy.stop()`), then end the session so you don't burn GPU quota.

## Which notebook when

| Notebook | Stack | Open it when you want |
|---|---|---|
| `run_model.ipynb` | llama.cpp LLMs | an OpenAI-compatible chat/completions API + built-in chat UI |
| `run_image.ipynb` | diffusers txt2img | PNGs from prompts in notebook cells (z-image / krea / flux / ideogram) |
| `run_video.ipynb` | headless ComfyUI | video generation (ltx-2.3, scail-2, lingbot) with the full node GUI |

## Changing settings (no registry edits needed)

Registry entries are **defaults**. Override anything per-call:

```python
run("qwen3.6-35b-a3b-hotswap", MODELS,
    quant="Q3_K_M",   # switch quant BY NAME -- resolved to the real filename for you
    ctx=16384,        # any registry field works: ngl, tensor_split, n_cpu_moe, extra_args...
    api_key="secret")
list_quants("qwen3.6-35b-a3b-hotswap", MODELS)  # every gguf in the repo, with sizes
```

Unknown quant names fail fast **listing what the repo actually offers**, and a size check warns before you download something that won't fit (12GB single-T4 / 26GB dual). Edit `model_registry.py` only to change a *default*; a proven override deserves promotion into the registry.

## UIs you get for free

1. **llama-server chat UI** — the tunnel URL root serves a full chat web UI in any browser; the API lives at `<url>/v1`.
2. **Gradio control panel** — `from control_panel import launch_panel; launch_panel(auth=("user","pass"))`: dropdown model + quant picker (with sizes), launch/stop/status buttons. The share link is public — **always set auth**.
3. **ComfyUI node GUI** — the video stack's tunnel URL serves the entire ComfyUI editor; build workflows in the browser, or export API-format JSON and drive it with `comfy.queue_workflow(...)`.

## Security

`trycloudflare.com` URLs are **publicly reachable by anyone who has them**. For LLMs always pass `run(..., api_key="something")` — the server then requires `Authorization: Bearer something`. For the Gradio panel always pass `auth=`. ComfyUI has no auth: treat that URL as a secret and stop the tunnel when done.

## Cache Dataset: cold boot ~15–20 min → ~2 min

One-time, at the end of a session that built things:

1. Save these files from `/kaggle/tmp` into a private Kaggle Dataset (e.g. `llm-inference-cache`):
   - `llama.cpp-<slug>/build/bin/llama-server` → name it `llama-server-<slug>` (per llama.cpp repo: mainline + each fork get their own; the build prints the exact tip)
   - `cloudflared`
   - the gguf files you actually use (hot models only — Datasets cap at ~100GB total)
2. Next session: **Add Input** → attach the dataset, and set in `harness.py`:
   `CACHE_DATASET_DIR = "/kaggle/input/llm-inference-cache"`
3. The harness then copies binaries out (dataset mounts drop the executable bit — handled) and uses cached ggufs directly. No rebuilds, no re-downloads.

## Troubleshooting

Every process logs to a file — read the tail, the answer is usually right there:

| Component | Log |
|---|---|
| llama-server | `/kaggle/tmp/llama-server.log` |
| cloudflared tunnel | `/kaggle/tmp/cloudflared.log` |
| ComfyUI | `/kaggle/tmp/comfyui.log` |
| transcribe()'s vllm | `/kaggle/tmp/vllm-transcribe.log` |

`!tail -50 /kaggle/tmp/llama-server.log` in a cell. Failed health checks and crashed launches already include the tail in the raised error. Out of disk? Everything big must be under `/kaggle/tmp` (~60GB ephemeral), never `/kaggle/working` (~20GB, persists).

## Licenses & caveats

| Model | Caveat |
|---|---|
| FLUX.1-dev | **Non-commercial** license; gated — accept on the model page, needs `HF_TOKEN` |
| Ideogram 4 | **Non-commercial** + **gated** — accept license on the model page, needs `HF_TOKEN` (Kaggle secret) |
| Bonsai 27B (ternary + 1-bit) | Needs the PrismML llama.cpp fork (auto-built); published perf is H100 — **unverified on T4 kernels**, treat first boot as a smoke test |
| Abliterated / uncensored entries (`-abl`) | Reduced refusals — review outputs; you are responsible for what you generate |
| Everything | **NSFW generation violates Kaggle ToS** — don't; it risks your account |

`HF_TOKEN` comes from Kaggle **Secrets** (Add-ons → Secrets) — never hardcode tokens in notebooks.
