# kaggle-model-server

Serve real AI models from Kaggle's free tier (2× Tesla T4) and reach them from anywhere through cloudflared tunnels. This repo is the single source of truth: thin Kaggle notebooks clone it at session start, all logic lives in the `.py` modules, and nothing assumes a previous session ever existed. Three stacks: **LLMs** via llama.cpp (`harness.py` + `model_registry.py`), **text-to-image** via diffusers (`image_models.py`), **video** via headless ComfyUI (`comfy_bootstrap.py`), plus small side-task helpers (`tasks.py`).

## Kaggle quickstart

1. New notebook → Accelerator **GPU T4 ×2** → Settings → **Internet ON**.
2. File → Import notebook → pick `run_studio.ipynb` (the UI) or `run_model.ipynb` (code cells) from this repo.
3. **Run All**. First boot builds/downloads for ~15–20 min (see cache recipe below to cut that to ~2).
4. Copy the printed `https://…trycloudflare.com` URL — that's your server, from any device.
5. When done: run `stop()` (or `comfy.stop()`), then end the session so you don't burn GPU quota.

## Which notebook when

| Notebook | Stack | Open it when you want |
|---|---|---|
| `run_studio.ipynb` | all three | the point-and-click studio: launch/chat/image/video tabs from one link |
| `run_model.ipynb` | llama.cpp LLMs | the gradio-free code path: an OpenAI-compatible API + built-in chat UI |

Image and video from code (no studio): `from image_models import install, load, generate` and `import comfy_bootstrap as comfy; comfy.install(); comfy.fetch_stack("ltx-2.3"); comfy.start()` in any notebook that clones this repo — the module docstrings carry the full recipes.

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

Models that aren't in the registry at all: `detect_entry("author/AnyModel-GGUF")` builds an entry automatically (best-fitting quant + single/dual-GPU placement from real file sizes) — same thing the studio's "import from hugging face" box does, which also prints a paste-ready registry line.

## UIs you get for free

1. **llama-server chat UI** — the tunnel URL root serves a full chat web UI in any browser; the API lives at `<url>/v1`.
2. **Gradio studio** — `run_studio.ipynb` (or `from control_panel import launch_panel; launch_panel(auth=("user","pass"))`): four tabs — Launch (llm + quant picker with sizes, launch/stop/status with per-GPU VRAM, cache harvest), Chat (streams from the running model, system-prompt box, card-recommended sampling applied automatically), Image (diffusers spread across both T4s, prompt → png in the browser), Video (boots ComfyUI, shows its GUI url, ships stack workflow JSONs into its workflow browser). Every tab imports new models straight from HF. The share link is public — **always set auth** (the studio notebook generates a random password and prints it).
3. **ComfyUI node GUI** — the video stack's tunnel URL serves the entire ComfyUI editor; build workflows in the browser, or export API-format JSON and drive it with `comfy.queue_workflow(...)`.

## Security

`trycloudflare.com` URLs are **publicly reachable by anyone who has them**. For LLMs always pass `run(..., api_key="something")` — the server then requires `Authorization: Bearer something`. For the Gradio panel always pass `auth=`. ComfyUI has no auth: treat that URL as a secret and stop the tunnel when done.

## Cache Dataset: cold boot ~15–20 min → ~2 min

One-time, at the end of a session that built things:

1. Run `harvest_cache()` in a cell (or press **Harvest cache** in the studio) — it stages the built binaries, cloudflared, and this session's ggufs into `/kaggle/working` with the right names, watching the ~19.5GB quota.
2. **Save Version** → notebook **Output** tab → **New Dataset**.
3. Next session: **Add Input** → attach the dataset. That's it — the harness **auto-discovers** any attached dataset containing `llama-server-*`/`cloudflared`; `CACHE_DATASET_DIR` exists only to pin one when several are attached. Binaries are copied out and re-chmodded (dataset mounts drop the executable bit), cached ggufs are used directly.

## Stable URL (optional, recommended)

Quick tunnels mint a new `trycloudflare.com` URL every session, so anything you point at the API breaks daily. Free fix: Cloudflare dashboard → Zero Trust → Networks → Tunnels → create one, set its public hostname to `http://localhost:8080`, then add two Kaggle secrets: `CF_TUNNEL_TOKEN` (the tunnel token — required) and `CF_TUNNEL_HOSTNAME` (your hostname, so printed URLs are real). Export them like `HF_TOKEN` and the harness runs your named tunnel instead — **same URL every session**. Non-8080 ports (ComfyUI) still get quick tunnels.

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
| Ideogram 4 (incl. fal instant/fast) | **Non-commercial** + **gated** — accept license on the model page, needs `HF_TOKEN` (Kaggle secret) |
| stable-fast-3d / stable-point-aware-3d | Stability **community license** + gated — accept on the model pages |
| ltx-10eros / sulphur-2 stacks | Community finetunes aimed at explicit content — NSFW output violates Kaggle ToS |
| Bonsai 27B (ternary + 1-bit) | Needs the PrismML llama.cpp fork (auto-built); published perf is H100 — **unverified on T4 kernels**, treat first boot as a smoke test |
| Abliterated / uncensored entries (`-abl`) | Reduced refusals — review outputs; you are responsible for what you generate |
| Everything | **NSFW generation violates Kaggle ToS** — don't; it risks your account |

`HF_TOKEN` comes from Kaggle **Secrets** (Add-ons → Secrets) — never hardcode tokens in notebooks.
