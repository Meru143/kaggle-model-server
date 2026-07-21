"""browser studio for the whole box: launch/stop llms, switch quants, chat,
generate images, and boot comfyui -- one gradio app, one public link.

    from control_panel import launch_panel
    launch_panel(auth=("user", "a-real-password"))

needs: pip install gradio. the gradio share link is the panel's own public
url -- independent of the model's cloudflared url, both coexist fine. share
links are PUBLIC AND GUESSABLE: always pass auth=("user", "pass"), anyone
with the link can otherwise start/stop your models (run_studio.ipynb
generates a random password and prints it).

design: terminal-native control room -- dark slate, jetbrains mono as the
single family, one green run-light that carries state only (live/launching/
failed), statuses rendered as an led + console log tail. no decorative
motion; 150ms state feedback, reduced-motion respected.
"""

import html
import inspect
import json
import os
import re
import subprocess
import threading
import time
import traceback

import requests

import comfy_bootstrap as comfy
import image_models
import sdcpp
from harness import (SERVER_LOG, _fmt_eta, _tail, detect_entry, list_quants,
                     progress_line, run, set_progress, stop)
from image_models import IMAGE_MODELS
from model_registry import MODELS

_PORT = 8080  # panel always launches on harness's default port
STUDIO_LOG = "/kaggle/tmp/studio.log"  # full tracebacks from every studio action
# "gen" is a launch-generation counter: Stop (or a new launch) bumps it so an
# in-flight background thread knows it's been cancelled and must not commit its
# result -- lets Stop unlock the UI even while a launch is mid-flight
_state = {"busy": False, "url": None, "error": None, "model": None,
          "api_key": None, "gen": 0}
_img_state = {"busy": False, "pipe": None, "error": None, "model": None,
              "gen_busy": False, "gen_error": None, "last_image": None,
              "backend": None,  # "comfy" (headless) or "diffusers" (in-process)
              "gen_t0": None, "gen_secs": None,  # live timer + final duration
              "gen_warn": None}  # e.g. black frame -- saved, but not usable
_vid_state = {"busy": False, "url": None, "error": None, "stack": None, "gen": 0}


def _log_tb(context):
    """append the current traceback to the studio log; return a short head"""
    tb = traceback.format_exc()
    try:
        with open(STUDIO_LOG, "a") as f:
            f.write(f"\n=== {time.strftime('%H:%M:%S')} {context}\n{tb}")
    except OSError:
        pass  # off-kaggle
    return tb


# ---- statusline console (the one signature element) ----------------------

def _gpu_note():
    """one-line per-gpu vram readout, or None off-gpu boxes"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return " · ".join(
            f"gpu{i.strip()} {int(u) / 1024:.1f}/{int(t) / 1024:.0f}GB"
            for i, u, t in (l.split(",") for l in out.stdout.strip().splitlines()))
    except Exception:
        return None


def _console(led, head, log_path=None, link=None, note=None, err=None, gpu=False,
             prog=False):
    """led + one-line status + optional progress/link/note/vram/log tail, as safe html"""
    parts = [f'<span class="km-led {led}">&#9679;</span> '
             f'<span class="km-head">{html.escape(head)}</span>']
    if prog and (line := progress_line()):
        parts.append(f'<div class="km-prog">{html.escape(line)}</div>')
    if link:
        safe = html.escape(link, quote=True)
        parts.append(f'<div class="km-note"><a href="{safe}" target="_blank">{safe}</a></div>')
    if note:
        parts.append(f'<div class="km-note">{html.escape(note)}</div>')
    if gpu and (vram := _gpu_note()):
        parts.append(f'<div class="km-note">{html.escape(vram)}</div>')
    if err:
        parts.append(f'<pre class="km-err">{html.escape(err)}</pre>')
    if log_path:
        parts.append(f'<pre>{html.escape(_tail(log_path, 40))}</pre>')
    return f'<div class="km-console-inner">{"".join(parts)}</div>'


# ---- llm launch ----------------------------------------------------------

def _set_hf_token(tok):
    """stash the token in the env (everything downstream reads HF_TOKEN at
    call time) and validate it with a whoami. never echo the token back."""
    import gradio as gr
    tok = (tok or "").strip()
    if not tok:
        os.environ.pop("HF_TOKEN", None)
        return gr.update(value=""), "token cleared"
    try:
        from huggingface_hub import HfApi
        user = HfApi(token=tok).whoami()["name"]
    except Exception as e:
        # don't store a rejected token: it would 401 every later download that
        # reads HF_TOKEN -- even ungated ones. leave the env untouched.
        return gr.update(value=""), \
            f"token rejected by hf ({type(e).__name__}) — not stored, double-check it"
    os.environ["HF_TOKEN"] = tok  # only keep a token that actually works
    msg = (f"token set — authenticated as {user}. gated models still need "
           f"their license accepted on the model page (one click, once).")
    return gr.update(value=""), msg


def _on_model_change(model_key):
    import gradio as gr
    entry = MODELS[model_key]
    try:
        rows = list_quants(model_key, MODELS)
        choices = [(f"{name} — {gb:.1f} GB", name) for name, gb in rows]
    except Exception as e:  # offline / rate-limited: registry default still works
        choices = []
        print(f"list_quants failed: {e}")
    return (gr.update(choices=choices, value=None),
            gr.update(value=entry.get("ctx", 8192)),
            gr.update(value=entry.get("n_cpu_moe")))


def _launch(model_key, gguf_file, ctx, n_cpu_moe, api_key):
    if _state["busy"]:
        return _console("busy", "a launch is already running — press Stop to cancel it first")
    overrides = {}
    if gguf_file:
        overrides["hf_file"] = gguf_file  # exact file picked from the quant list
    if ctx:
        overrides["ctx"] = int(ctx)
    # gradio 6 renders an empty Number as 0 -- treat 0 as "not set" (a real
    # n_cpu_moe of 0 is meaningless anyway)
    if n_cpu_moe not in (None, "", 0):
        overrides["n_cpu_moe"] = int(n_cpu_moe)

    _state["gen"] += 1
    my_gen = _state["gen"]

    def work():
        _state.update(busy=True, url=None, error=None,
                      model=model_key, api_key=api_key or None)
        try:
            url = run(model_key, MODELS, api_key=api_key or None, **overrides)
            if _state["gen"] == my_gen:
                _state["url"] = url
            else:
                stop()  # cancelled mid-flight -- tear down what this launch built
        except Exception as e:
            if _state["gen"] == my_gen:
                _log_tb(f"launch {model_key}")
                _state["error"] = f"{type(e).__name__}: {e}"
        finally:
            # only the current-generation launch owns the shared state; a
            # cancelled one must not flip busy back on / clobber a newer launch
            if _state["gen"] == my_gen:
                _state["busy"] = False
                set_progress("idle")

    # never block the click handler -- a cold launch takes minutes
    threading.Thread(target=work, daemon=True).start()
    return _console("busy", f"launching {model_key}", prog=True, note="press Stop to cancel")


def _stop():
    # bump gen so any in-flight launch thread is invalidated, unlock the UI
    # immediately, THEN kill procs (which also unblocks a stuck health-wait --
    # proc death makes _wait_for_health return at once)
    _state["gen"] += 1
    _state.update(busy=False, url=None, error=None, model=None)
    set_progress("idle")
    stop()
    return _console("idle", "stopped — nothing running")


def _status():
    if _state["busy"]:
        return _console("busy", f"launching {_state['model']}",
                        prog=True, log_path=SERVER_LOG, gpu=True)
    if _state["error"]:
        return _console("err", "launch failed", err=_state["error"], log_path=SERVER_LOG)
    if _state["url"]:
        return _console("live", f"{_state['model']} live",
                        link=_state["url"],
                        note=f"chat in the Chat tab · openai api at {_state['url']}/v1",
                        log_path=SERVER_LOG, gpu=True)
    return _console("idle", "nothing running — pick a model and press Launch", gpu=True)


# timer ticks: render while the phase is active + exactly one final render
# when it clears (to show the result), then gr.skip() forever after -- so an
# idle/running studio never re-renders on its own
_tick_active = {"launch": False, "img": False, "vid": False}


def _tick_launch():
    import gradio as gr
    busy = _state["busy"]
    if busy or _tick_active["launch"]:
        _tick_active["launch"] = busy
        return _status()
    return gr.skip()


def _tick_img():
    import gradio as gr
    active = _img_state["busy"] or _img_state["gen_busy"]
    if active or _tick_active["img"]:
        _tick_active["img"] = active
        return _img_refresh()
    return gr.skip(), gr.skip()


def _tick_vid():
    import gradio as gr
    busy = _vid_state["busy"]
    if busy or _tick_active["vid"]:
        _tick_active["vid"] = busy
        return _vid_status()
    return gr.skip()


def _harvest():
    try:
        from harness import harvest_cache
        staged = harvest_cache()
    except Exception as e:
        return _console("err", "cache harvest failed", err=f"{type(e).__name__}: {e}")
    if not staged:
        return _console("idle", "nothing cacheable found (build/download something first)")
    return _console("live", f"staged {len(staged)} files in /kaggle/working",
                    note="Save Version → notebook Output tab → New Dataset. attach it to "
                         "future notebooks; the harness auto-discovers it — boots drop to ~2 min")


def _import_model(repo):
    """add any hf gguf repo to the session's registry with auto-detected
    settings, and point the launch controls at it"""
    import gradio as gr
    repo = (repo or "").strip().strip("/")
    if repo.count("/") != 1:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                'expected "author/repo-name" (e.g. bartowski/SomeModel-GGUF)')
    try:
        entry = detect_entry(repo, read_card=True)  # also scans the card for flags
    except Exception as e:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                f"import failed: {type(e).__name__}: {e}")
    key = repo  # keys are the full hf repo id (author/name), like the registry
    MODELS[key] = entry
    quant_update, ctx_update, moe_update = _on_model_change(key)
    gpus = "both t4s" if entry["tensor_split"] else "one t4"
    card_bits = []
    if entry.get("sampling"):
        card_bits.append("sampling " + ", ".join(f"{k}={v}" for k, v in entry["sampling"].items()))
    if entry.get("extra_args", ["--jinja"]) != ["--jinja"]:
        card_bits.append("flags " + " ".join(entry.get("extra_args", [])))
    card_line = ("from card (verify): " + " · ".join(card_bits) + "\n") if card_bits else \
                "no recipe found in the card — using --jinja + default sampling\n"
    msg = (f"added {key!r}: {entry['hf_file']} ({entry['est_vram_gb'] - 2:.1f}GB -> {gpus})\n"
           f"{card_line}"
           f"to keep it permanently, paste into model_registry.py:\n"
           f'    "{key}": {json.dumps(entry)},')
    return (gr.update(choices=sorted(MODELS), value=key),
            quant_update, ctx_update, moe_update, msg)


# ---- chat ----------------------------------------------------------------

def _to_text(content):
    """gradio content may be a string or a list of content blocks"""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content if not (isinstance(b, dict) and "file" in b))
    return str(content)


def _normalize_history(history):
    """gradio's chat history shape varies across versions (message dicts,
    content-block lists, legacy [user, assistant] pairs). accept them all."""
    msgs = []
    for m in history:
        if isinstance(m, dict):
            msgs.append({"role": m.get("role", "user"),
                         "content": _to_text(m.get("content", ""))})
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            for role, part in (("user", m[0]), ("assistant", m[1])):
                if part:
                    msgs.append({"role": role, "content": _to_text(part)})
        # unknown shapes: skip rather than crash mid-chat
    return msgs


def _chat(message, history, system_prompt=""):
    """streams from the local llama-server openai endpoint, applying the
    registry entry's card-recommended sampling automatically"""
    if _state["busy"]:
        yield "model is still launching -- check Refresh in the Launch tab"
        return
    msgs = ([{"role": "system", "content": system_prompt.strip()}]
            if (system_prompt or "").strip() else [])
    msgs += _normalize_history(history)
    msgs.append({"role": "user", "content": _to_text(message)})
    headers = ({"Authorization": f"Bearer {_state['api_key']}"}
               if _state["api_key"] else {})
    payload = {"model": _state["model"] or "local", "messages": msgs, "stream": True}
    payload.update(MODELS.get(_state["model"], {}).get("sampling") or {})
    try:
        r = requests.post(
            f"http://127.0.0.1:{_PORT}/v1/chat/completions",
            json=payload, headers=headers, stream=True, timeout=600)
        r.raise_for_status()
        acc = ""
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:].decode("utf-8")
            if data == "[DONE]":
                break
            try:
                choices = json.loads(data).get("choices") or [{}]
            except json.JSONDecodeError:
                continue  # skip a malformed/partial sse chunk
            acc += choices[0].get("delta", {}).get("content") or ""
            if acc:
                yield acc
    except requests.exceptions.RequestException as e:
        yield (f"no model answering on port {_PORT} ({type(e).__name__}) -- "
               "launch one in the Launch tab first")


# ---- image ---------------------------------------------------------------

def _llm_running():
    """is a llama-server actually alive? (ground truth, not just UI state)"""
    from harness import _current
    s = _current.get("server")
    return s is not None and s.poll() is None


def _img_repo_label(key):
    """the hf repo (author/name) that best identifies a diffusers entry --
    transformer_from when a variant is defined by it (fal instant/fast), else
    the base pipeline repo. keeps every entry's label unique and recognizable."""
    e = IMAGE_MODELS[key]
    return e.get("transformer_from") or e["hf_repo"]


def _img_choices():
    """image dropdown: comfy stacks (headless, reliable) then diffusers models,
    each LABELED by its hf repo (author/name); the value stays the short key."""
    return ([(f"{comfy.stack_repo(k)}  ·  comfy ✓ reliable", k) for k in comfy.image_stacks()] +
            [(f"{k}  ·  sd.cpp", k) for k in sorted(sdcpp.SD_MODELS)] +
            [(f"{_img_repo_label(k)}  ·  diffusers (experimental)", k) for k in sorted(IMAGE_MODELS)])


def _import_image_model(repo):
    """add any diffusers-format hf repo to the image dropdown"""
    import gradio as gr
    repo = (repo or "").strip().strip("/")
    if repo.count("/") != 1:
        return gr.update(), 'expected "author/repo-name" (e.g. Tongyi-MAI/Z-Image-Turbo)'
    try:
        entry = image_models.detect_image_entry(repo)
    except Exception as e:
        return gr.update(), f"import failed: {type(e).__name__}: {e}"
    key = re.sub(r"[^a-z0-9.]+", "-", repo.split("/")[-1].lower()).strip("-")
    IMAGE_MODELS[key] = entry
    notes = ["transformer nf4-quantized for the t4" if entry["quantize"]
             else "small enough to load fp16 as-is"]
    if entry["gated"]:
        notes.append("GATED: needs an HF_TOKEN secret + accepted license on the model page")
    msg = (f"added {key!r} ({'; '.join(notes)})\n"
           f"steps/guidance use the pipeline defaults -- check the model card and use "
           f"the steps box to override. to keep it, paste into image_models.py:\n"
           f'    "{key}": {json.dumps(entry)},')
    return gr.update(choices=_img_choices(), value=key), msg


def _on_img_model_change(key):
    """list the picked stack's gguf quants, like the launch tab does for llms.
    empty for diffusers models and for stacks whose denoiser isn't gguf."""
    import gradio as gr
    if key not in comfy.IMAGE_STACKS and key not in sdcpp.SD_MODELS:
        return gr.update(choices=[], value=None,
                         label="quant — diffusers models have no gguf variants")
    try:
        rows = (sdcpp.list_quants(key) if key in sdcpp.SD_MODELS
                else comfy.list_stack_quants(key))
    except Exception as e:
        print(f"quant lookup for {key} failed: {e}")
        rows = []
    return gr.update(
        choices=[(f"{fn.rsplit('/', 1)[-1]} — {gb:.1f} GB", fn) for fn, gb in rows],
        value=None,
        label=("quant — blank = stack default" if rows else
               "quant — this stack's denoiser isn't gguf, nothing to pick"))


def _list_img_loras(key, repo):
    """populate the lora picker from any repo id (blank = the stack's own)"""
    import gradio as gr
    if key in sdcpp.SD_MODELS:
        # sd.cpp takes loras as <lora:name:weight> inside the prompt plus
        # --lora-model-dir, not as a loader node -- different mechanism, not wired
        return gr.update(choices=[], value=None,
                         label="lora — not wired for the sd.cpp backend yet")
    if key not in comfy.IMAGE_STACKS:
        return gr.update(choices=[], value=None,
                         label="lora — comfy stacks only")
    try:
        rows = comfy.list_stack_loras(key, (repo or "").strip() or None)
    except Exception as e:
        return gr.update(choices=[], value=None, label=f"lora — lookup failed: {e}")
    return gr.update(
        choices=[(f"{f.rsplit('/', 1)[-1]} — {gb:.2f} GB", f) for f, gb in rows],
        value=None,
        label=f"lora — {len(rows)} found" if rows else "lora — none in that repo")


def _img_setup(key, quant=None, lora=None, lora_repo=None, lora_strength=1.0):
    if _img_state["busy"]:
        return _console("busy", "already installing — press Refresh for progress")
    if _llm_running():
        # an llm on gpu 0 + a heavy image job = OOM. one heavy gpu job at a time.
        return _console("err", f"an LLM ({_state['model']}) is running on gpu 0",
                        note="press Stop in the Launch tab first — image models want most "
                             "of the box's vram and will OOM sharing the gpu with an llm")

    if key in sdcpp.SD_MODELS:
        # models comfy's gguf loader can't read (ideogram4/flux2 archs). sd.cpp
        # runs them natively -- one cli call, no node graph.
        def work():
            _img_state.update(busy=True, error=None, pipe=None, model=key, backend="sdcpp")
            try:
                # comfy keeps its models resident, so a stack loaded earlier is
                # still holding ~11GB of gpu0 -- sd.cpp then ooms allocating its
                # own weights. only one image backend gets the gpus at a time.
                comfy.stop()
                sdcpp.install()          # ~10-20 min the first time (cuda build)
                sdcpp.fetch(key, quants={"diffusion": quant} if quant else None)
                _img_state["pipe"] = ("sdcpp", key)
            except Exception as e:
                _log_tb(f"sdcpp setup {key}")
                _img_state["error"] = f"{type(e).__name__}: {e}"
            finally:
                _img_state["busy"] = False
                set_progress("idle")
        threading.Thread(target=work, daemon=True).start()
        return _console("busy", f"building sd.cpp + fetching {key}", prog=True,
                        note="first run compiles sd-cli with cuda (10-20 min); "
                             "after that it's just the download")

    if key in comfy.IMAGE_STACKS:
        # the reliable path: boot comfy HEADLESS and drive it over http from the
        # Generate button -- no node gui needed. comfy runs one job at a time.
        def work():
            _img_state.update(busy=True, error=None, pipe=None, model=key, backend="comfy")
            try:
                comfy.install()
                comfy.fetch_stack(key, unet=quant or None, lora=lora or None,
                                  lora_repo=(lora_repo or "").strip() or None,
                                  lora_strength=float(lora_strength or 1.0))
                comfy.start()  # also mints a gui url for power users; not required here
                _img_state["pipe"] = ("comfy", key)  # truthy sentinel = ready to generate
            except Exception as e:
                _log_tb(f"comfy image setup {key}")
                _img_state["error"] = f"{type(e).__name__}: {e}"
            finally:
                _img_state["busy"] = False
                set_progress("idle")
        threading.Thread(target=work, daemon=True).start()
        return _console("busy", f"installing comfy + fetching {key} — 5-15GB first time",
                        prog=True, note="runs in the background; press Refresh to follow")

    # diffusers path (experimental on t4)
    if IMAGE_MODELS.get(key, {}).get("comfy_only"):
        # name the comfy entry by the SAME label the dropdown shows, or the
        # pointer reads like it's talking about a model that isn't listed
        return _console("err", f"{key} needs both t4s — not doable via diffusers here",
                        note=f"pick '{comfy.stack_repo('ideogram4')}  ·  comfy ✓ reliable' from "
                             "this dropdown instead — same model, runs headless on both cards")

    def work():
        _img_state.update(busy=True, error=None, pipe=None, model=key, backend="diffusers")
        try:
            image_models.install(key)
            # loads on one t4 (bnb resident / unquantized offload) -- diffusers
            # multi-gpu split proved unreliable here; big models -> comfy stack
            _img_state["pipe"] = image_models.load(key)
        except Exception as e:
            _log_tb(f"image load {key}")
            _img_state["error"] = f"{type(e).__name__}: {e}"
        finally:
            _img_state["busy"] = False
            set_progress("idle")

    threading.Thread(target=work, daemon=True).start()
    return _console("busy", f"installing + loading {key} on gpu 0 — takes minutes",
                    note="press Refresh to follow progress")


def _img_status():
    if _img_state["busy"]:
        return _console("busy", f"loading {_img_state['model']}", prog=True)
    if _img_state["error"]:
        return _console("err", "image model failed to load", err=_img_state["error"],
                        note=f"full traceback: {STUDIO_LOG}")
    if _img_state["gen_busy"]:
        # live clock: a t4 run is minutes for the heavy models, and a frozen
        # "generating…" with no number is indistinguishable from a hang
        el = time.time() - (_img_state["gen_t0"] or time.time())
        return _console("busy", f"generating — {_fmt_eta(el)} elapsed", gpu=True,
                        note="heavy models take minutes on a t4; fewer steps and "
                             "768 instead of 1024 are the two big speed levers")
    if _img_state["gen_error"]:
        secs = _img_state["gen_secs"]
        return _console("err", f"generation failed after {_fmt_eta(secs or 0)}",
                        err=_img_state["gen_error"],
                        note=f"full traceback also in {STUDIO_LOG}")
    if _img_state["last_image"]:
        secs = _img_state["gen_secs"]
        took = f" in {_fmt_eta(secs)}" if secs else ""
        if _img_state["gen_warn"]:  # saved, but the image is unusable -- say so
            return _console("err", f"unusable output after {_fmt_eta(secs or 0)}",
                            note=_img_state["gen_warn"])
        return _console("live", f"saved {_img_state['last_image']}{took}"
                                " — press Refresh after the next Generate")
    if _img_state["pipe"] is not None:
        return _console("live", f"{_img_state['model']} ready — write a prompt and press Generate")
    return _console("idle", "no image model loaded — pick one and press Install + load")


def _img_refresh():
    return _img_status(), _img_state["last_image"]


def _img_generate(prompt, steps, width, height, init_image=None, denoise=0.75):
    """runs in the background: a t4 generation takes minutes, and holding the
    request open that long gets killed by the share tunnel (the bare 'Error'
    pills with no message). refresh pulls the result."""
    if _img_state["pipe"] is None or _img_state["gen_busy"]:
        return _img_refresh()

    def work():
        _img_state.update(gen_busy=True, gen_error=None, last_image=None,
                          gen_t0=time.time(), gen_secs=None, gen_warn=None)

        # clamp so a public link can't be used to OOM/DoS the shared gpu --
        # blank passes through (each backend picks its own default).
        def _bounded(v, lo, hi):
            if not v:
                return v
            try:
                return max(lo, min(hi, int(v)))
            except (TypeError, ValueError):
                return v
        nonlocal width, height, steps  # rebind the handler params, not new locals
        width, height = _bounded(width, 256, 2048), _bounded(height, 256, 2048)
        steps = _bounded(steps, 1, 50)
        try:
            if _img_state.get("backend") == "sdcpp":
                _img_state["last_image"] = sdcpp.generate(
                    _img_state["model"], prompt,
                    width=int(width) if width else 1024,
                    height=int(height) if height else 1024,
                    steps=int(steps) if steps else None,
                    init_image=init_image or None,
                    strength=float(denoise or 0.75))
            elif _img_state.get("backend") == "comfy":
                _img_state["last_image"] = comfy.generate_image(
                    _img_state["model"], prompt,
                    width=int(width) if width else None,
                    height=int(height) if height else None,
                    steps=int(steps) if steps else None,
                    init_image=init_image or None,
                    denoise=float(denoise or 0.75))
            elif init_image:
                # diffusers img2img is a different pipeline class entirely, not a
                # kwarg -- don't silently ignore the attachment and hand back a
                # plain text2image, which would look like the upload did nothing
                raise RuntimeError(
                    "attaching an image only works on the comfy models for now "
                    "(the '· comfy ✓ reliable' ones). the diffusers path needs a "
                    "separate img2img pipeline.")
            else:
                kwargs = {}
                if steps:
                    kwargs["num_inference_steps"] = int(steps)
                if width:
                    kwargs["width"] = int(width)
                if height:
                    kwargs["height"] = int(height)
                _img_state["last_image"] = image_models.generate(
                    _img_state["pipe"], prompt, **kwargs)
                # a black frame still "saves" -- carry the reason to the console
                _img_state["gen_warn"] = image_models.LAST_WARNING
        except Exception:
            _img_state["gen_error"] = _log_tb("image generate")[-1500:]
        finally:
            _img_state["gen_secs"] = time.time() - (_img_state["gen_t0"] or time.time())
            _img_state["gen_busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return _console("busy", "generating — 30-90s on t4s; press Refresh for the image",
                    gpu=True), None


# ---- video ---------------------------------------------------------------

def _vid_choices():
    """video dropdown labeled by each stack's primary hf repo; value = short key"""
    return [(comfy.stack_repo(k), k) for k in comfy.video_stacks()]


def _import_video_stack(repo, quant):
    """add a best-effort single-repo stack to the video dropdown"""
    import gradio as gr
    repo = (repo or "").strip().strip("/")
    if repo.count("/") != 1:
        return gr.update(), 'expected "author/repo-name" (e.g. realrebelai/SCAIL-2_GGUF)'
    try:
        files, skipped = comfy.detect_stack(repo, quant=(quant or "").strip() or None)
    except Exception as e:
        return gr.update(), f"import failed: {type(e).__name__}: {e}"
    key = re.sub(r"[^a-z0-9.]+", "-", repo.split("/")[-1].lower()).strip("-")
    key = re.sub(r"-(gguf|comfyui|gguf-comfyui)$", "", key)
    comfy.STACKS[key] = files
    lines = [f"added {key!r}:"]
    lines += [f"  {f} -> models/{sub}/" for _, f, sub in files]
    if skipped:
        lines.append(f"  ({len(skipped)} other files skipped, e.g. {skipped[0]})")
    lines.append("single-repo import: if the model card lists encoders/vae from OTHER "
                 "repos, this stack is incomplete -- those need a curated STACKS entry "
                 "in comfy_bootstrap.py (see ltx-2.3 / scail-2 for the shape).")
    return gr.update(choices=_vid_choices(), value=key), "\n".join(lines)


def _vid_start(key):
    if _vid_state["busy"]:
        return _console("busy", "already setting up — press Stop to cancel it first")
    if _llm_running():
        return _console("err", f"an LLM ({_state['model']}) is running on gpu 0",
                        note="press Stop in the Launch tab first — comfyui wants most of "
                             "the box's vram, and an llm sharing the gpu will OOM it")

    _vid_state["gen"] += 1
    my_gen = _vid_state["gen"]

    def work():
        _vid_state.update(busy=True, url=None, error=None, stack=key)
        try:
            comfy.install()
            comfy.fetch_stack(key)
            url = comfy.start()
            if _vid_state["gen"] == my_gen:
                _vid_state["url"] = url
            else:
                comfy.stop()  # cancelled mid-flight
        except Exception as e:
            if _vid_state["gen"] == my_gen:
                _log_tb(f"video setup {key}")
                _vid_state["error"] = f"{type(e).__name__}: {e}"
        finally:
            if _vid_state["gen"] == my_gen:
                _vid_state["busy"] = False
                set_progress("idle")

    threading.Thread(target=work, daemon=True).start()
    return _console("busy", f"installing comfyui + fetching {key} — 15-25GB first time",
                    prog=True, note="press Stop to cancel")


def _vid_status():
    if _vid_state["busy"]:
        return _console("busy", f"setting up {_vid_state['stack']}",
                        prog=True, log_path=comfy.COMFY_LOG)
    if _vid_state["error"]:
        return _console("err", "comfyui setup failed", err=_vid_state["error"],
                        log_path=comfy.COMFY_LOG)
    if _vid_state["url"]:
        return _console("live", "comfyui node gui live — open it in a new tab",
                        link=_vid_state["url"], log_path=comfy.COMFY_LOG)
    return _console("idle", "comfyui not running — pick a stack and press Install + start")


def _vid_stop():
    _vid_state["gen"] += 1  # cancel any in-flight setup
    _vid_state.update(busy=False, url=None, error=None, stack=None)
    set_progress("idle")
    comfy.stop()
    return _console("idle", "stopped comfyui")


# ---- ui ------------------------------------------------------------------

_C = {"bg": "#0F172A", "panel": "#1E293B", "field": "#0B1120", "border": "#334155",
      "ink": "#F8FAFC", "sub": "#94A3B8", "run": "#22C55E", "run_ink": "#052E16",
      "warn": "#F59E0B", "err": "#EF4444"}

_CSS = f"""
.gradio-container {{ max-width: 1180px !important; margin: 0 auto !important; }}
#km-hdr {{ border-bottom: 1px solid {_C['border']}; padding: 2px 0 12px; margin-bottom: 4px; }}
#km-hdr h1 {{ font-size: 1.05rem; font-weight: 500; letter-spacing: .01em; margin: 0; }}
#km-hdr h1 .km-cursor {{ color: {_C['run']}; }}
#km-hdr p {{ margin: 3px 0 0; color: {_C['sub']}; font-size: .78rem; }}
.km-console {{ background: {_C['field']}; border: 1px solid {_C['border']};
               border-radius: 6px; padding: 12px 14px; min-height: 52px; }}
.km-console .km-head {{ font-size: .85rem; color: {_C['ink']}; }}
.km-led {{ font-size: .8rem; }}
.km-led.live {{ color: {_C['run']}; }}
.km-led.busy {{ color: {_C['warn']}; }}
.km-led.err  {{ color: {_C['err']}; }}
.km-led.idle {{ color: {_C['sub']}; }}
.km-console .km-note {{ color: {_C['sub']}; font-size: .76rem; margin-top: 6px; }}
.km-console .km-note a {{ color: {_C['run']}; text-decoration: none;
                          border-bottom: 1px solid {_C['run']}44; }}
.km-console pre {{ background: transparent; border: 0; margin: 10px 0 0; padding: 10px 0 0;
                   border-top: 1px solid {_C['border']}; color: {_C['sub']};
                   font-size: .7rem; line-height: 1.5; max-height: 300px; overflow: auto; }}
.km-console pre.km-err {{ color: {_C['err']}; border-top: 0; padding-top: 4px; margin-top: 6px; }}
.km-console .km-prog {{ color: {_C['run']}; font-size: .78rem; margin-top: 8px;
                        white-space: pre-wrap; font-variant-numeric: tabular-nums; }}
button {{ transition: background 150ms ease-out, border-color 150ms ease-out; }}
button:focus-visible, input:focus-visible {{ outline: 2px solid {_C['run']} !important; outline-offset: 1px; }}
@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
}}
"""

_HDR = ('<div><h1>kaggle-model-server<span class="km-cursor">_</span></h1>'
        '<p>llm · image · video off a free t4x2 — launch, watch the run light, go</p></div>')


def _style(gr):
    """terminal control room: dark slate + jetbrains mono + one green run
    light, forced in both browser modes. falls back to a stock theme when a
    gradio version rejects any knob."""
    try:
        mono = [gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"]
        t = gr.themes.Base(primary_hue=gr.themes.colors.green,
                           neutral_hue=gr.themes.colors.slate,
                           radius_size=gr.themes.sizes.radius_sm,
                           font=mono, font_mono=mono)
        pal = {
            "body_background_fill": _C["bg"],
            "body_text_color": _C["ink"],
            "body_text_color_subdued": _C["sub"],
            "background_fill_primary": _C["panel"],
            "background_fill_secondary": _C["bg"],
            "border_color_primary": _C["border"],
            "block_background_fill": _C["panel"],
            "block_border_color": _C["border"],
            "block_title_text_color": _C["sub"],
            "block_label_text_color": _C["sub"],
            "input_background_fill": _C["field"],
            "input_border_color": _C["border"],
            "button_primary_background_fill": _C["run"],
            "button_primary_background_fill_hover": "#16A34A",
            "button_primary_text_color": _C["run_ink"],
            "button_secondary_background_fill": _C["panel"],
            "button_secondary_background_fill_hover": "#273248",
            "button_secondary_text_color": _C["ink"],
            "button_secondary_border_color": _C["border"],
        }
        t.set(**pal, **{f"{k}_dark": v for k, v in pal.items()})
        theme = t
    except Exception as e:
        print(f"custom theme unavailable on this gradio ({type(e).__name__}: {e}); using stock")
        theme = gr.themes.Soft() if hasattr(gr, "themes") else None
    return {"theme": theme, "css": _CSS}


def _launch_takes_style(gr):
    # gradio 6 moved theme/css from the Blocks constructor to launch()
    return "theme" in inspect.signature(gr.Blocks.launch).parameters


def _build():
    import gradio as gr

    style = {} if _launch_takes_style(gr) else _style(gr)
    with gr.Blocks(title="kaggle model server", **style) as demo:
        gr.Markdown(_HDR, elem_id="km-hdr")
        with gr.Accordion("hf token — needed for gated models (flux, ideogram, fal)",
                          open=False):
            with gr.Row():
                tok_box = gr.Textbox(label="paste token (hf_...)", type="password",
                                     scale=4)
                tok_btn = gr.Button("Save token", variant="primary", scale=1)
            tok_status = gr.Textbox(
                label="status", lines=2,
                value=("HF_TOKEN already set from the environment"
                       if os.environ.get("HF_TOKEN") else
                       "no token set — fine for ungated models. get one at "
                       "huggingface.co/settings/tokens (read access)"))
            tok_btn.click(_set_hf_token, inputs=tok_box, outputs=[tok_box, tok_status])
        with gr.Tab("launch"):
            with gr.Row():
                with gr.Column(scale=5):
                    model = gr.Dropdown(choices=sorted(MODELS), label="model",
                                        value=sorted(MODELS)[0])
                    quant = gr.Dropdown(choices=[], value=None, allow_custom_value=True,
                                        label="gguf file — pick a model to list; empty = registry default")
                    with gr.Row():
                        ctx = gr.Number(label="ctx",
                                        value=MODELS[sorted(MODELS)[0]].get("ctx", 8192))
                        n_cpu_moe = gr.Number(label="n_cpu_moe — blank = default", value=None)
                    api_key = gr.Textbox(label="api key — recommended, the tunnel url is public", type="password")
                    with gr.Row():
                        launch_btn = gr.Button("Launch", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                        refresh_btn = gr.Button("Refresh")
                        harvest_btn = gr.Button("Harvest cache")
                with gr.Column(scale=6):
                    status = gr.Markdown(_status(), elem_classes="km-console")
                    with gr.Accordion("import a model from hugging face", open=False):
                        gr.Markdown("paste any gguf repo id — file choice and gpu split are "
                                    "auto-detected from the repo's real file sizes.")
                        imp_repo = gr.Textbox(label="repo id",
                                              placeholder="bartowski/SomeModel-GGUF")
                        imp_btn = gr.Button("Fetch and add", variant="primary")
                        imp_status = gr.Textbox(label="result", lines=4)

            model.change(_on_model_change, inputs=model, outputs=[quant, ctx, n_cpu_moe])
            launch_btn.click(_launch, inputs=[model, quant, ctx, n_cpu_moe, api_key],
                             outputs=status)
            stop_btn.click(_stop, outputs=status)
            refresh_btn.click(_status, outputs=status)
            harvest_btn.click(_harvest, outputs=status)
            imp_btn.click(_import_model, inputs=imp_repo,
                          outputs=[model, quant, ctx, n_cpu_moe, imp_status])
        with gr.Tab("chat"):
            # kaggle preinstalls an older gradio whose history defaults to the
            # deprecated tuples format; gradio 6 removed the kwarg entirely.
            # ask for openai-style messages wherever the knob still exists
            # (_normalize_history copes with either format regardless).
            chat_kwargs = (
                {"type": "messages"}
                if "type" in inspect.signature(gr.ChatInterface.__init__).parameters
                else {})
            gr.ChatInterface(
                _chat,
                additional_inputs=[gr.Textbox(
                    label="system prompt — optional; model identity/behavior lives here",
                    value="")],
                **chat_kwargs)
            gr.Markdown('<div class="km-note">card-recommended sampling from the '
                        'registry entry is applied automatically.</div>')
        with gr.Tab("image"):
            with gr.Row():
                with gr.Column(scale=5):
                    img_model = gr.Dropdown(
                        choices=_img_choices(), value="z-image",
                        label="image model — comfy ones run headless (reliable); diffusers in-process")
                    img_quant = gr.Dropdown(choices=[], value=None, allow_custom_value=True,
                                            label="quant — pick a model to list; blank = stack default")
                    with gr.Accordion("lora — optional, any repo", open=False):
                        gr.Markdown('<div class="km-note">blank repo = look in the stack\'s own '
                                    'repo. paste any hf repo id to pull a lora from elsewhere; '
                                    'it stacks on top of a model\'s built-in lora.</div>')
                        with gr.Row():
                            img_lora_repo = gr.Textbox(label="lora repo — blank = this stack's",
                                                       placeholder="author/some-loras", scale=3)
                            img_lora_list = gr.Button("List", scale=1)
                        img_lora = gr.Dropdown(choices=[], value=None, allow_custom_value=True,
                                               label="lora — blank = none")
                        img_lora_str = gr.Number(label="strength", value=1.0)
                    with gr.Row():
                        img_setup_btn = gr.Button("Install + load", variant="primary")
                        img_refresh_btn = gr.Button("Refresh")
                    img_prompt = gr.Textbox(label="prompt", lines=3)
                    with gr.Row():
                        img_steps = gr.Number(label="steps — blank = default", value=None)
                        img_w = gr.Number(label="width — blank = default", value=None)
                        img_h = gr.Number(label="height — blank = default", value=None)
                    gr.Markdown('<div class="km-note">comfy models render at 1024² (blank = default) '
                                'and run on both t4s via comfyui in the background — no node gui to open. '
                                'diffusers models run in-process on one t4: keep those to 768².</div>')
                    with gr.Accordion("attach an image — optional (img2img)", open=False):
                        gr.Markdown('<div class="km-note">start from a picture instead of noise. '
                                    'strength is how much freedom the model gets: 0.3 keeps your '
                                    'image and restyles lightly, 0.8 keeps only the composition. '
                                    'the output size comes from your image, so width/height are '
                                    'ignored. comfy models only, and not flux2/ideogram yet.</div>')
                        img_init = gr.Image(label="source image", type="filepath")
                        img_denoise = gr.Number(label="strength — 0.3 subtle … 0.9 loose",
                                                value=0.75)
                    img_go = gr.Button("Generate", variant="primary")
                    gr.Markdown('<div class="km-note">the comfy models are the dependable path on t4 '
                                '(comfy\'s fp16 works where diffusers black-frames). ideogram4 · comfy is '
                                'experimental — its graph is newer and may still need the node gui. '
                                'flux / ideogram need an HF_TOKEN + accepted license.</div>')
                with gr.Column(scale=6):
                    img_status = gr.Markdown(_img_status(), elem_classes="km-console")
                    img_out = gr.Image(label="result")
                    with gr.Accordion("import a diffusers model from hugging face", open=False):
                        gr.Markdown("paste any diffusers-format repo id — gating and "
                                    "nf4 need are auto-detected from the repo.")
                        img_imp_repo = gr.Textbox(label="repo id",
                                                  placeholder="Tongyi-MAI/Z-Image-Turbo")
                        img_imp_btn = gr.Button("Fetch and add", variant="primary")
                        img_imp_status = gr.Textbox(label="result", lines=4)

            img_model.change(_on_img_model_change, inputs=img_model, outputs=img_quant)
            img_lora_list.click(_list_img_loras, inputs=[img_model, img_lora_repo],
                                outputs=img_lora)
            img_setup_btn.click(
                _img_setup,
                inputs=[img_model, img_quant, img_lora, img_lora_repo, img_lora_str],
                outputs=img_status)
            img_refresh_btn.click(_img_refresh, outputs=[img_status, img_out])
            img_imp_btn.click(_import_image_model, inputs=img_imp_repo,
                              outputs=[img_model, img_imp_status])
            img_go.click(_img_generate,
                         inputs=[img_prompt, img_steps, img_w, img_h, img_init, img_denoise],
                         outputs=[img_status, img_out])
        with gr.Tab("video"):
            gr.Markdown('<div class="km-note">video runs through comfyui on both t4s. it boots '
                        'once and serves its full node gui at a public url — open that to drive the '
                        'workflow. wants most of the box\'s vram — stop the llm first if it ooms. '
                        '(image models moved to the image tab; they generate there with no gui.)</div>')
            vid_stack = gr.Dropdown(choices=_vid_choices(), value="ltx-2.3", label="video stack")
            with gr.Row():
                vid_start_btn = gr.Button("Install + start", variant="primary")
                vid_stop_btn = gr.Button("Stop", variant="stop")
                vid_refresh_btn = gr.Button("Refresh")
            vid_status = gr.Markdown(_vid_status(), elem_classes="km-console")
            gr.Markdown('<div class="km-note">the url serves comfyui\'s full node gui — any '
                        'workflow jsons shipped with the stack appear in its workflow browser. '
                        'open it in a new browser tab once the run light is green.</div>')
            with gr.Accordion("import a video pack from hugging face", open=False):
                gr.Markdown("paste a comfyui-style repo (gguf pack or comfy-org repackage) — "
                            "files are mapped into comfyui's model dirs by their paths, one "
                            "unet quant is picked for the t4. multi-repo recipes still need "
                            "a curated stack.")
                vid_imp_repo = gr.Textbox(label="repo id",
                                          placeholder="realrebelai/SCAIL-2_GGUF")
                vid_imp_quant = gr.Textbox(label="unet quant — blank = auto (~Q4, fits a t4)",
                                           placeholder="Q3_K_M")
                vid_imp_btn = gr.Button("Fetch and add", variant="primary")
                vid_imp_status = gr.Textbox(label="result", lines=6)

            vid_start_btn.click(_vid_start, inputs=vid_stack, outputs=vid_status)
            vid_stop_btn.click(_vid_stop, outputs=vid_status)
            vid_refresh_btn.click(_vid_status, outputs=vid_status)
            vid_imp_btn.click(_import_video_stack, inputs=[vid_imp_repo, vid_imp_quant],
                              outputs=[vid_stack, vid_imp_status])

        # auto-refresh ONLY while something is actively launching/loading, then
        # go silent -- gr.skip() at rest means no re-render (no scroll jump, no
        # flicker, no event-queue flooding while a model is just running/idle).
        # the manual Refresh buttons still work anytime.
        if hasattr(gr, "Timer") and hasattr(gr, "skip"):
            timer = gr.Timer(3)
            timer.tick(_tick_launch, outputs=status)
            timer.tick(_tick_img, outputs=[img_status, img_out])
            timer.tick(_tick_vid, outputs=vid_status)
    return demo


def _hold_cell():
    """keep the notebook CELL running so kaggle doesn't reap the session.

    kaggle stops an interactive session after ~20 min of "idle", where idle
    means no cell is executing. gradio's launch() returns immediately inside a
    notebook (the server moves to a background thread), so the cell completes,
    the idle timer starts, and the session dies mid-generation -- taking
    /kaggle/tmp with it, i.e. the sd.cpp build and every downloaded weight.
    blocking here keeps a cell 'running' for as long as the studio is up.
    the heartbeat is also the only liveness signal you get in the notebook."""
    print("holding this cell so kaggle doesn't idle-stop the session "
          "(interrupt the cell to release it; the studio keeps running).",
          flush=True)
    try:
        while True:
            time.sleep(300)
            busy = ("launching " + str(_state["model"])) if _state["busy"] else (
                "generating" if _img_state["gen_busy"] else "idle")
            print(f"[{time.strftime('%H:%M:%S')}] studio alive — {busy}", flush=True)
    except KeyboardInterrupt:
        print("released; kaggle's idle timer is now running again.", flush=True)


def launch_panel(auth=None, keep_alive=True, *, share=True, allow_public=False):
    """builds and serves the studio; returns the gradio app.

    the share url is PUBLIC AND GUESSABLE, so auth=("user","pass") is required
    whenever share=True: calling launch_panel() bare would otherwise hand anyone
    on the internet full control of your gpus. to deliberately run an open panel
    (e.g. on a trusted lan), pass allow_public=True.

    keep_alive=True blocks the calling cell (see _hold_cell) -- pass False if
    you want the cell back."""
    import gradio as gr
    if share and auth is None and not allow_public:
        raise ValueError(
            "launch_panel() refuses to share publicly without auth -- anyone with "
            'the link would control your gpus. pass auth=("user", "pass"), or '
            "allow_public=True to deliberately run an open panel.")
    if auth is None:
        print("WARNING: no auth -- anyone with the share link controls your gpus.")
    demo = _build()
    # show_error surfaces the real message in the browser when an event
    # fails client-side (instead of gradio's bare "Error" pill).
    # allowed_paths: gradio only serves files from cwd/tempdir by default,
    # and generated images live under /kaggle/tmp/outputs
    kwargs = dict(share=share, auth=auth, server_port=7860, show_error=True,
                  allowed_paths=[image_models.OUT_DIR, f"{comfy.COMFY_DIR}/output",
                                 sdcpp.OUT_DIR])
    if _launch_takes_style(gr):
        kwargs.update(_style(gr))
    demo.launch(**kwargs)
    if keep_alive:
        _hold_cell()   # blocks: kaggle reaps sessions with no cell running
    return demo
