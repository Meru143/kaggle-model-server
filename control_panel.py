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
import re
import threading

import requests

import comfy_bootstrap as comfy
import image_models
from harness import SERVER_LOG, _tail, detect_entry, list_quants, run, stop
from image_models import IMAGE_MODELS
from model_registry import MODELS

_PORT = 8080  # panel always launches on harness's default port
_state = {"busy": False, "url": None, "error": None, "model": None, "api_key": None}
_img_state = {"busy": False, "pipe": None, "error": None, "model": None}
_vid_state = {"busy": False, "url": None, "error": None, "stack": None}


# ---- statusline console (the one signature element) ----------------------

def _console(led, head, log_path=None, link=None, note=None, err=None):
    """led + one-line status + optional link/note/log tail, as safe html"""
    parts = [f'<span class="km-led {led}">&#9679;</span> '
             f'<span class="km-head">{html.escape(head)}</span>']
    if link:
        safe = html.escape(link, quote=True)
        parts.append(f'<div class="km-note"><a href="{safe}" target="_blank">{safe}</a></div>')
    if note:
        parts.append(f'<div class="km-note">{html.escape(note)}</div>')
    if err:
        parts.append(f'<pre class="km-err">{html.escape(err)}</pre>')
    if log_path:
        parts.append(f'<pre>{html.escape(_tail(log_path, 40))}</pre>')
    return f'<div class="km-console-inner">{"".join(parts)}</div>'


# ---- llm launch ----------------------------------------------------------

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
        return _console("busy", f"a launch is already running — press Refresh for progress")
    overrides = {}
    if gguf_file:
        overrides["hf_file"] = gguf_file  # exact file picked from the quant list
    if ctx:
        overrides["ctx"] = int(ctx)
    # gradio 6 renders an empty Number as 0 -- treat 0 as "not set" (a real
    # n_cpu_moe of 0 is meaningless anyway)
    if n_cpu_moe not in (None, "", 0):
        overrides["n_cpu_moe"] = int(n_cpu_moe)

    def work():
        _state.update(busy=True, url=None, error=None,
                      model=model_key, api_key=api_key or None)
        try:
            _state["url"] = run(model_key, MODELS, api_key=api_key or None, **overrides)
        except Exception as e:
            _state["error"] = f"{type(e).__name__}: {e}"
        finally:
            _state["busy"] = False

    # never block the click handler -- a cold launch takes minutes
    threading.Thread(target=work, daemon=True).start()
    return _console("busy", f"launching {model_key} — build/download/load takes minutes",
                    note="press Refresh to follow progress")


def _stop():
    stop()
    _state.update(url=None, error=None, model=None)
    return _console("idle", "stopped — nothing running")


def _status():
    if _state["busy"]:
        return _console("busy", f"launching {_state['model']} — build/download/load takes minutes",
                        log_path=SERVER_LOG)
    if _state["error"]:
        return _console("err", "launch failed", err=_state["error"], log_path=SERVER_LOG)
    if _state["url"]:
        return _console("live", f"{_state['model']} live",
                        link=_state["url"],
                        note=f"chat in the Chat tab · openai api at {_state['url']}/v1",
                        log_path=SERVER_LOG)
    return _console("idle", "nothing running — pick a model and press Launch")


def _import_model(repo):
    """add any hf gguf repo to the session's registry with auto-detected
    settings, and point the launch controls at it"""
    import gradio as gr
    repo = (repo or "").strip().strip("/")
    if repo.count("/") != 1:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                'expected "author/repo-name" (e.g. bartowski/SomeModel-GGUF)')
    try:
        entry = detect_entry(repo)
    except Exception as e:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                f"import failed: {type(e).__name__}: {e}")
    key = re.sub(r"[^a-z0-9.]+", "-", repo.split("/")[-1].lower()).strip("-")
    key = re.sub(r"-gguf$", "", key)
    MODELS[key] = entry
    quant_update, ctx_update, moe_update = _on_model_change(key)
    gpus = "both t4s" if entry["tensor_split"] else "one t4"
    msg = (f"added {key!r}: {entry['hf_file']} ({entry['est_vram_gb'] - 2:.1f}GB -> {gpus})\n"
           f"defaults are auto-detected guesses -- check the model card for sampling/MTP flags.\n"
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


def _chat(message, history):
    """streams from the local llama-server openai endpoint"""
    if _state["busy"]:
        yield "model is still launching -- check Refresh in the Launch tab"
        return
    msgs = _normalize_history(history)
    msgs.append({"role": "user", "content": _to_text(message)})
    headers = ({"Authorization": f"Bearer {_state['api_key']}"}
               if _state["api_key"] else {})
    try:
        r = requests.post(
            f"http://127.0.0.1:{_PORT}/v1/chat/completions",
            json={"model": _state["model"] or "local", "messages": msgs, "stream": True},
            headers=headers, stream=True, timeout=600)
        r.raise_for_status()
        acc = ""
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:].decode("utf-8")
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or [{}]
            acc += choices[0].get("delta", {}).get("content") or ""
            if acc:
                yield acc
    except requests.exceptions.RequestException as e:
        yield (f"no model answering on port {_PORT} ({type(e).__name__}) -- "
               "launch one in the Launch tab first")


# ---- image ---------------------------------------------------------------

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
    return gr.update(choices=sorted(IMAGE_MODELS), value=key), msg


def _img_setup(key):
    if _img_state["busy"]:
        return _console("busy", "already installing — press Refresh for progress")

    def work():
        _img_state.update(busy=True, error=None, pipe=None, model=key)
        try:
            image_models.install(key)
            # gpu 1 so image gen coexists with a llama-server on gpu 0
            _img_state["pipe"] = image_models.load(key, gpu=1)
        except Exception as e:
            _img_state["error"] = f"{type(e).__name__}: {e}"
        finally:
            _img_state["busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return _console("busy", f"installing + loading {key} on gpu 1 — takes minutes",
                    note="press Refresh to follow progress")


def _img_status():
    if _img_state["busy"]:
        return _console("busy", f"loading {_img_state['model']} — pip install + checkpoint download")
    if _img_state["error"]:
        return _console("err", "image model failed to load", err=_img_state["error"])
    if _img_state["pipe"] is not None:
        return _console("live", f"{_img_state['model']} ready — write a prompt and press Generate")
    return _console("idle", "no image model loaded — pick one and press Install + load")


def _img_generate(prompt, steps):
    if _img_state["pipe"] is None:
        return _img_status(), None
    kwargs = {"num_inference_steps": int(steps)} if steps else {}
    try:
        path = image_models.generate(_img_state["pipe"], prompt, **kwargs)
        return _console("live", f"saved {path}"), path
    except Exception as e:
        return _console("err", "generation failed", err=f"{type(e).__name__}: {e}"), None


# ---- video ---------------------------------------------------------------

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
    return gr.update(choices=sorted(comfy.STACKS), value=key), "\n".join(lines)


def _vid_start(key):
    if _vid_state["busy"]:
        return _console("busy", "already setting up — press Refresh for progress")

    def work():
        _vid_state.update(busy=True, url=None, error=None, stack=key)
        try:
            comfy.install()
            comfy.fetch_stack(key)
            _vid_state["url"] = comfy.start()
        except Exception as e:
            _vid_state["error"] = f"{type(e).__name__}: {e}"
        finally:
            _vid_state["busy"] = False

    threading.Thread(target=work, daemon=True).start()
    return _console("busy", f"installing comfyui + fetching {key} — 15-25GB first time",
                    note="press Refresh to follow progress")


def _vid_status():
    if _vid_state["busy"]:
        return _console("busy", f"setting up {_vid_state['stack']} — big downloads, be patient",
                        log_path=comfy.COMFY_LOG)
    if _vid_state["error"]:
        return _console("err", "comfyui setup failed", err=_vid_state["error"],
                        log_path=comfy.COMFY_LOG)
    if _vid_state["url"]:
        return _console("live", "comfyui node gui live — open it in a new tab",
                        link=_vid_state["url"], log_path=comfy.COMFY_LOG)
    return _console("idle", "comfyui not running — pick a stack and press Install + start")


def _vid_stop():
    comfy.stop()
    _vid_state.update(url=None, error=None, stack=None)
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
button {{ transition: background 150ms ease-out, border-color 150ms ease-out; }}
button:focus-visible, input:focus-visible {{ outline: 2px solid {_C['run']} !important; outline-offset: 1px; }}
@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
}}
"""

_HDR = ('<div><h1>kaggle-model-server<span class="km-cursor">_</span></h1>'
        '<p>llm · image · video off a free t4×2 — launch, watch the run light, go</p></div>')


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
                    api_key = gr.Textbox(label="api key — recommended, the tunnel url is public")
                    with gr.Row():
                        launch_btn = gr.Button("Launch", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                        refresh_btn = gr.Button("Refresh")
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
            gr.ChatInterface(_chat, **chat_kwargs)
        with gr.Tab("image"):
            with gr.Row():
                with gr.Column(scale=5):
                    img_model = gr.Dropdown(choices=sorted(IMAGE_MODELS),
                                            value="z-image-turbo", label="image model")
                    with gr.Row():
                        img_setup_btn = gr.Button("Install + load", variant="primary")
                        img_refresh_btn = gr.Button("Refresh")
                    img_prompt = gr.Textbox(label="prompt", lines=3)
                    img_steps = gr.Number(label="steps — blank = model default", value=None)
                    img_go = gr.Button("Generate", variant="primary")
                    gr.Markdown('<div class="km-note">runs on gpu 1, beside the llm on gpu 0. '
                                'flux1-dev / ideogram-4 need an HF_TOKEN secret + accepted license.</div>')
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

            img_setup_btn.click(_img_setup, inputs=img_model, outputs=img_status)
            img_refresh_btn.click(_img_status, outputs=img_status)
            img_imp_btn.click(_import_image_model, inputs=img_imp_repo,
                              outputs=[img_model, img_imp_status])
            img_go.click(_img_generate, inputs=[img_prompt, img_steps],
                         outputs=[img_status, img_out])
        with gr.Tab("video"):
            vid_stack = gr.Dropdown(choices=sorted(comfy.STACKS), value="ltx-2.3",
                                    label="stack")
            with gr.Row():
                vid_start_btn = gr.Button("Install + start", variant="primary")
                vid_stop_btn = gr.Button("Stop", variant="stop")
                vid_refresh_btn = gr.Button("Refresh")
            vid_status = gr.Markdown(_vid_status(), elem_classes="km-console")
            gr.Markdown('<div class="km-note">the url serves comfyui\'s full node gui — build '
                        'workflows there. wants most of the box\'s vram: stop the llm first if it '
                        'ooms. relaunching an llm recycles the tunnel slot — restart comfyui after.</div>')
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
    return demo


def launch_panel(auth=None):
    """builds and serves the studio; returns the gradio app. auth=("user","pass")
    is strongly recommended -- the share url is public."""
    import gradio as gr
    if auth is None:
        print("WARNING: no auth -- anyone with the share link controls your gpus. "
              'pass auth=("user", "pass").')
    demo = _build()
    kwargs = dict(share=True, auth=auth, server_port=7860)
    if _launch_takes_style(gr):
        kwargs.update(_style(gr))
    demo.launch(**kwargs)
    return demo
