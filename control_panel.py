"""browser studio for the llm stack: launch/stop models, switch quants, and
chat with whatever's running -- one gradio app, unsloth-studio style
"open a notebook, Run All, click the link".

    from control_panel import launch_panel
    launch_panel(auth=("user", "a-real-password"))

needs: pip install gradio. the gradio share link is the panel's own public
url -- independent of the model's cloudflared url, both coexist fine. share
links are PUBLIC AND GUESSABLE: always pass auth=("user", "pass"), anyone
with the link can otherwise start/stop your models (run_studio.ipynb
generates a random password and prints it).

honest v1: launches run in a background thread (a cold launch takes
minutes: build + download + load) with a manual "refresh status" button;
the chat tab streams from the local openai endpoint.
"""

import inspect
import json
import threading

import requests

import comfy_bootstrap as comfy
import image_models
from harness import SERVER_LOG, _tail, list_quants, run, stop
from image_models import IMAGE_MODELS
from model_registry import MODELS

_PORT = 8080  # panel always launches on harness's default port
_state = {"busy": False, "url": None, "error": None, "model": None, "api_key": None}
_img_state = {"busy": False, "pipe": None, "error": None, "model": None}
_vid_state = {"busy": False, "url": None, "error": None, "stack": None}


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
        return "a launch is already in progress -- click refresh status"
    overrides = {}
    if gguf_file:
        overrides["hf_file"] = gguf_file  # exact file picked from list_quants
    if ctx:
        overrides["ctx"] = int(ctx)
    if n_cpu_moe not in (None, ""):
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
    return f"launching {model_key} in the background... click refresh status"


def _stop():
    stop()
    _state.update(url=None, error=None, model=None)
    return "stopped"


def _status():
    if _state["busy"]:
        head = f"LAUNCHING {_state['model']} (build/download/load -- takes minutes)"
    elif _state["error"]:
        head = f"FAILED: {_state['error']}"
    elif _state["url"]:
        head = (f"{_state['model']} LIVE AT: {_state['url']}\n"
                f"  chat here in the Chat tab, or at that url; api at {_state['url']}/v1")
    else:
        head = "nothing running"
    return f"{head}\n\n--- last 40 lines of {SERVER_LOG} ---\n{_tail(SERVER_LOG, 40)}"


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
    content-block lists, legacy [user, assistant] pairs -- 6.20 passes pairs
    at runtime despite its own MessageDict type hints). accept them all."""
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
        yield "model is still launching -- check refresh status in the Launch tab"
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


def _img_setup(key):
    if _img_state["busy"]:
        return "already installing/loading -- click refresh"

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
    return f"installing + loading {key} on gpu 1 (pip + weights download -- minutes)... click refresh"


def _img_status():
    if _img_state["busy"]:
        return f"LOADING {_img_state['model']} -- pip install + checkpoint download takes minutes"
    if _img_state["error"]:
        return f"FAILED: {_img_state['error']}"
    if _img_state["pipe"] is not None:
        return f"{_img_state['model']} ready -- prompt away"
    return "no image model loaded"


def _img_generate(prompt, steps):
    if _img_state["pipe"] is None:
        return _img_status(), None
    kwargs = {"num_inference_steps": int(steps)} if steps else {}
    try:
        path = image_models.generate(_img_state["pipe"], prompt, **kwargs)
        return f"saved {path}", path
    except Exception as e:
        return f"generation failed: {type(e).__name__}: {e}", None


def _vid_start(key):
    if _vid_state["busy"]:
        return "already setting up -- click refresh"

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
    return f"installing comfyui + fetching {key} (15-25GB first time)... click refresh"


def _vid_status():
    if _vid_state["busy"]:
        return (f"SETTING UP {_vid_state['stack']} -- big downloads, be patient "
                f"(log: /kaggle/tmp/comfyui.log)")
    if _vid_state["error"]:
        return f"FAILED: {_vid_state['error']}"
    if _vid_state["url"]:
        return f"comfyui node gui LIVE AT: {_vid_state['url']}"
    return "comfyui not running"


def _vid_stop():
    comfy.stop()
    _vid_state.update(url=None, error=None, stack=None)
    return "stopped comfyui"


def _build():
    import gradio as gr

    with gr.Blocks(title="kaggle model server") as demo:
        gr.Markdown("## kaggle model server — studio")
        with gr.Tab("Launch"):
            model = gr.Dropdown(choices=sorted(MODELS), label="model",
                                value=sorted(MODELS)[0])
            quant = gr.Dropdown(choices=[], value=None, allow_custom_value=True,
                                label="gguf file (pick model first to populate; empty = registry default)")
            with gr.Row():
                ctx = gr.Number(label="ctx", value=MODELS[sorted(MODELS)[0]].get("ctx", 8192))
                n_cpu_moe = gr.Number(label="n_cpu_moe (blank = default)", value=None)
            api_key = gr.Textbox(label="api key (optional but wise -- tunnel urls are public)")
            status = gr.Textbox(label="status", lines=14)
            with gr.Row():
                launch_btn = gr.Button("Launch", variant="primary")
                stop_btn = gr.Button("Stop")
                refresh_btn = gr.Button("Refresh status")

            model.change(_on_model_change, inputs=model, outputs=[quant, ctx, n_cpu_moe])
            launch_btn.click(_launch, inputs=[model, quant, ctx, n_cpu_moe, api_key],
                             outputs=status)
            stop_btn.click(_stop, outputs=status)
            refresh_btn.click(_status, outputs=status)
        with gr.Tab("Chat"):
            # kaggle preinstalls an older gradio whose history defaults to the
            # deprecated tuples format; gradio 6 removed the kwarg entirely.
            # ask for openai-style messages wherever the knob still exists
            # (_normalize_history copes with either format regardless).
            chat_kwargs = (
                {"type": "messages"}
                if "type" in inspect.signature(gr.ChatInterface.__init__).parameters
                else {})
            gr.ChatInterface(_chat, **chat_kwargs)
        with gr.Tab("Image"):
            gr.Markdown("loads on **gpu 1**, so it runs beside an llm on gpu 0. "
                        "flux1-dev / ideogram-4 need an HF_TOKEN secret + accepted license.")
            img_model = gr.Dropdown(choices=sorted(IMAGE_MODELS), value="z-image-turbo",
                                    label="image model")
            img_status = gr.Textbox(label="status", lines=2)
            with gr.Row():
                img_setup_btn = gr.Button("Install + load", variant="primary")
                img_refresh_btn = gr.Button("Refresh status")
            img_prompt = gr.Textbox(label="prompt")
            img_steps = gr.Number(label="steps (blank = model default)", value=None)
            img_go = gr.Button("Generate", variant="primary")
            img_out = gr.Image(label="result")

            img_setup_btn.click(_img_setup, inputs=img_model, outputs=img_status)
            img_refresh_btn.click(_img_status, outputs=img_status)
            img_go.click(_img_generate, inputs=[img_prompt, img_steps],
                         outputs=[img_status, img_out])
        with gr.Tab("Video"):
            gr.Markdown("boots headless comfyui + tunnels its **full node gui** -- "
                        "open the printed url to build workflows. wants most of the "
                        "box's vram: stop the llm first if things oom. relaunching an "
                        "llm recycles the tunnel slot -- restart comfyui after.")
            vid_stack = gr.Dropdown(choices=sorted(comfy.STACKS), value="ltx-2.3",
                                    label="stack")
            vid_status = gr.Textbox(label="status", lines=3)
            with gr.Row():
                vid_start_btn = gr.Button("Install + start", variant="primary")
                vid_stop_btn = gr.Button("Stop")
                vid_refresh_btn = gr.Button("Refresh status")

            vid_start_btn.click(_vid_start, inputs=vid_stack, outputs=vid_status)
            vid_stop_btn.click(_vid_stop, outputs=vid_status)
            vid_refresh_btn.click(_vid_status, outputs=vid_status)
    return demo


def launch_panel(auth=None):
    """builds and serves the studio; returns the gradio app. auth=("user","pass")
    is strongly recommended -- the share url is public."""
    if auth is None:
        print("WARNING: no auth -- anyone with the share link controls your gpus. "
              'pass auth=("user", "pass").')
    demo = _build()
    demo.launch(share=True, auth=auth, server_port=7860)
    return demo
