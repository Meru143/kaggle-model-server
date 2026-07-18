"""point-and-click alternative to editing the notebook config cell (llm stack).

    from control_panel import launch_panel
    launch_panel(auth=("user", "a-real-password"))

needs: pip install gradio. the gradio share link is the panel's own public
url -- independent of the model's cloudflared url, both coexist fine. share
links are PUBLIC AND GUESSABLE: always pass auth=("user", "pass"), anyone
with the link can otherwise start/stop your models.

honest v1: no websockets, no auto-polling -- launches run in a background
thread (a cold launch takes minutes: build + download + load) and a manual
"refresh status" button shows progress via the server log tail.
"""

import threading

from harness import SERVER_LOG, _tail, list_quants, run, stop
from model_registry import MODELS

_state = {"busy": False, "url": None, "error": None, "model": None}


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
        _state.update(busy=True, url=None, error=None, model=model_key)
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
                f"  chat web ui at that url, openai-compatible api at {_state['url']}/v1")
    else:
        head = "nothing running"
    return f"{head}\n\n--- last 40 lines of {SERVER_LOG} ---\n{_tail(SERVER_LOG, 40)}"


def launch_panel(auth=None):
    """builds and serves the panel; returns the gradio app. auth=("user","pass")
    is strongly recommended -- the share url is public."""
    import gradio as gr

    with gr.Blocks(title="kaggle model server") as demo:
        gr.Markdown("## kaggle model server — control panel")
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

    if auth is None:
        print("WARNING: no auth -- anyone with the share link controls your gpus. "
              'pass auth=("user", "pass").')
    demo.launch(share=True, auth=auth, server_port=7860)
    return demo
