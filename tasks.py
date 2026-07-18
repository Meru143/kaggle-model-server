"""small in-notebook models for side tasks: ocr, pdf parsing, transcription,
embeddings, tts. lazy imports everywhere -- import tasks costs nothing until
a helper is called.

every helper takes gpu (default 1) and sets CUDA_VISIBLE_DEVICES before
touching torch, so these can share the box with a llama-server occupying
gpu 0. caveat: the env var only sticks if the framework hasn't initialized
cuda in this process yet -- call helpers before any other torch-on-gpu work
in the same notebook kernel.

each call loads its model fresh (simple > stateful); wrap in
functools.lru_cache yourself if you're looping.

t4 rules applied throughout: fp16 only (cards say bf16 -> adapted, sm75 has
no bf16), no flash-attention (needs ampere+) -> sdpa/eager.
"""

import os
import shutil
import subprocess
import time

_OCR_PROMPT = (
    "\nExtract all readable content from the image in natural human reading "
    "order and output the result as a single Markdown document. For charts or "
    'images, represent them using an HTML image tag: <img src="images/'
    'bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, '
    "bottom are bounding box coordinates scaled to [0, 1000). Format formulas "
    "as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all "
    "other text as standard Markdown. Preserve the original text without "
    "translation or paraphrasing."
)  # verbatim from the ovisocr2 card -- the model is trained on this prompt


def ocr_pages(images, gpu: int = 1):
    """page images (paths or PIL) -> list of markdown strings (ATH-MaaS/OvisOCR2)"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        from PIL import Image
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise ImportError('needs: pip install "vllm==0.22.1" pillow  '
                          "(vllm version pinned by the model card)") from e
    llm = LLM(model="ATH-MaaS/OvisOCR2", tensor_parallel_size=1,
              gpu_memory_utilization=0.8,
              gdn_prefill_backend="triton")  # card-pinned backend
    prompt = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": _OCR_PROMPT}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pils = [im if not isinstance(im, str) else Image.open(im) for im in images]
    outs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": im}} for im in pils],
        SamplingParams(max_tokens=16384, temperature=0.0))
    return [o.outputs[0].text for o in outs]


def parse_pdf(path, out_dir="/kaggle/tmp/outputs/ocr", gpu: int = 1):
    """pdf -> per-page markdown via baidu/Unlimited-OCR infer_multi.
    returns out_dir (results are also written there by the model)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        import fitz  # pymupdf
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError("needs: pip install transformers accelerate pymupdf") from e
    tok = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "baidu/Unlimited-OCR", trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.float16,  # card says bf16; t4 is sm75 -> fp16
    ).eval().cuda()
    os.makedirs(out_dir, exist_ok=True)
    pages = []
    doc = fitz.open(path)
    for i, page in enumerate(doc):  # card recipe: rasterize at 300 dpi
        p = os.path.join(out_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72)).save(p)
        pages.append(p)
    doc.close()
    model.infer_multi(tok, prompt="<image>Multi page parsing.", image_files=pages,
                      output_path=out_dir, image_size=1024, max_length=32768,
                      no_repeat_ngram_size=35, ngram_window=1024, save_results=True)
    return out_dir


def transcribe(audio_path, gpu: int = 1, port=8009):
    """speaker-labelled transcript ([S01]/[S02] + timestamps) via
    OpenMOSS-Team/MOSS-Transcribe-Diarize. boots a throwaway vllm server (the
    card's cuda-12 serving path), transcribes, tears it down."""
    if not shutil.which("vllm"):
        raise ImportError(
            "needs the card's pinned vllm nightly (cu129 = the cuda-12 build):\n"
            "pip install -U vllm --extra-index-url "
            "https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129")
    import requests
    log_path = "/kaggle/tmp/vllm-transcribe.log"
    log = open(log_path, "w")  # log to file, never PIPE (unread pipes block the child)
    proc = subprocess.Popen(
        ["vllm", "serve", "OpenMOSS-Team/MOSS-Transcribe-Diarize",
         "--trust-remote-code", "--port", str(port)],
        stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    try:
        deadline = time.time() + 900
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"vllm exited during startup -- tail {log_path}")
            if time.time() > deadline:
                raise RuntimeError(f"vllm not healthy within 900s -- tail {log_path}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=3).ok:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(3)
        with open(audio_path, "rb") as f:
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/audio/transcriptions",
                files={"file": f},
                data={"model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
                      "response_format": "json", "temperature": "0"},
                timeout=3600)
        r.raise_for_status()
        return r.json()["text"]
    finally:
        proc.terminate()
        log.close()


def embed(texts, gpu: int = 1):
    """texts -> numpy embeddings via nvidia/Nemotron-3-Embed-1B-BF16 (loaded fp16)"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("needs: pip install sentence-transformers") from e
    model = SentenceTransformer(
        "nvidia/Nemotron-3-Embed-1B-BF16", device="cuda",
        model_kwargs={"dtype": torch.float16,  # card says bf16; t4 -> fp16
                      # card says flash_attention_2; that needs ampere+
                      "attn_implementation": "eager"})
    model.max_seq_length = 32768
    return model.encode(texts)


def tts(text, out_path, speaker="Ryan", language="Auto", instruct=None, gpu: int = 1):
    """text -> wav at out_path via Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice.
    speakers: Vivian/Serena/Uncle_Fu/Dylan/Eric/Ryan/Aiden/Ono_Anna/Sohee."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise ImportError("needs: pip install qwen-tts soundfile") from e
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
        dtype=torch.float16,             # card says bf16; t4 is sm75 -> fp16
        attn_implementation="sdpa")      # card says flash-attn-2; needs ampere+
    kwargs = {"instruct": instruct} if instruct else {}
    wavs, sr = model.generate_custom_voice(text=text, language=language,
                                           speaker=speaker, **kwargs)
    sf.write(out_path, wavs[0], sr)
    return out_path
