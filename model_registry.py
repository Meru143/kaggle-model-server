"""
one entry per model. repos and filenames below are real and were verified
against hugging face in july 2026 -- no more TODOs.

field notes:
  ngl             -1 or a big number (99) offloads all layers to gpu
  tensor_split    only set for models spanning both t4s, e.g. "1,1" for even split
  gpu_devices     physical gpu indices this model is allowed to see (CUDA_VISIBLE_DEVICES)
  n_cpu_moe       optional. for MoE models, pushes N expert blocks to cpu/ram while
                  keeping attention + shared layers on gpu. useful for squeezing a
                  MoE model that's just over the vram budget without dropping quant.
  llama_cpp_repo  optional. per-model llama.cpp source. some models (bonsai) need
                  a fork with custom kernels; omit to use mainline ggml-org/llama.cpp.
                  each repo gets its own build dir + cached-binary name in the harness.
  extra_args      optional. raw extra llama-server flags appended to the command.
  est_vram_gb     your own benchmarked number, not enforced by the harness -- just
                  here so the registry doubles as a reference when picking a model.
  sampling        optional. the card's recommended request-time sampling params in
                  openai style (llama-server also accepts top_k/repeat_penalty/min_p).
                  the studio chat applies these automatically; api users copy them.
                  qwen3.6-family entries carry the thinking-mode defaults.

kaggle 2x t4 + 30GB ram budget cheat sheet:
  one t4      ~15GB usable -> gguf file <= ~12GB leaves room for 8k kv + buffers
  both t4s    gguf file <= ~26GB with tensor_split "1,1" (per-card ceiling still 15GB)
  n_cpu_moe   30GB system ram -> up to ~20GB of MoE expert weights can spill to cpu

philosophy: entries are DEFAULTS, not law. every field here can be overridden
per-call via run() kwargs (see harness.run), including quant switching by name:
    run("bartowski/Qwen_Qwen3.6-35B-A3B-GGUF", MODELS, quant="Q3_K_M", ctx=16384)
list_quants(key, MODELS) shows what a repo offers. a proven override earns a
promotion back into its entry; edit this file only to change defaults.
"""

MODELS = {
    # prismml's ternary build of qwen3.6-27b. 7.2GB weights, ~8.7GB peak at
    # 8-10k ctx -> lots of headroom on one t4; you could raise ctx well past
    # 8192 if you want. NEEDS the prismml llama.cpp fork (custom Q2_0_g128
    # kernels) -- mainline llama.cpp cannot run this pack. their published
    # cuda numbers are from h100, so treat the first t4 boot as a smoke test.
    # optional extras in the same repo: mmproj (vision, 0.63GB) and the
    # dspark drafter (~2GB, lossless speculative-decoding speedup on cuda).
    "prism-ml/Ternary-Bonsai-27B-gguf": {
        "hf_repo": "prism-ml/Ternary-Bonsai-27B-gguf",
        "hf_file": "Ternary-Bonsai-27B-Q2_0.gguf",
        "llama_cpp_repo": "https://github.com/PrismML-Eng/llama.cpp",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20},  # card
        "est_vram_gb": 8.7,  # 7.2 weights + kv/buffers at 8k ctx
    },
    # q4_k_m is ~7.3GB and comfortable on one t4. the 12GB figure you had
    # matches the Q8_0 file (12.7GB) -- that one is too tight on a single t4
    # at 8k ctx once kv + buffers land on top; if you want q8, give it
    # tensor_split "1,1" and both gpus instead. --jinja applies the gemma 4
    # chat template shipped in the gguf.
    "bartowski/gemma-4-12B-it-GGUF": {
        "hf_repo": "bartowski/gemma-4-12B-it-GGUF",
        "hf_file": "gemma-4-12B-it-Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # gemma defaults
        "est_vram_gb": 9,
    },
    # dual-t4 hot-swap tier -- doesn't fit on one t4, needs both. displaces
    # whatever's resident when it runs (see harness.run). ~20GB q4_k_m split
    # across 2x15GB. alternative if you'd rather keep gpu 1 free: it's an
    # A3B MoE, so one t4 + n_cpu_moe (experts parked in the 30GB system ram)
    # also works, just slower.
    "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF": {
        "hf_repo": "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
        "hf_file": "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": "1,1",
        "n_cpu_moe": None,
        "gpu_devices": [0, 1],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "est_vram_gb": 20,
    },

    # ---- single-t4 tier (file <= ~12GB, gpu 0 only) ---------------------

    # 1-bit companion to bonsai-27b-resident: binary weights, 3.8GB file,
    # ~90% of fp16 quality (card) vs the ternary's 95%. same prismml fork
    # requirement (custom Q1_0_g128 kernels), same t4 smoke-test caveat.
    # optional in-repo extras: mmproj (vision) + dspark drafter.
    "prism-ml/Bonsai-27B-gguf": {
        "hf_repo": "prism-ml/Bonsai-27B-gguf",
        "hf_file": "Bonsai-27B-Q1_0.gguf",
        "llama_cpp_repo": "https://github.com/PrismML-Eng/llama.cpp",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20},  # card
        "est_vram_gb": 5,  # 3.8 weights + 4-bit kv on a hybrid-attn backbone
    },
    # claude-mythos-5 distill on qwen3.5-9b. the advertised 1M context is NOT
    # feasible in 15GB vram -- keep 8k. card sampling: temp 0.6, top-p 0.95,
    # top-k 20, repeat-penalty 1.05. same repo also ships -MTP- twins with the
    # speculative head (quant="MTP-Q4_K_M") and an mmproj for vision.
    "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF": {
        "hf_repo": "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF",
        "hf_file": "Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repeat_penalty": 1.05},
        "est_vram_gb": 7,
    },
    # qwythos v2: looping trained out, greedy decoding stays coherent (card).
    # q6_k because a 9b leaves lots of single-t4 headroom; MTP twins in-repo.
    "empero-ai/Qwythos-9B-v2-GGUF": {
        "hf_repo": "empero-ai/Qwythos-9B-v2-GGUF",
        "hf_file": "Qwythos-9B-v2-Q6_K.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repeat_penalty": 1.05},
        "est_vram_gb": 9,
    },
    # coding/terminal-agentic gemma-4 finetune (~3.5x base on tau2 telecom).
    # author: Q4_K_M is the sweet spot, no Q2 shipped; set rep_pen ~1.1 if
    # output loops. --jinja is required for gemma 4's native tool format.
    "yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF": {
        "hf_repo": "yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF",
        "hf_file": "gemma4-v2-Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.1},
        "est_vram_gb": 9,
    },
    # text-to-3D-mesh llm (llama-3.1-8b base): chat it a shape description,
    # it emits OBJ vertices/faces as plain text.
    "bartowski/LLaMA-Mesh-GGUF": {
        "hf_repo": "bartowski/LLaMA-Mesh-GGUF",
        "hf_file": "LLaMA-Mesh-Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "est_vram_gb": 6.5,
    },
    # official dense 4B sibling of agents-a1 (released 2026-07-14): strong
    # small agentic model. same card sampling as the 35B: temp 0.85,
    # top_p 0.95, top_k 20, presence_penalty 1.1. tiny -> huge ctx headroom.
    "InternScience/Agents-A1-4B-Q4_K_M-GGUF": {
        "hf_repo": "InternScience/Agents-A1-4B-Q4_K_M-GGUF",
        "hf_file": "Agents-A1-4B-Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 0.85, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.1},
        "est_vram_gb": 4,
    },
    # abliterated twin of qwythos-9b-1m (huihui). same 1M-ctx caveat: default
    # 8192, the 1M is not feasible here. card runs MTP flags on these files
    # (--spec-type draft-mtp, n-max 2) -- add via extra_args override if wanted.
    "huihui-ai/Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-GGUF": {
        "hf_repo": "huihui-ai/Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-GGUF",
        "hf_file": "Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-Q4_K.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repeat_penalty": 1.05},
        "est_vram_gb": 7.5,
    },
    # abliterated gemma-4-12b QAT (thinking + non-thinking both abliterated).
    # twin of gemma4-12b in spirit -- qat q4_0-trained weights requantized.
    # card's MTP recipe needs a second drafter file (mtp-ggml-model-bf16.gguf
    # + --spec-draft-model), which the single-file harness doesn't fetch.
    "huihui-ai/Huihui-gemma-4-12B-it-qat-q4_0-unquantized-abliterated-GGUF": {
        "hf_repo": "huihui-ai/Huihui-gemma-4-12B-it-qat-q4_0-unquantized-abliterated-GGUF",
        "hf_file": "Huihui-gemma-4-12B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # gemma defaults
        "est_vram_gb": 9,
    },
    # abliterated gemma-4 coder finetune (mradermacher static quants, note the
    # dot-separated filenames). base entry in spirit: gemma4-12b-agentic.
    "mradermacher/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated-GGUF": {
        "hf_repo": "mradermacher/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated-GGUF",
        "hf_file": "Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated.Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},  # gemma defaults
        "est_vram_gb": 9,
    },

    # 3B agentic model (Q4_K_M ~2.5GB, huge headroom on single T4).
    # default sampling: temp 0.0 for structured JSON extraction tasks
    # default sampling: temp 0.0 for structured JSON extraction tasks
        # (topic inference, deep research analysis). raise temp per-call
        # when you need a bit of creativity instead.
        "owao/Nanbeige4.2-3B-GGUF": {
            "hf_repo": "owao/Nanbeige4.2-3B-GGUF",
            "hf_file": "nanbeige4.2-3b-Q4_K_M.gguf",
            "llama_cpp_repo": "https://github.com/Nanbeige/llama.cpp",
        "ctx": 4096,
        "ngl": 99,
        "tensor_split": None,
        "n_cpu_moe": None,
        "gpu_devices": [0],
        "extra_args": [],
        "sampling": {"temperature": 0.0, "top_p": 0.95, "top_k": 1},
        "est_vram_gb": 3,
    },

    # ---- dual-t4 tier (file <= ~26GB, tensor_split both gpus) -----------

    # reasoning finetune of qwen3.6-27b that answers in ~half the tokens.
    # card: Q4_K_M + MTP self-speculation is their small-footprint pick;
    # their recommended draft length is 4 (accepts ~3.75 tokens/verify).
    "bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF": {
        "hf_repo": "bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF",
        "hf_file": "ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": "1,1",
        "n_cpu_moe": None,
        "gpu_devices": [0, 1],
        "extra_args": ["--jinja", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "est_vram_gb": 19,
    },
    # base qwen3.6-27b with the MTP head kept -- llama.cpp's MTP support
    # gives a big decode speedup with no separate draft model. UD-Q4_K_XL is
    # the card's own serving pick. card says n-max 2 (not 6) for this repo,
    # and notes --mmproj is not yet supported together with MTP.
    "unsloth/Qwen3.6-27B-MTP-GGUF": {
        "hf_repo": "unsloth/Qwen3.6-27B-MTP-GGUF",
        "hf_file": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": "1,1",
        "n_cpu_moe": None,
        "gpu_devices": [0, 1],
        "extra_args": ["--jinja", "-fa", "on", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "est_vram_gb": 20,
    },
    # deepreinforce's swe/terminal coding distill (qwen3.6-35b-a3b base).
    # card serves 262k ctx on big iron; 8k here, raise per-call if needed.
    "deepreinforce-ai/Ornith-1.0-35B-GGUF": {
        "hf_repo": "deepreinforce-ai/Ornith-1.0-35B-GGUF",
        "hf_file": "ornith-1.0-35b-Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": "1,1",
        "n_cpu_moe": None,
        "gpu_devices": [0, 1],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "est_vram_gb": 23,
    },
    # the OFFICIAL agents-a1 quant (qwen35moe arch, 21.2GB single file).
    # never download the bf16 InternScience/Agents-A1 repo (70GB). card
    # sampling: temp 0.85, top_p 0.95, top_k 20, presence_penalty 1.1.
    # optional Agents-A1-mmproj.gguf in-repo adds vision.
    "InternScience/Agents-A1-Q4_K_M-GGUF": {
        "hf_repo": "InternScience/Agents-A1-Q4_K_M-GGUF",
        "hf_file": "Agents-A1-Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": "1,1",
        "n_cpu_moe": None,
        "gpu_devices": [0, 1],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 0.85, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.1},
        "est_vram_gb": 23,
    },
    # hauhaucs uncensored qwen3.6-35b-a3b (0/465 refusals; uncensored twin of
    # qwen3.6-35b-a3b-hotswap). repo has custom K_P "perfect" quants: ~5-15%
    # bigger for 1-2 quant levels of quality -- quant="Q4_K_P" (23.4GB) fits.
    # hf's hardware-compat widget doesn't recognize K_P files, trust
    # list_quants over the widget. optional mmproj-*-f16.gguf adds vision.
    # card wants >=128k ctx to preserve thinking mode -- that kv budget does
    # not exist on 2x t4, so 8-16k it is: thinking mode is degraded, prefer
    # non-thinking (card: temp 0.7, top_p 0.8, top_k 20, presence 1.5).
    "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive": {
        "hf_repo": "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
        "hf_file": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": "1,1",
        "n_cpu_moe": None,
        "gpu_devices": [0, 1],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
        "est_vram_gb": 23,
    },
    # abliterated twin of agents-a1 (huihui). same card sampling as the base
    # entry: temp 0.85, top_p 0.95, top_k 20, presence_penalty 1.1.
    "huihui-ai/Huihui-Agents-A1-abliterated-GGUF": {
        "hf_repo": "huihui-ai/Huihui-Agents-A1-abliterated-GGUF",
        "hf_file": "Agents-A1-abliterated-Q4_K.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": "1,1",
        "n_cpu_moe": None,
        "gpu_devices": [0, 1],
        "extra_args": ["--jinja"],
        "sampling": {"temperature": 0.85, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.1},
        "est_vram_gb": 23,
    },
    # abliterated twin of qwen3.6-27b-mtp (huihui), MTP drafter kept.
    # card invocation: -fa on + draft-mtp with n-max 6 for this repo.
    "huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF": {
        "hf_repo": "huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF",
        "hf_file": "Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": "1,1",
        "n_cpu_moe": None,
        "gpu_devices": [0, 1],
        "extra_args": ["--jinja", "-fa", "on", "--spec-type", "draft-mtp", "--spec-draft-n-max", "6"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "est_vram_gb": 19,
    },
    # abliteration of lordx64's claude-4.7-opus reasoning distill of
    # qwen3.6-35b-a3b (36B qwen35moe, MTP kept, multimodal base -- repo ships
    # mmproj-model-f16.gguf, but note mmproj + MTP don't combine yet).
    # filenames use the -ggml-model-<QUANT> infix and the compat widget
    # misses the Q4_K file -- another reason list_quants beats guessing.
    # fallback: quant="Q2_K" (13.3GB) squeezes onto ONE t4:
    #     run(key, MODELS, quant="Q2_K", tensor_split=None, gpu_devices=[0])
    # Q6_K (29.2GB) is over the dual-t4 budget -- skip it.
    "huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF": {
        "hf_repo": "huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF",
        "hf_file": "Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf",
        "ctx": 8192,
        "ngl": 99,
        "tensor_split": "1,1",
        "n_cpu_moe": None,
        "gpu_devices": [0, 1],
        "extra_args": ["--jinja", "-fa", "on", "--spec-type", "draft-mtp", "--spec-draft-n-max", "6"],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "est_vram_gb": 23.5,
    },
}
