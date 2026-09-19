import os
import time
import traceback

import runpod

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ============================================================
# AI ATENA vLLM
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Docker内にモデルを保存する場所
BASE_MODEL_DIR = "/workspace/base_model"

# ATENA v9 LoRA
ATENA_ROOT = "/workspace/atena_v9"

MAX_NEW_TOKENS = 96

MAX_MODEL_LEN = 2048


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
あなたはAI ATENAです。
JIGUZAGAのAIアシスタントとして自然な日本語で回答してください。

JIGUZAGAはショート動画、LIVE、AI、ショッピング等を統合したサービスです。
JIGUMAPは位置情報付き動画やライブ情報を扱う地図機能です。

予約機能は現在未実装です。
確認できない情報を作り話で補わないでください。
ユーザーの現在の質問を最優先してください。
""".strip()


# ============================================================
# LoRA検索
# ============================================================

def find_adapter_dir(root: str):

    if not os.path.isdir(root):
        return None

    for current_root, dirs, files in os.walk(root):

        if "adapter_config.json" in files:
            return current_root

    return None


ADAPTER_DIR = find_adapter_dir(
    ATENA_ROOT
)

if ADAPTER_DIR is None:

    raise RuntimeError(
        f"ATENA LoRAが見つかりません: {ATENA_ROOT}"
    )


print("=" * 60)
print("AI ATENA vLLM STARTING")
print("MODEL:", BASE_MODEL_DIR)
print("LORA:", ADAPTER_DIR)
print("=" * 60)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(

    BASE_MODEL_DIR,

    trust_remote_code=True,

    local_files_only=True
)


# ============================================================
# vLLM
# ============================================================
#
# 24GB GPU以上推奨
#
# FP16 Qwen2.5 7B
# +
# LoRA
# +
# KV Cache
#
# ============================================================

load_start = time.time()


llm = LLM(

    model=BASE_MODEL_DIR,

    tokenizer=BASE_MODEL_DIR,

    dtype="float16",

    trust_remote_code=True,

    enable_lora=True,

    max_lora_rank=64,

    max_model_len=MAX_MODEL_LEN,

    gpu_memory_utilization=0.90,

    enable_prefix_caching=True,

    max_num_seqs=8,

    disable_log_stats=True,

    enforce_eager=False,
)


ATENA_LORA = LoRARequest(

    "atena-v9",

    1,

    ADAPTER_DIR
)


print(
    "MODEL LOAD SEC:",
    round(time.time() - load_start, 2)
)

print("=" * 60)
print("AI ATENA vLLM READY")
print("=" * 60)


# ============================================================
# 入力取得
# ============================================================

def get_question(job_input):

    # prompt
    prompt = job_input.get("prompt")

    if (
        isinstance(prompt, str)
        and prompt.strip()
    ):
        return prompt.strip()


    # message
    message = job_input.get("message")

    if (
        isinstance(message, str)
        and message.strip()
    ):
        return message.strip()


    # messages
    messages = job_input.get("messages")

    if isinstance(messages, list):

        for msg in reversed(messages):

            if not isinstance(msg, dict):
                continue

            if msg.get("role") != "user":
                continue

            content = msg.get(
                "content",
                ""
            )

            if (
                isinstance(content, str)
                and content.strip()
            ):

                return content.strip()


    return ""


# ============================================================
# Prompt
# ============================================================

def build_prompt(question):

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": question
        }

    ]


    prompt = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True
    )


    return prompt


# ============================================================
# Generate
# ============================================================

def generate_answer(question):

    prompt = build_prompt(
        question
    )


    token_count = len(
        tokenizer.encode(
            prompt,
            add_special_tokens=False
        )
    )


    print(
        "INPUT TOKENS:",
        token_count
    )


    # ------------------------------------
    # 短文中心の高速設定
    # ------------------------------------

    sampling_params = SamplingParams(

        temperature=0.0,

        max_tokens=MAX_NEW_TOKENS,

        repetition_penalty=1.0,

        stop_token_ids=[
            tokenizer.eos_token_id
        ]
    )


    start = time.time()


    outputs = llm.generate(

        [prompt],

        sampling_params,

        lora_request=ATENA_LORA
    )


    generation_sec = (
        time.time()
        - start
    )


    answer = (
        outputs[0]
        .outputs[0]
        .text
        .strip()
    )


    generated_tokens = len(
        outputs[0]
        .outputs[0]
        .token_ids
    )


    print(
        "GENERATED TOKENS:",
        generated_tokens
    )

    print(
        "GENERATION SEC:",
        round(
            generation_sec,
            2
        )
    )


    if generation_sec > 0:

        print(
            "TOKENS/SEC:",
            round(
                generated_tokens
                / generation_sec,
                2
            )
        )


    return answer


# ============================================================
# RunPod Handler
# ============================================================

def handler(job):

    job_start = time.time()


    try:

        print("=" * 60)

        print(
            "JOB ID:",
            job.get(
                "id",
                "unknown"
            )
        )


        job_input = job.get(
            "input",
            {}
        )


        if not isinstance(
            job_input,
            dict
        ):

            job_input = {}


        question = get_question(
            job_input
        )


        if not question:

            return {

                "success": False,

                "error":
                    "質問が入力されていません。"
            }


        print(
            "USER:",
            question[:500]
        )


        answer = generate_answer(
            question
        )


        total_sec = (
            time.time()
            - job_start
        )


        print(
            "ANSWER:",
            answer[:1000]
        )

        print(
            "TOTAL SEC:",
            round(
                total_sec,
                2
            )
        )

        print("=" * 60)


        return {

            "success": True,

            "answer": answer,

            "model":
                "AI ATENA 7B v9 vLLM",

            "generation_seconds":
                round(total_sec, 2)
        }


    except Exception as e:

        traceback.print_exc()


        return {

            "success": False,

            "error": str(e),

            "model":
                "AI ATENA 7B v9 vLLM"
        }


# ============================================================
# RunPod Serverless
# ============================================================

runpod.serverless.start(

    {

        "handler": handler

    }

)
