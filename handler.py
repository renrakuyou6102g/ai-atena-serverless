import os
import time
import traceback

import runpod

from transformers import AutoTokenizer

from vllm import (
    LLM,
    SamplingParams,
)

from vllm.lora.request import LoRARequest


# ============================================================
# AI ATENA 設定
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

ATENA_ROOT = "/workspace/atena_v9"

MAX_NEW_TOKENS = 96

MAX_MODEL_LEN = 2048


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
あなたはAI ATENAです。

JIGUZAGAのAIアシスタントとして、
自然で分かりやすい日本語で回答してください。

JIGUZAGAは、
ショート動画、LIVE配信、AI、ショッピングなどを
統合したプラットフォームです。

JIGUMAPは、
位置情報付き動画やライブ情報などを扱う
JIGUZAGAの地図機能です。

予約機能は現在実装されていません。

確認できない情報を作り話で補わないでください。

ユーザーの現在の質問を最優先してください。
""".strip()


# ============================================================
# LoRAフォルダ検索
# ============================================================

def find_adapter_dir(root):

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
        "ATENA v9のadapter_config.jsonが見つかりません。"
    )


print("=" * 70)

print(
    "AI ATENA vLLM STARTING"
)

print(
    "BASE MODEL:",
    MODEL_NAME
)

print(
    "ATENA LORA:",
    ADAPTER_DIR
)

print("=" * 70)


# ============================================================
# Tokenizer
# ============================================================

print(
    "Loading tokenizer..."
)


tokenizer = AutoTokenizer.from_pretrained(

    MODEL_NAME,

    trust_remote_code=True
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token_id = (
        tokenizer.eos_token_id
    )


# ============================================================
# vLLM
# ============================================================
#
# 初回Worker起動時だけ
# Qwen2.5-7BをHugging Faceから取得します。
#
# Active Worker = 1なら
# モデルを読み込んだ状態を維持できます。
#
# ============================================================

print(
    "Loading Qwen2.5-7B with vLLM..."
)


model_load_start = time.time()


llm = LLM(

    model=MODEL_NAME,

    tokenizer=MODEL_NAME,

    trust_remote_code=True,

    dtype="float16",

    download_dir="/workspace/huggingface",

    max_model_len=MAX_MODEL_LEN,

    gpu_memory_utilization=0.90,

    enable_prefix_caching=True,

    enable_lora=True,

    max_lora_rank=64,

    max_num_seqs=8
)


print(
    "MODEL LOAD:",
    round(
        time.time()
        - model_load_start,
        2
    ),
    "sec"
)


# ============================================================
# LoRA
# ============================================================

ATENA_LORA = LoRARequest(

    "atena-v9",

    1,

    ADAPTER_DIR
)


print("=" * 70)

print(
    "AI ATENA vLLM READY"
)

print("=" * 70)


# ============================================================
# 入力取得
# ============================================================

def get_question(job_input):

    # --------------------------------
    # prompt
    # --------------------------------

    prompt = job_input.get(
        "prompt"
    )

    if (
        isinstance(prompt, str)
        and prompt.strip()
    ):

        return prompt.strip()


    # --------------------------------
    # message
    # --------------------------------

    message = job_input.get(
        "message"
    )

    if (
        isinstance(message, str)
        and message.strip()
    ):

        return message.strip()


    # --------------------------------
    # messages
    # --------------------------------

    messages = job_input.get(
        "messages"
    )

    if isinstance(
        messages,
        list
    ):

        for msg in reversed(
            messages
        ):

            if not isinstance(
                msg,
                dict
            ):

                continue


            if msg.get(
                "role"
            ) != "user":

                continue


            content = msg.get(
                "content",
                ""
            )


            if (
                isinstance(
                    content,
                    str
                )
                and content.strip()
            ):

                return content.strip()


    return ""


# ============================================================
# Prompt作成
# ============================================================

def build_prompt(
    question
):

    messages = [

        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT
        },

        {
            "role":
                "user",

            "content":
                question
        }

    ]


    prompt = (
        tokenizer
        .apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True
        )
    )


    return prompt


# ============================================================
# Generate
# ============================================================

def generate_answer(
    question
):

    prompt = build_prompt(
        question
    )


    # --------------------------------
    # 入力トークン数
    # --------------------------------

    input_ids = (
        tokenizer.encode(

            prompt,

            add_special_tokens=False
        )
    )


    print(
        "INPUT TOKENS:",
        len(input_ids)
    )


    # --------------------------------
    # 高速生成設定
    # --------------------------------

    sampling_params = SamplingParams(

        temperature=0.0,

        max_tokens=
            MAX_NEW_TOKENS,

        repetition_penalty=1.0
    )


    generation_start = (
        time.time()
    )


    outputs = llm.generate(

        [prompt],

        sampling_params,

        lora_request=
            ATENA_LORA
    )


    generation_time = (

        time.time()
        - generation_start
    )


    # --------------------------------
    # 回答
    # --------------------------------

    result = (
        outputs[0]
        .outputs[0]
    )


    answer = (
        result
        .text
        .strip()
    )


    generated_tokens = len(
        result.token_ids
    )


    print(
        "GENERATED TOKENS:",
        generated_tokens
    )


    print(
        "GENERATION TIME:",
        round(
            generation_time,
            2
        ),
        "sec"
    )


    if generation_time > 0:

        print(
            "TOKENS / SEC:",
            round(

                generated_tokens
                / generation_time,

                2
            )
        )


    return (
        answer,
        generation_time,
        generated_tokens
    )


# ============================================================
# RunPod Handler
# ============================================================

def handler(
    job
):

    total_start = (
        time.time()
    )


    try:

        print("=" * 70)


        job_id = job.get(
            "id",
            "unknown"
        )


        print(
            "JOB ID:",
            job_id
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

                "success":
                    False,

                "error":
                    "質問が入力されていません。"
            }


        print(
            "USER:",
            question[:1000]
        )


        (
            answer,
            generation_time,
            generated_tokens

        ) = generate_answer(
            question
        )


        total_time = (

            time.time()
            - total_start
        )


        print(
            "ATENA:",
            answer[:1500]
        )


        print(
            "TOTAL TIME:",
            round(
                total_time,
                2
            ),
            "sec"
        )


        print("=" * 70)


        return {

            "success":
                True,

            "answer":
                answer,

            "model":
                "AI ATENA 7B v9 vLLM",

            "generation_seconds":
                round(
                    generation_time,
                    2
                ),

            "total_seconds":
                round(
                    total_time,
                    2
                ),

            "generated_tokens":
                generated_tokens
        }


    except Exception as e:

        print(
            "ATENA ERROR:"
        )

        traceback.print_exc()


        return {

            "success":
                False,

            "error":
                str(e),

            "model":
                "AI ATENA 7B v9 vLLM"
        }


# ============================================================
# RunPod Serverless
# ============================================================

runpod.serverless.start(

    {

        "handler":
            handler

    }

)
