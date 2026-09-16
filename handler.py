import os
import traceback

import torch
import runpod

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# パス
# ============================================================

BASE_MODEL_DIR = "/workspace/base_model"
ATENA_ROOT = "/workspace/atena_v9"

MAX_NEW_TOKENS = 300


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
あなたはAI ATENAです。
必ず自然な日本語で回答してください。

【ATENA】
AI ATENAは、JIGUZAGAで利用できるAIアシスタントとして
開発されています。

JIGUZAGA内で、
質問対応、
Web検索、
文章作成、
要約などを支援します。

【JIGUZAGA】
JIGUZAGAは、
ショート動画、
LIVE配信、
AI、
ショッピングなどを
統合したプラットフォームです。

株価予測だけを目的とした
投資サービスではありません。

【JIGUMAP】
JIGUMAPは、
位置情報付き動画やライブ情報などを
地図上で扱うための
JIGUZAGAのマップ機能です。

【制限】
予約機能は現在実装されていません。

予約を実行した、
予約できる、
などと回答しないでください。

xAI、
Alibaba、
アリババクラウド、
NTTドコモ、
証券取引所などを
ATENAの運営元として勝手に作らないでください。

確認できない情報を
作り話で補わないでください。
"""


# ============================================================
# LoRAディレクトリ検索
# ============================================================

def find_adapter_dir(root):

    for current_root, dirs, files in os.walk(root):

        if "adapter_config.json" in files:
            return current_root

    return None


ADAPTER_DIR = find_adapter_dir(ATENA_ROOT)

if ADAPTER_DIR is None:

    raise RuntimeError(
        "ATENA v9のadapter_config.jsonが見つかりません。"
    )


print("=" * 60)
print("AI ATENA 7B v9 STARTING")
print("BASE MODEL:", BASE_MODEL_DIR)
print("ADAPTER:", ADAPTER_DIR)
print("=" * 60)


# ============================================================
# Tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True
)


# ============================================================
# 4bit
# ============================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,
)


# ============================================================
# Base Model
# ============================================================

print("Loading local Qwen 7B...")

base_model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL_DIR,

    local_files_only=True,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=torch.float16,

    low_cpu_mem_usage=True,

    trust_remote_code=True,
)


# ============================================================
# LoRA
# ============================================================

print("Loading ATENA v9 LoRA...")

model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_DIR,

    is_trainable=False,
)

model.eval()

model.config.use_cache = True


print("=" * 60)
print("AI ATENA 7B v9 READY")
print("=" * 60)


# ============================================================
# 入力取得
# ============================================================

def get_question(job_input):

    prompt = job_input.get("prompt")

    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()


    message = job_input.get("message")

    if isinstance(message, str) and message.strip():
        return message.strip()


    messages = job_input.get("messages")

    if isinstance(messages, list):

        for msg in reversed(messages):

            if not isinstance(msg, dict):
                continue

            if msg.get("role") == "user":

                content = msg.get("content", "")

                if isinstance(content, str):
                    return content.strip()

    return ""


# ============================================================
# 推論
# ============================================================

def generate_answer(question):

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


    inputs = tokenizer(

        prompt,

        return_tensors="pt"
    )


    device = next(model.parameters()).device


    inputs = {

        key: value.to(device)

        for key, value in inputs.items()

    }


    with torch.inference_mode():

        output = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            repetition_penalty=1.1,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=tokenizer.eos_token_id,
        )


    new_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]


    answer = tokenizer.decode(

        new_tokens,

        skip_special_tokens=True
    )


    return answer.strip()


# ============================================================
# RunPod Handler
# ============================================================

def handler(job):

    try:

        job_input = job.get(
            "input",
            {}
        )


        question = get_question(
            job_input
        )


        if not question:

            return {

                "success": False,

                "error": "質問が入力されていません。"

            }


        print("USER:", question)


        answer = generate_answer(
            question
        )


        print("ATENA:", answer)


        return {

            "success": True,

            "answer": answer,

            "model": "AI ATENA 7B v9"

        }


    except Exception as e:

        traceback.print_exc()


        return {

            "success": False,

            "error": str(e)

        }


# ============================================================
# Serverless
# ============================================================

runpod.serverless.start(

    {

        "handler": handler

    }

)
