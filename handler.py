import os
import json
import zipfile
import urllib.request
import shutil
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
# ATENA 設定
# ============================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen2.5-7B-Instruct"
)

MODEL_ZIP_URL = os.getenv(
    "MODEL_ZIP_URL",
    ""
)

ZIP_PATH = "/workspace/ATENA_7B_v9.zip"

EXTRACT_DIR = "/workspace/atena_model"

MAX_NEW_TOKENS = int(
    os.getenv("MAX_NEW_TOKENS", "300")
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
あなたはAI ATENAです。
必ず自然な日本語で回答してください。

【最重要】
AI ATENAは、JIGUZAGAで利用できるAIアシスタントとして開発されています。

ユーザーから、
「ATENAとは？」
「あなたは誰？」
「JIGUZAGAとATENAの関係は？」
「ATENAはJIGUZAGAの何？」
などと聞かれた場合は、次の事実を優先してください。

AI ATENAは、JIGUZAGAで利用できるAIアシスタントとして開発されています。
JIGUZAGA内で、質問対応、Web検索、文章作成、要約などを支援します。

【JIGUZAGA】
JIGUZAGAは、
ショート動画、
LIVE配信、
AI、
ショッピング
などを統合したプラットフォームです。

株価予測だけを目的とした投資サービスではありません。

【JIGUMAP】
JIGUMAPは、
位置情報付き動画やライブ情報などを
地図上で扱うためのJIGUZAGAのマップ機能です。

【現在の制限】
予約機能は現在実装されていません。

予約を実行した、
予約できる、
などと答えないでください。

xAI、
Alibaba、
アリババクラウド、
NTTドコモ、
証券取引所、
その他の企業がATENAを運営しているという情報を
勝手に作らないでください。

最新情報が必要な質問については、
確認できない内容を作り話で補わないでください。
"""


# ============================================================
# ZIPダウンロード
# ============================================================

def download_model_zip():

    if os.path.exists(ZIP_PATH):
        print("ATENA ZIP already exists.")
        return

    if not MODEL_ZIP_URL:
        raise RuntimeError(
            "MODEL_ZIP_URL が設定されていません。"
        )

    print("Downloading ATENA v9 ZIP...")
    print("URL:", MODEL_ZIP_URL)

    urllib.request.urlretrieve(
        MODEL_ZIP_URL,
        ZIP_PATH
    )

    print(
        "ZIP downloaded:",
        os.path.getsize(ZIP_PATH),
        "bytes"
    )


# ============================================================
# ZIP展開
# ============================================================

def extract_model():

    adapter_config = find_adapter_directory(
        EXTRACT_DIR
    )

    if adapter_config:
        print(
            "ATENA model already extracted:",
            adapter_config
        )

        return adapter_config

    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)

    os.makedirs(
        EXTRACT_DIR,
        exist_ok=True
    )

    print("Extracting ATENA v9...")

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as z:

        z.extractall(
            EXTRACT_DIR
        )

    adapter_dir = find_adapter_directory(
        EXTRACT_DIR
    )

    if not adapter_dir:
        raise RuntimeError(
            "adapter_config.json がZIP内に見つかりません。"
        )

    print(
        "Adapter directory:",
        adapter_dir
    )

    return adapter_dir


# ============================================================
# adapter_config.json を探す
# ============================================================

def find_adapter_directory(root):

    if not os.path.exists(root):
        return None

    for current_root, dirs, files in os.walk(root):

        if "adapter_config.json" in files:

            return current_root

    return None


# ============================================================
# ATENA読み込み
# ============================================================

print("=" * 60)
print("Starting AI ATENA 7B v9")
print("=" * 60)

download_model_zip()

ADAPTER_PATH = extract_model()


# ============================================================
# Tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)


# ============================================================
# 4bit設定
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,
)


# ============================================================
# Base model
# ============================================================

print("Loading Qwen 7B base model...")

base_model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=torch.float16,

    low_cpu_mem_usage=True,

    trust_remote_code=True,
)


# ============================================================
# LoRA v9
# ============================================================

print(
    "Loading ATENA v9 LoRA:",
    ADAPTER_PATH
)

model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_PATH,

    is_trainable=False,
)

model.eval()

model.config.use_cache = True


print("=" * 60)
print("AI ATENA 7B v9 READY")
print("=" * 60)


# ============================================================
# ユーザー入力取得
# ============================================================

def get_user_message(job_input):

    # prompt
    prompt = job_input.get("prompt")

    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    # message
    message = job_input.get("message")

    if isinstance(message, str) and message.strip():
        return message.strip()

    # messages
    messages = job_input.get("messages")

    if isinstance(messages, list):

        for message in reversed(messages):

            if not isinstance(message, dict):
                continue

            if message.get("role") == "user":

                content = message.get(
                    "content",
                    ""
                )

                if isinstance(content, str):
                    return content.strip()

    return ""


# ============================================================
# ATENA 推論
# ============================================================

def generate_answer(question):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    text = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True,
    )

    inputs = tokenizer(

        text,

        return_tensors="pt",
    )

    device = next(
        model.parameters()
    ).device

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
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

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True,
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

        question = get_user_message(
            job_input
        )

        if not question:

            return {
                "success": False,
                "error": "質問が入力されていません。"
            }

        print(
            "USER:",
            question
        )

        answer = generate_answer(
            question
        )

        print(
            "ATENA:",
            answer
        )

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
# Serverless起動
# ============================================================

runpod.serverless.start(
    {
        "handler": handler
    }
)
