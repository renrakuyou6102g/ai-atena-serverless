import torch
import runpod

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ATENA_MODEL = "/workspace/AI_ATENA_v3"

print("AI ATENA loading...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    ATENA_MODEL
)

model.eval()

print("AI ATENA ready")


def handler(job):

    data = job.get("input", {})

    message = str(
        data.get("message", "")
    ).strip()

    if not message:
        return {
            "ok": False,
            "error": "質問がありません"
        }

    messages = [
        {
            "role": "system",
            "content":
                "あなたはAI ATENAです。"
                "日本語で自然に回答してください。"
                "あなたの名前はAI ATENAです。"
        },
        {
            "role": "user",
            "content": message
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return {
        "ok": True,
        "answer": answer,
        "sources": []
    }


if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })
