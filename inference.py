import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import gc
import os
import json
import random

def generage_answers(model_path:str, test_data:str, save_path:str) -> None:

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = "auto",
        torch_dtype = torch.bfloat16,
        trust_remote_code = True
    )

    results = []
    for item in tqdm.tqdm(test_data, desc=f"Generating answers for {os.path.basename(model_path)}"):
        prompt = (
            "你是中国历史领域专家。请严格根据真实历史事实回答以下问题。\n"
            "要求：\n"
            "- 只使用已确认的历史知识，不允许任何杜撰、虚构或推测。\n"
            "- 如果不确定或信息不足，直接回答“无法确认”或“我不知道”。\n"
            "- 回答要简洁、准确、客观。\n\n"
            f"问题：{item['instruction']}"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,  # 确定性输出，便于评估
                pad_token_id=tokenizer.eos_token_id
            )
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response[len(prompt):].strip()
        results.append({
            "question": item["instruction"],
            "reference": item["output"],
            "prediction": answer
        })

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':

    BASE_MODEL_PATH = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"
    LORA_MODEL_PATH = "./merged_qwen_history"
    TEST_DATA_PATH = "./data_extract/test_qa.json"

    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_data = random.sample(test_data, 150)
    generage_answers(BASE_MODEL_PATH, TEST_DATA_PATH, './data_extract/base_test_inference.json')
    generage_answers(LORA_MODEL_PATH, TEST_DATA_PATH, './data_extract/lora_test_inference.json')