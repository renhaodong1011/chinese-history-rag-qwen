"""
Created on 2025/12/25 by Renhaodong
Description:
    编写函数用于调用计算模型：准确率(precision),F1
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
from bert_score import score
import os
import gc
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # ← 加这行

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")

def compute_bertscore(data):

    preds = [item["prediction"] for item in data]
    refs = [item["reference"] for item in data]

    P, R, F1 = score(preds, refs, lang="zh", verbose=True)
    return P.mean().item(), F1.mean().item()

def generage_answers(model_path:str, test_data:str) -> []:

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = "auto",
        torch_dtype = torch.bfloat16,
        trust_remote_code = True
    )

    results = []
    for item in tqdm.tqdm(test_data, desc=f"Generating answers for {os.path.basename(model_path)}"):
        prompt = f"请准确回答以下问题：\n{item['question']}"
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
            "question": item["question"],
            "reference": item["answer"],
            "prediction": answer
        })
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return results

def evaluate_precision_F1(model_path:str, test_data:str):

    data = generage_answers(model_path, test_data)
    precision, f1 =  compute_bertscore(data)
    return precision, f1, data