"""
Created on 2025/12/23 by Renhaodong
Description:
    用于对微调后的模型进行合并。
"""
from peft import AutoPeftModelForCausalLM
import torch

OUTPUT_DIR = '/qwen_history_lora/checkpoint-12_23'
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
merged_model = merged_model.merge_and_unload()  # 合并 LoRA
merged_model.save_pretrained("./merged_qwen_history")