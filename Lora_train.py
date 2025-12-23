import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import matplotlib.pyplot as plt  # 新增：绘图库
import numpy as np           # 新增

# ==================== 配置区 ====================
MODEL_PATH = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"
DATASET_PATH = "./data_extract/chinese_history_qa.json"
OUTPUT_DIR = "./qwen_history_lora"
MAX_SEQ_LENGTH = 512

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# LoRA 参数
R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# 训练超参数
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100
LOGGING_STEPS = 10        # 每 10 步记录一次
SAVE_STEPS = 200
EVAL_STEPS = 100          # 新增：每 100 步评估一次（可选）

# 新增：用于记录指标的列表
train_losses = []
eval_losses = []  # 如果你想加验证集的话
steps = []

# ================================================

# 1-2. 加载 tokenizer、模型、LoRA 配置（保持不变）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()

# 3-4. 数据加载和预处理（保持不变）
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def preprocess_function(examples):
    instructions = examples['instruction']
    inputs = examples.get('input', [""] * len(instructions))  # 防止 input 列不存在
    outputs = examples['output']

    texts = []
    for instr, inp, out in zip(instructions, inputs, outputs):
        text = f"### 指令：\n{instr}\n\n### 输入：\n{inp}\n\n### 回答：\n{out}"
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    tokenized["labels"] = [labels for labels in tokenized["input_ids"]]  # 复制 labels
    return tokenized

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==================== 新增：自定义回调记录 loss ====================
from transformers import TrainerCallback

class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            train_losses.append(logs["loss"])
            steps.append(state.global_step)
            print(f"Step {state.global_step} - Loss: {logs['loss']:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            eval_losses.append(metrics["eval_loss"])
            print(f"Step {state.global_step} - Eval Loss: {metrics['eval_loss']:.4f}")

# 实例化回调
loss_callback = LossLoggingCallback()

# ==================== 训练参数（新增 eval_strategy） ====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,                    # 新增：评估频率
    evaluation_strategy="steps",              # 新增：开启评估
    save_strategy="steps",
    load_best_model_at_end=True,              # 新增：加载最佳模型
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",
    # optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
)

# ==================== Trainer（添加 callbacks） ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(min(1000, len(tokenized_dataset)))),  # 用前1000条做验证集（避免太慢）
    data_collator=data_collator,
    callbacks=[loss_callback],  # 新增：添加回调
)

# ==================== 开始训练 ====================
print("开始训练...")
trainer.train()

# ==================== 保存 LoRA ====================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA 权重已保存到 {OUTPUT_DIR}")

# ==================== 绘制 loss 曲线 ====================
if len(train_losses) > 0:
    plt.figure(figsize=(12, 5))

    # 训练 loss
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_losses, label="Train Loss", marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 验证 loss（如果有）
    if eval_losses:
        eval_steps = list(range(EVAL_STEPS, len(steps)*LOGGING_STEPS, EVAL_STEPS))
        plt.subplot(1, 2, 2)
        plt.plot(eval_steps[:len(eval_losses)], eval_losses, label="Eval Loss", color="orange", marker='s')
        plt.title("Evaluation Loss")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.show()
    print(f"Loss 曲线已保存到 {OUTPUT_DIR}/loss_curve.png")