"""
Created on 2025/12/25 by Renhaodong
Description:
    该文件用于对微调前的QWen-2.5-7B-Instruct和微调后的模型进行评估
    评估标准包括：准确率(precision),F1,幻觉率等
"""
import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # ← 加这行
from eval.precision_F1_eval import evaluate_precision_F1
from eval.hallucination_eval import evaluate_hallucination_rate
import matplotlib.pyplot as plt
import numpy as np

BASE_MODEL_PATH = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"
LORA_MODEL_PATH = "./merged_qwen_history"
TEST_DATA_PATH = "./data_extract/chinese_history_test.json"

MODELS = {
    "baseline": BASE_MODEL_PATH,
    "after Lora": LORA_MODEL_PATH
}

with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

if __name__ == '__main__':

    print("-----------------------Evaluating baseline-----------------------")
    base_precision, base_f1, base_data = evaluate_precision_F1(MODELS['baseline'], test_data)
    print("precision:{}, F1:{}".format(base_precision, base_f1))
    print("-----------------------Evaluating Lora Model-----------------------")
    Lora_precision, Lora_f1, Lora_data = evaluate_precision_F1(MODELS['after Lora'], test_data)
    print("precision:{}, F1:{}".format(Lora_precision, Lora_f1))

    print("-----------------------Evaluating baseline hallucination-----------------------")
    base_rate, base_hallucination_count, total = evaluate_hallucination_rate(base_data)
    print("baseline ： 幻觉率：{}, 有幻觉的测试集数：{}".format(base_rate, base_hallucination_count))

    print("-----------------------Evaluating Lora hallucination-----------------------")
    Lora_rate, Lora_hallucination_count, total = evaluate_hallucination_rate(Lora_data)
    print("baseline ： 幻觉率：{}, 有幻觉的测试集数：{}".format(Lora_rate, Lora_hallucination_count))

    metrics = ['Accuracy', 'BERTScore F1', 'Hallucination Rate']
    base_values = [base_precision, base_f1, base_rate]
    lora_values = [Lora_precision, Lora_f1, Lora_rate]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    # 绘制柱状图
    bars1 = ax.bar(x - width / 2, base_values, width, label='Baseline (Before Fine-tuning)', color='#1f77b4', alpha=0.9)
    bars2 = ax.bar(x + width / 2, lora_values, width, label='LoRA Fine-tuned (After)', color='#ff7f0e', alpha=0.9)
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}' if 'Rate' in metrics[bars.index(bar)] else f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    add_value_labels(bars1)
    add_value_labels(bars2)
    ax.set_xlabel('Evaluation Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Comparison: Before vs After LoRA Fine-tuning\n(Chinese History RAG Task)', fontsize=14,
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05)  # 留出顶部空间显示标签
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("./img/evaluation_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("柱状图已保存至 ./img/evaluation_comparison.png")