"""
Created on 2025/12/25 by Renhaodong
Description:
    该文件用于对微调前的QWen-2.5-7B-Instruct和微调后的模型进行评估
    评估标准包括：准确率(precision),F1,幻觉率等
"""
import json
from eval.precision_F1_eval import eval
from eval.hallucination_eval import evaluate_hallucination_rate
import matplotlib.pyplot as plt
import numpy as np

BASE_DATA_PATH = './data_extract/base_test_inference.json'
LORA_DATA_PATH = './data_extract/lora_test_inference.json'

with open(BASE_DATA_PATH, 'r', encoding='utf-8') as f:
    base_data = json.load(f)

with open(LORA_DATA_PATH, "r", encoding="utf-8") as f:
    lora_data = json.load(f)

if __name__ == '__main__':

    print("-----------------------Evaluating baseline-----------------------")
    base_precision, base_recall, base_f1 = eval(base_data)
    print("base precision:{}, recall:{}, F1:{}".format(base_precision, base_recall, base_f1))
    print("-----------------------Evaluating Lora Model-----------------------")
    lora_precision, lora_recall, lora_f1 = eval(lora_data)
    print("lora precision:{}, recall:{}, F1:{}".format(lora_precision, lora_recall, lora_f1))

    print("-----------------------Evaluating baseline hallucination-----------------------")
    base_rate, base_hallucination_count, total = evaluate_hallucination_rate(base_data)
    print("baseline ： 幻觉率：{}, 有幻觉的测试集数：{}".format(base_rate, base_hallucination_count))

    print("-----------------------Evaluating Lora hallucination-----------------------")
    Lora_rate, Lora_hallucination_count, total = evaluate_hallucination_rate(lora_data)
    print("Lora ： 幻觉率：{}, 有幻觉的测试集数：{}".format(Lora_rate, Lora_hallucination_count))

    metrics = ['precision', 'recall', 'f1', 'Hallucination Rate']
    base_values = [base_precision, base_recall, base_f1, base_rate]
    lora_values = [lora_precision, lora_recall, lora_f1, Lora_rate]

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