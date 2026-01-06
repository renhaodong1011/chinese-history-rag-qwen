"""
Created on 2025/12/25 by Renhaodong
Description:
    采用 LLM-as-a-Judge评估微调前和微调后的幻觉率
    Judge 模型采用 Qwen-3-8B
"""
from tqdm import tqdm
import requests
import warnings
from settings import payload
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")

url = 'https://api.siliconflow.cn/v1/chat/completions'
headers = {
    'Authorization': 'Bearer sk-ozeipucgopigsokwgmvpsauexrmsjlazfqbkxfpppdpykhvy',
    'Content-Type': 'application/json'
}

JUDGE_PROMPT = """你是一个严谨的历史事实核查专家。只需判断模型回答是否在历史事实上与参考答案冲突。
严格检查以下类型错误：
1. 时间、朝代错误
2. 人物姓名、身份、关系错误
3. 事件存在性、过程或结果错误
4. 与史实不符的因果关系
规则：
- 只有当模型回答中明确出现与参考答案冲突的具体事实时，才输出“是”。
- 表述不同但事实一致、缺少细节但无错、合理推测（带“可能”等词）均输出“否”。
- 不需要任何的多余解释。
- 不要太严格，只要有严重事实错误的时候，才输出“是”。
问题：{question}
模型回答：{prediction}
直接输出：是 或 否 不要输出其余多余的内容"""

class Qwen:

  @staticmethod
  def get_answer(message: str):

    payload ['messages'][0]['content'] = message
    resp = requests.post(url, headers=headers, json=payload)
    answer_message = resp.json()
    answer = answer_message['choices'][0]['message']['content']
    return answer

def judge_hallucination(question, prediction):

    prompt = JUDGE_PROMPT.format(question=question, prediction=prediction)
    answer = Qwen.get_answer(prompt)
    return True if answer == "是" else False

def evaluate_hallucination_rate(data):

    hallucination_count = 0
    for item in tqdm(data, desc="Evaluating hallucination rate........."):

      has_hallucination = judge_hallucination(
        item["question"],
        item["prediction"]
      )
      if has_hallucination:
        hallucination_count += 1

    total = len(data)
    rate = hallucination_count / total
    return rate, hallucination_count, total