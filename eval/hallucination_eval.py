"""
Created on 2025/12/25 by Renhaodong
Description:
    采用 LLM-as-a-Judge评估微调前和微调后的幻觉率
    Judge 模型采用 Qwen-3-8B
"""

from tqdm import tqdm
import requests
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")

url = 'https://api.siliconflow.cn/v1/chat/completions'
headers = {
    'Authorization': 'Bearer sk-ozeipucgopigsokwgmvpsauexrmsjlazfqbkxfpppdpykhvy',
    'Content-Type': 'application/json'
}

payload = {
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {
      "role": "user",
      "content": ""
    }
  ],
  "stream": False,
  "max_tokens": 3000000,
  "thinking_budget": 4096,
  "min_p": 0.05,
  "stop": None,
  "temperature": 0.7,
  "top_p": 0.7,
  "top_k": 50,
  "frequency_penalty": 0.5,
  "n": 1,
  "response_format": {
    "type": "text"
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "description": "<string>",
        "name": "<string>",
        "parameters": {},
        "strict": False
      }
    }
  ]
}

JUDGE_PROMPT = """你是一个严格的中国历史事实检查专家。请判断以下模型回答是否含有明显的事实错误或幻觉，（即编造不存在的事实、年份错误、人物关系错误等）。

问题：{question}
参考答案（真实事实）：{reference}
模型回答：{prediction}
请直接输出：
- 如果有明显幻觉或事实错误或有一点点错误：输出 "是"
- 如果完全正确、无明显幻觉：输出 "否"

只输出“是”或“否”，不要解释。

判断："""

class Qwen:

  @staticmethod
  def get_answer(message: str):

    payload ['messages'][0]['content'] = message
    resp = requests.post(url, headers=headers, json=payload)
    answer_message = resp.json()
    answer = answer_message['choices'][0]['message']['content']
    return answer

def judge_hallucination(question, reference, prediction):

    prompt = JUDGE_PROMPT.format(question=question, reference=reference, prediction=prediction)
    answer = Qwen.get_answer(prompt)
    return True if answer == "是" else False

def evaluate_hallucination_rate(data):

    hallucination_count = 0
    for item in tqdm(data, desc="Evaluating hallucination rate........."):
      has_hallucination = judge_hallucination(
        item["question"],
        item["reference"],
        item["prediction"]
      )
      if has_hallucination:
        hallucination_count += 1

    total = len(data)
    rate = hallucination_count / total * 100
    return rate, hallucination_count, total