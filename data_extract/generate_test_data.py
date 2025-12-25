"""
Created on 2025/12/25 by RenHaodong
Description:
    用于从爬取的文件中，生成test数据集
"""
import requests
from typing import *
import json
import os
import time
from random import randint

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

all_QA_data = []

class ExtractAnswer:

    @ staticmethod
    def extract(json_text : Union[dict, list]):

      if isinstance(json_text, dict):
        if len(json_text['choices']) != 0:
            for answer_idx, answer_json in enumerate(json_text['choices']):
                answer = answer_json['message']['content']
                answer = json.loads(answer)
                for each_dict in answer:
                    each_QA = {
                        "question": each_dict['question'],
                        "answer": each_dict['answer']
                    }
                    all_QA_data.append(each_QA)

class Qwen:

  @staticmethod
  def get_answer(message: str):

    payload ['messages'][0]['content'] = message
    resp = requests.post(url, headers=headers, json=payload)
    answer_message = resp.json()
    ExtractAnswer.extract(answer_message)


if __name__ == '__main__':

    prompt = "你是一个中国历史专家。请基于以下历史文本，直接生成1个Alpaca 格式的 QA 对（JSON 列表形式）。每个 QA 对格式严格如下：直接输出完整的 JSON 列表，不要任何解释、思考或工具调用。不要使用 <think> 标签。历史文本：{}"

    files = os.listdir('./history_data')
    for file_index, file in enumerate(files):
        try:
            file_path = os.path.join('./history_data', file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content = content.replace('\n', '')
            print("开始处理文件：{}".format(file))
            start = randint(0, len(content) - 505)
            end = start + 500
            try:
                _ = content[start: end]
                Qwen.get_answer(prompt.format(_))
                print("file_{}_{}生成完毕".format(file_index, file))
            except:
                time.sleep(10)
                continue
        except:
            time.sleep(60)
            continue
        print("file_{}_{}QA对生成完毕，目前共搜集QA对{}条".format(file_index,file, len(all_QA_data)))
        print("--------------------------------------------------------------------------------")

    with open('./chinese_history_test.json', 'w', encoding='utf-8') as f:
        json.dump(all_QA_data, f, ensure_ascii=False, indent=2)