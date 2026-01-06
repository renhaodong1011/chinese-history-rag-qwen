"""
Created on 2025/12/20 by RenHaodong
Description:
    用于从爬取的文件中，提取QA对，用于LORA微调,Qwen3-Next-80B-A3B-Thinking
"""
import requests
from typing import *
import json
import os
import time
import random

url = 'https://api.siliconflow.cn/v1/chat/completions'
headers = {
    'Authorization': 'Bearer sk-ozeipucgopigsokwgmvpsauexrmsjlazfqbkxfpppdpykhvy',
    'Content-Type': 'application/json'
}

payload = {
  "model": "Qwen/Qwen2.5-32B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": ""
    }
  ],
  "stream": False,
  "max_tokens": 10000,
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
test_QA_data = []

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
                        "instruction": each_dict['question'],
                        "input": "",
                        "output": each_dict['answer']
                    }
                    if (len(all_QA_data)+ len(test_QA_data) + 1) % 10 == 0:
                        test_QA_data.append(each_QA)
                    else:
                        all_QA_data.append(each_QA)

class Qwen:

  @staticmethod
  def get_answer(message: str):

    payload ['messages'][0]['content'] = message
    resp = requests.post(url, headers=headers, json=payload)
    answer_message = resp.json()
    ExtractAnswer.extract(answer_message)

if __name__ == '__main__':

    prompt = """你是一个严谨的中国历史数据合成专家。你的任务是：基于我提供的【历史文本】，生成{}个高质量的Alpaca格式指令微调数据（question-answer对）。

    严格要求：
    1. 所有问题和答案必须100%来自提供的【历史文本】，不允许加入任何文本之外的知识、不允许杜撰、不允许推测。
    2. 如果文本信息不足以支撑一个完整答案，就不要生成该问题。
    3. 问题要多样：包括事实性问题、因果关系、时间顺序、人物关系、细节解读等。
    4. 答案要准确、完整，可以是长文本，直接引用或合理组织文本内容。
    5. 最后额外生成3个通史类QA对：基于中国从夏商周到中华民国的标准通史知识（不需要依赖当前文本），问题覆盖朝代更迭、重大事件脉络、兴衰原因等，确保事实标准无争议。

    输出格式：直接输出一个完整的JSON列表，不要任何解释、思考、工具调用、<think>标签或多余文字。
    示例输出格式：
    [
      {{"question": "xxx", "answer": "xxx"}},
      ...
    ]
    历史文本：{}
    """
    files = os.listdir('./history_data')
    for file_index, file in enumerate(files):
        try:
            file_path = os.path.join('./history_data', file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content = content.replace('\n', '')
            print("开始处理文件：{}".format(file))
            for i in range(0, len(content)-402 ,200):
                try:
                    _ = content[i: i+400]
                    num_qa = random.randint(6, 10)
                    Qwen.get_answer(prompt.format(num_qa,_))
                    print("file_{}_{}_{}生成完毕".format(file_index, file, i//400+1))
                except:
                    time.sleep(10)
                    continue
        except:
            time.sleep(60)
            continue
        print("file_{}_{}QA对生成完毕，目前共搜集训练集QA对{}条，测试集QA对{}条".format(
            file_index,file, len(all_QA_data), len(test_QA_data)))
        print("--------------------------------------------------------------------------------")

    with open('./train_qa.json', 'w', encoding='utf-8') as f:
        json.dump(all_QA_data, f, ensure_ascii=False, indent=2)

    with open('./test_qa.json', 'w', encoding='utf-8') as f:
        json.dump(test_QA_data, f, ensure_ascii=False, indent=2)