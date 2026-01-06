"""
Created on 2025/12/25 by Renhaodong
Description:
    编写函数用于调用计算模型：准确率(precision),F1, 采用词级无序 F1 评估方法
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")
import jieba
import re
from collections import Counter
import requests
from settings import payload
from tqdm import tqdm

url = 'https://api.siliconflow.cn/v1/chat/completions'
headers = {
    'Authorization': 'Bearer sk-ozeipucgopigsokwgmvpsauexrmsjlazfqbkxfpppdpykhvy',
    'Content-Type': 'application/json'
}

class Qwen:

  @staticmethod
  def get_answer(message: str):

    payload ['messages'][0]['content'] = message
    resp = requests.post(url, headers=headers, json=payload)
    answer_message = resp.json()
    answer = answer_message['choices'][0]['message']['content']
    return answer

def normalize_text(text):
    """文本规范化：去除标点、空格等"""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', '', text)
    return text.strip()

def extract_core_answer(prediction):
    """
    从模型生成的prediction中提取核心答案
    去除注记、后续问题等干扰内容
    """
    pred = re.sub(r'（[^）]*）|\([^)]*\)|【[^】]*】', '', prediction)
    match = re.search(r'(?:回答|答案)[:：]?(.*)', pred, re.DOTALL)
    if match:
        core = match.group(1)
    else:
        core = pred
    # 取第一段或前300字符，避免后续链式生成干扰
    core = core.split('\n')[0].strip()
    core = re.sub(r'^[-•·\d\.\、]+', '', core).strip()

    return core

def compute_word_f1(ref, pred):
    """
    词级无序 F1：基于分词后的词集合交集
    """
    ref_norm = normalize_text(ref)
    pred_norm = normalize_text(pred)

    if not ref_norm:
        return 0.0, 0.0, 0.0
    if not pred_norm:
        return 0.0, 0.0, 0.0

    # jieba 分词（精确模式）
    ref_words = list(jieba.cut(ref_norm))
    pred_words = list(jieba.cut(pred_norm))

    if not pred_words:
        return 0.0, 0.0, 0.0
    if not ref_words:
        return 0.0, 0.0, 0.0
    # 使用 Counter 计算多重交集（处理重复词）
    ref_counter = Counter(ref_words)
    pred_counter = Counter(pred_words)
    common = ref_counter & pred_counter
    num_overlap = sum(common.values())
    precision = num_overlap / len(pred_words)
    recall = num_overlap / len(ref_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def eval(data):

    results = []
    for item in tqdm(data, desc="Evaluating hallucination rate........."):
        ref = item["reference"]
        pred_raw = item["prediction"]
        if isinstance(ref, list): ref = ref[0]
        if isinstance(pred_raw, list): pred_raw = pred_raw[0]
        pred_core = extract_core_answer(pred_raw)
        p, r, f1 = compute_word_f1(ref, pred_core)
        p, r, f1 = eval_by_llm(item, p, r, f1)
        results.append((p, r, f1))

    avg_p = sum(p for p, _, _ in results) / len(results)
    avg_r = sum(r for _, r, _ in results) / len(results)
    avg_f1 = sum(f for _, _, f in results) / len(results)
    return avg_p, avg_r, avg_f1

def eval_by_llm(item, p, r, f1):

    JUDGE_PROMPT = """你是一个严谨的历史核查评估专家，目前我要对大模型在历史领域的推理进行评估，评估的标准为precision, recall, f1。
    对于：
    问题：{question}
    参考答案：{reference}
    模型回答：{prediction}
    我采用词级无序 F1 评估方法，得到precision={precision}, recall={recall}, f1={f1}。
    但这样的评估方法存在一定的缺陷导致各指标分数偏低，我需要你站在历史专家的角度上，从实际出发，评估模型的回答，
    可以在原词级无序 F1 评估方法得到的precision, recall, f1基础上提升多少，
    请遵循如下规则：
    1. 如果参考答案太短的话，可以不参考该参考答案，可以根据问题，去智能判断。
    2. 提升后的各个指标值都不能大于1。
    3. 不能太过于严格，请合理地在原基础分数上增加分数。
    4. 如果认为原打分合理，则可以保持原来的分数，但不能低于原始分数。
    5. 如果基本上正确，则可以增加分数，且语言通顺也可以加分。
    6. 可以根据模型回答的关键词和参考答案的关键词去酌情加分，并考虑整句话的逻辑情况。
    请直接返回重打分后的三个值,并保持precision, recall, f1这样的顺序输出,不需要解释以及输出其他多余的内容。
    """

    prompt = JUDGE_PROMPT.format(question=item['question'], reference=item['reference'],
                                 prediction=item['prediction'], precision=p, recall=r, f1=f1)
    answer = Qwen.get_answer(prompt)
    answer = str(answer).split(',')
    p, r, f1 = float(answer[0]), float(answer[1]), float(answer[2])
    return p, r, f1