# 中国历史RAG问答系统

作为南航数学学院的一位研究生，我对中国的历史颇感兴趣，因此，我将构建一个基于Qwen大模型的RAG历史问答系统，并使用Lora微调提升领域准确性。
- **github仓库**：https://github.com/renhaodong1011/chinese-history-rag-qwen 。
- **目前已实现内容**：1.编写了脚本model_download.py 用于下载Qwen-2.5-7B-Instruct模型到AutoDL本地；2.在文件夹data_extract中，编写脚本spider.py用于在中华上下五千年网站中爬取所有文本数据，共282个txt文 件；3.在文件夹data_extract中，编写脚本generate_QA_data.py,使用QWen4-8B模型，智能从爬取的所有数据中生成QA对，用于后续的Lora微调，共计提取QA对9978条，保存在相应文件夹下的chinese_history_qa.json文件中；4.在RAG.py文件中通过streamlit实现了一个完全本地化的中国历史领域 RAG（Retrieval-Augmented Generation）Web问答系统,支持流式输出。 且该第一版项目已经在AutoDL,A100-PCIE-40GB(40GB)单卡下部署成功。并将一些项目过程中的输出以截图的方式展示在了img文件夹中，包括数据浏览，QA对生成，RAG问答截图等。 后续的迭代，将以Lora微调模型为主线展开

## 运行环境

- **平台** ： AutoDL(https://www.autodl.com/)。
- **镜像配置**： PyTorch==2.8.0 Python==3.12(ubuntu22.04) CUDA==12.8。
- **GPU** ： A100-PCIE-40GB(40GB)。

## 依赖安装

**执行命令**: pip install -r requirement.txt


## 数据来源

- **RAG知识库数据** : 用于RAG的检索模块，提供外部知识增强生成。 使用python爬虫从中华上下五千年网站（https://zhonghua.5000yan.com/）中，爬取所有的相关内容。
- **Lora微调数据获取** ： 使用现有的QWen3-8B模型，对每个历史文件提取并生成若干个QA对，用于LORA微调。

## 基础模型

使用 Qwen-2.5-7B-instruct 为基础模型
在AutoDL中，在 /root/autodl-tmp 路径下新建 model_download.py 用于下载完整的模型。
``` Python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
```
[模型目录](./img/1.png)

## 文件作用介绍
- **1.model_download.py**:用于下载Qwen-2.5-7B-Instruct模型到AutoDL本地。
- **2.data_extract/spider.py**: 用于在中华上下五千年网站中爬取所有文本数据，共282个txt文件。
- **3.data_extract/generate_QA_data.py**:使用QWen4-8B模型，智能从爬取的所有数据中生成QA对，用于后续的Lora微调，共计提取QA对9978条，保存在相应文件夹下的chinese_history_qa.json文件中。
- **4.RAG.py**: 通过streamlit实现了一个完全本地化的中国历史领域 RAG（Retrieval-Augmented Generation）Web问答系统,支持流式输出。

## 启动方式

streamlit run RAG.py