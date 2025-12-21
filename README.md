# 中国历史RAG问答系统

    作为南航数学学院的一位研究生，我对中国的历史颇感兴趣，因此，我将构建一个基于Qwen大模型的RAG历史问答系统，并使用Lora微调提升领域准确性。
    github仓库：https://github.com/renhaodong1011/chinese-history-rag-qwen

## 运行环境

    **平台** ： AutoDL(https://www.autodl.com/)；
    **镜像配置**： PyTorch==2.8.0 Python==3.12(ubuntu22.04) CUDA==12.8；
    **GPU** ： A100-PCIE-40GB(40GB)；

## 依赖安装

    可创建requirements.txt文件，并执行pip install -i https:pypi.tuna.tsinghua.edu.cn/simple requirement.txt
    torch==2.8.0+cu128
    modelscope==1.18.0
    transformers==4.44.2
    streamlit==1.24.0
    sentencepiece==0.2.0
    accelerate==0.34.2
    datasets==2.20.0
    peft==0.11.1

## 数据来源

    RAG知识库数据: 用于RAG的检索模块，提供外部知识增强生成。 使用python爬虫从中华上下五千年网站（https://zhonghua.5000yan.com/）中，爬取所有的相关内容。
    Lora微调数据获取： 使用现有的QWen2.5-7B-Instruct模型，对每个历史文件提取并生成5-10个QA对，用于LORA微调。

## 基础模型

    使用 Qwen-2.5-7B-instruct 为基础模型
    在AutoDL中，在 /root/autodl-tmp 路径下新建 model_download.py 用于下载完整的模型。
    ``` Python
    import torch
    from modelscope import snapshot_download, AutoModel, AutoTokenizer
    import os
    model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp', revision='master')

    ![模型目录](./img/1.png)

## 启动方式

    streamlit run RAG.py