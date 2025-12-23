import streamlit as st
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm

# ==========================================
# 配置区域（请根据实际情况修改）
# ==========================================
ST_TITLE = "中国历史 RAG 问答系统"

# txt 文件夹路径（修改为你的实际路径）
TXT_FOLDER = "./data_extract/history_data"
# 本地 Qwen2.5-7B-Instruct 路径（你之前下载到 pub 的）
# LOCAL_MODEL_PATH = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"
LOCAL_MODEL_PATH = "./merged_qwen_history"
# 嵌入模型"BAAI/bge-m3"（支持更长文本）
EMBEDDING_MODEL = "BAAI/bge-m3"
# 向量库持久化目录
VECTOR_DB_PATH = "./chroma_db_history"

# ==========================================
# 初始化 RAG 系统（只运行一次，缓存）
# ==========================================
@st.cache_resource
def initialize_rag_system():
    # 1. 检查 txt 文件夹
    if not os.path.exists(TXT_FOLDER):
        return None, f"txt 文件夹不存在: {TXT_FOLDER}"

    txt_files = [f for f in os.listdir(TXT_FOLDER) if f.endswith(".txt")]
    if not txt_files:
        return None, f"文件夹 {TXT_FOLDER} 中没有 txt 文件"

    st.info(f"发现 {len(txt_files)} 个 txt 文件，正在加载...")

    # 2. 加载所有 txt 文件
    docs = []
    for file_name in tqdm(txt_files, desc="加载 txt 文件"):
        file_path = os.path.join(TXT_FOLDER, file_name)
        loader = TextLoader(file_path, encoding="utf-8")
        docs.extend(loader.load())

    st.success(f"成功加载 {len(docs)} 个文档（部分大文件可能被自动分段）")

    # 3. 文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 适合历史文本，可根据需要调整
        chunk_overlap=100,
        length_function=len,
    )
    splits = text_splitter.split_documents(docs)
    st.info(f"切分为 {len(splits)} 个 chunk")

    # 4. 本地嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 5. 构建或加载向量库
    if os.path.exists(VECTOR_DB_PATH):
        st.info("检测到已有向量库，直接加载...")
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    else:
        st.info("正在构建向量库（首次运行较慢，后续秒开）...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        st.success("向量库构建完成并已保存！")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 6. 加载本地 Qwen2.5-7B-Instruct
    if not os.path.exists(LOCAL_MODEL_PATH):
        return None, f"模型路径不存在: {LOCAL_MODEL_PATH}"

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # load_in_4bit=True,   # 如显存不够可开启（需 pip install bitsandbytes）
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

    # 正确方式：先包装成 HuggingFacePipeline，再用 ChatHuggingFace
    llm_pipeline = HuggingFacePipeline(pipeline=pipe)

    llm = ChatHuggingFace(
        llm=llm_pipeline,       # ← 必须用 llm= 参数
        tokenizer=tokenizer,
        streaming=True
    )
    
    # 7. Prompt 模板
    template = """
    你是一个中国历史专家。请根据以下检索到的上下文，准确、详尽地回答用户问题。
    如果上下文没有足够信息，请说“根据当前知识库，我无法提供确切答案”。
    
    上下文：
    {context}
    
    问题：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 8. RAG Chain
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain, f"系统就绪！知识库包含 {len(txt_files)} 个历史 txt 文件"


# ==========================================
# Streamlit 界面
# ==========================================
st.set_page_config(page_title=ST_TITLE, page_icon="📜")
st.title(ST_TITLE)

with st.sidebar:
    st.header("系统状态")
    with st.spinner("正在初始化 RAG 系统..."):
        rag_chain, msg = initialize_rag_system()

    if rag_chain:
        st.success("✅ RAG 系统已就绪")
        st.info(msg)
        st.info(f"🧠 模型: 本地 Qwen2.5-7B-Instruct\n\n📚 嵌入模型: {EMBEDDING_MODEL}")
    else:
        st.error(f"❌ 初始化失败: {msg}")
        st.stop()

    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入你想知道的中国历史相关问题"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in rag_chain.stream(prompt):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            st.error(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})