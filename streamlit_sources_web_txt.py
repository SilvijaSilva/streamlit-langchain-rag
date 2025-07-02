from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader

import os
from langchain import hub

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
import bs4

load_dotenv()

token = os.getenv("MY_TOKEN")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

web_loader = WebBaseLoader(
    web_paths=[
        "https://lt.wikipedia.org/wiki/Širvintos",
        "https://www.yesforskills.lt/ka-verta-pamatyti-sirvintose-lankytinos-vietos-su-vaikais-ir-idomus-faktai-apie-si-miesta"
    ]
)
web_docs = web_loader.load()

file_loader = TextLoader("sirvintos_info.txt", encoding="utf-8")
local_docs = file_loader.load()

all_docs = web_docs + local_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=10)
splits = text_splitter.split_documents(all_docs)

embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
)

vectorstore = InMemoryVectorStore(embeddings)
_ = vectorstore.add_documents(documents=splits)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    #print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

st.title("Klausk ir sužinok viską apie Širvintas")

def generate_response(input_text):
    llm = ChatOpenAI(base_url=endpoint, temperature=0.7, api_key=token, model=model)

    fetched_docs = vectorstore.search(input_text, search_type="similarity", k=3)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    
    result = rag_chain.invoke(input_text)
    st.info(result)

    st.subheader("📚 Informacijos šaltiniai")
    for i, doc in enumerate(fetched_docs, 1):
        with st.expander(f"Šaltinis {i}"):
            st.write(f"**Turinys:** {doc.page_content}")

with st.form("my_form"):
    text = st.text_area(
        "Įvesk klausimą:",
        "Kur yra Širvintos?",
    )
    submitted = st.form_submit_button("Pateikti")
    if submitted:
        generate_response(text)
