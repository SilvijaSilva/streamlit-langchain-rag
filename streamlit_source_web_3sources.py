from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

import os
from langchain import hub

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
import bs4  # BeautifulSoup for parsing HTML

# Load environment variables from .env file
load_dotenv()

token = os.getenv("MY_TOKEN")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

loader = WebBaseLoader(
    web_paths=[
        "https://lt.wikipedia.org/wiki/Širvintos",
        #"https://www.vle.lt/straipsnis/sirvintos/"
        "https://www.yesforskills.lt/ka-verta-pamatyti-sirvintose-lankytinos-vietos-su-vaikais-ir-idomus-faktai-apie-si-miesta"
    ]
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
splits = text_splitter.split_documents(docs)

embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token, # type: ignore
)

vectorstore = InMemoryVectorStore(embeddings)
_ = vectorstore.add_documents(documents=splits)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt = hub.pull("rlm/rag-prompt")

# Format documents for output
def format_docs(docs):
    #print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

# Streamlit app setup
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

    st.subheader("📚 Sources")
    for i, doc in enumerate(fetched_docs, 1):
        with st.expander(f"Source {i}"):
            st.write(f"**Content:** {doc.page_content}")

with st.form("my_form"):
    text = st.text_area(
        "Įvesk klausimą:",
        "Kokiai apskričiai priklauso Širvintos?",
    )
    submitted = st.form_submit_button("Pateikti")
    if submitted:
        generate_response(text)
