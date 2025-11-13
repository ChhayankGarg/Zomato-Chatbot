import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

@st.cache_resource
def get_vectorstore():
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embed, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 256}
    )

@st.cache_resource
def create_qa_chain():
    vectorstore = get_vectorstore()
    llm = load_llm()

    prompt = PromptTemplate(
        template="""
Use the context to answer the question.

Context:
{context}

Question:
{input}

Answer:
""",
        input_variables=["context", "input"]
    )

    combine_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    return create_retrieval_chain(retriever, combine_chain)

st.title("🍽️ Zomato Chatbot")
st.markdown("Ask anything about restaurants, dishes, or pricing!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

qa_chain = create_qa_chain()

for item in st.session_state.chat_history:
    st.write(f"**You:** {item['question']}")
    st.write(f"**Bot:** {item['answer']}")

user_input = st.text_input("Ask a new question:")

if user_input:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"input": user_input})
        answer = result["answer"]

    st.session_state.chat_history.append({
        "question": user_input,
        "answer": answer
    })

    st.rerun()
