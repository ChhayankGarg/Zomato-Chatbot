import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load vectorstore
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

# Load LLM using new HuggingFaceEndpoint (FIX)
@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        model_kwargs={"temperature": 0.4, "max_new_tokens": 256}
    )

# Build Retrieval Chain (new LangChain API)
@st.cache_resource
def create_qa_chain():
    vectorstore = get_vectorstore()
    llm = load_llm()

    # Prompt for combining documents
    prompt = PromptTemplate(
        template="""
You are a helpful assistant that answers questions based on the given context.

Context:
{context}

Question:
{input}

Answer concisely and clearly:
""",
        input_variables=["context", "input"],
    )

    # Combine-documents chain
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # Final retrieval chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    return create_retrieval_chain(retriever, combine_docs_chain)

# Streamlit App
st.title("🍽️ Zomato Chatbot")
st.markdown("Ask anything about restaurants, dishes, or pricing!")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

qa_chain = create_qa_chain()

# Display past conversation
for entry in st.session_state.chat_history:
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**Bot:** {entry['answer']}")

# Input field
user_input = st.text_input("Ask a new question:")

# When user submits
if user_input:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"input": user_input})
        answer = result["answer"]

    # Save chat
    st.session_state.chat_history.append({
        "question": user_input,
        "answer": answer
    })

    st.rerun()
