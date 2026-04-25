from flask import Flask, render_template, jsonify, request, session
from dotenv import load_dotenv
import os
import uuid

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import (
    create_retrieval_chain,
    create_history_aware_retriever
)
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.helper import download_embeddings, format_source_documents
from src.prompt import system_prompt


app = Flask(__name__)

load_dotenv()

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env")


INDEX_NAME = "domain-rag-assistant"

embeddings = download_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


# LangChain memory store
chat_store = {}


def get_session_id():
    """
    Create or return a unique session id for each browser session.
    """
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    return session["session_id"]


def get_session_history(session_id: str):
    """
    LangChain conversation memory.
    Each session_id gets its own chat history.
    """
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()

    return chat_store[session_id]


def build_rag_chain(selected_domain: str):
    """
    Build a conversational RAG chain with LangChain memory.
    """

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "filter": {"domain": selected_domain}
        }
    )

    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "rewrite the latest user question as a standalone question. "
        "Do not answer the question. "
        "Only rewrite it if needed. "
        "If the question is already clear, return it unchanged."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        chat_model,
        retriever,
        contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        chat_model,
        qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form.get("msg", "").strip()
    selected_domain = request.form.get("domain", "medical")

    if not user_message:
        return jsonify({
            "answer": "Please enter a question.",
            "sources": []
        })

    session_id = get_session_id()

    conversational_rag_chain = build_rag_chain(selected_domain)

    response = conversational_rag_chain.invoke(
        {
            "input": user_message
        },
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    answer = response.get("answer", "I could not generate an answer.")
    source_docs = response.get("context", [])
    sources = format_source_documents(source_docs)

    return jsonify({
        "answer": answer,
        "sources": sources
    })


@app.route("/clear", methods=["POST"])
def clear_chat():
    session_id = session.get("session_id")

    if session_id and session_id in chat_store:
        chat_store.pop(session_id)

    session.pop("session_id", None)

    return jsonify({
        "status": "cleared"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )