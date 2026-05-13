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

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from src.helper import download_embeddings, format_source_documents,is_greeting_or_small_talk
from src.prompt import get_system_prompt


app = Flask(__name__)

load_dotenv()

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CHAT_SESSION_TTL = int(os.getenv("CHAT_SESSION_TTL", 3600))

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

def get_greeting_response(selected_domain: str) -> str:
    """
    Return a friendly domain-aware greeting response.
    """
    domain_names = {
        "medical": "Medical",
        "machine_learning": "Machine Learning",
        "llm": "LLM"
    }

    examples = {
        "medical": [
            "What is acne?",
            "What are the symptoms of asthma?",
            "What is diabetes?"
        ],
        "machine_learning": [
            "What is gradient descent?",
            "What is overfitting?",
            "What is linear regression?"
        ],
        "llm": [
            "What is RAG?",
            "What is prompting?",
            "What is a transformer?"
        ]
    }

    domain_label = domain_names.get(selected_domain, "selected")
    domain_examples = examples.get(selected_domain, [])

    example_text = "\n".join([f"- {example}" for example in domain_examples])

    return (
        f"Hello! I am your Domain RAG Assistant.\n\n"
        f"You are currently using the {domain_label} knowledge base.\n\n"
        f"You can ask questions like:\n"
        f"{example_text}\n\n"
        f"I will answer using the selected documents and show sources when relevant."
    )
def get_session_id():
    """
    Create or return a unique session id for each browser session.
    """
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    return session["session_id"]


def get_session_history(session_id: str):
    """
    LangChain Redis-backed conversation memory.
    Each session_id gets its own chat history in Redis.
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        ttl=CHAT_SESSION_TTL
    )


def build_rag_chain(selected_domain: str):
    """
    Build a conversational RAG chain with domain-specific retrieval,
    domain-specific prompt, and LangChain memory.
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

    domain_system_prompt = get_system_prompt(selected_domain)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", domain_system_prompt),
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

    if selected_domain not in ["medical", "machine_learning", "llm"]:
        selected_domain = "medical"

    if not user_message:
        return jsonify({
            "answer": "Please enter a question.",
            "sources": [],
            "domain": selected_domain
        })
    if is_greeting_or_small_talk(user_message):
        return jsonify({
        "answer": get_greeting_response(selected_domain),
        "sources": [],
        "domain": selected_domain
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

    answer_lower = answer.lower()

    no_source_phrases = [
        "i do not know based on the provided medical documents",
        "i do not know based on the provided machine learning documents",
        "i do not know based on the provided llm documents",
        "this question appears to be outside the selected medical domain",
        "this question appears to be outside the selected machine learning domain",
        "this question appears to be outside the selected llm domain",
    ]

    if any(phrase in answer_lower for phrase in no_source_phrases):
        sources = []
    else:
        sources = format_source_documents(source_docs)

    return jsonify({
        "answer": answer,
        "sources": sources,
        "domain": selected_domain
    })

@app.route("/clear", methods=["POST"])
def clear_chat():
    session_id = session.get("session_id")

    if session_id:
        history = get_session_history(session_id)
        history.clear()

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