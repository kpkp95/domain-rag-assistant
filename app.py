from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_embeddings, format_source_documents
from src.prompt import system_prompt


app = Flask(__name__)

load_dotenv()

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


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

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "filter": {"domain": selected_domain}
        }
    )

    question_answer_chain = create_stuff_documents_chain(
        chat_model,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    response = rag_chain.invoke({
        "input": user_message
    })

    answer = response.get("answer", "I could not generate an answer.")
    source_docs = response.get("context", [])
    sources = format_source_documents(source_docs)

    return jsonify({
        "answer": answer,
        "sources": sources
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )