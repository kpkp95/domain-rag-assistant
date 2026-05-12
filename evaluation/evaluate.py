import json
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_embeddings, format_source_documents
from src.prompt import get_system_prompt


load_dotenv()

INDEX_NAME = "domain-rag-assistant"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.csv"


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env")


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def keyword_score(answer: str, expected_keywords: list[str]) -> tuple[int, int, float, list[str]]:
    answer_lower = answer.lower()

    matched_keywords = [
        keyword for keyword in expected_keywords
        if keyword.lower() in answer_lower
    ]

    matched_count = len(matched_keywords)
    total_count = len(expected_keywords)

    score = matched_count / total_count if total_count > 0 else 0

    return matched_count, total_count, score, matched_keywords


def build_rag_chain(domain: str, docsearch, chat_model):
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "filter": {"domain": domain}
        }
    )

    system_prompt = get_system_prompt(domain)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        chat_model,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    return rag_chain


def main():
    questions = load_questions()

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

    results = []

    for item in questions:
        question_id = item["id"]
        domain = item["domain"]
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
        should_have_sources = item.get("should_have_sources", True)

        print("=" * 80)
        print(f"Question ID: {question_id}")
        print(f"Domain: {domain}")
        print(f"Question: {question}")

        rag_chain = build_rag_chain(
            domain=domain,
            docsearch=docsearch,
            chat_model=chat_model
        )

        response = rag_chain.invoke({
            "input": question
        })

        answer = response.get("answer", "")
        source_docs = response.get("context", [])
        sources = format_source_documents(source_docs)

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

        matched_count, total_count, score, matched_keywords = keyword_score(
            answer,
            expected_keywords
        )

        has_sources = len(sources) > 0
        source_check_passed = has_sources == should_have_sources

        passed = score >= 0.5 and source_check_passed

        print(f"Answer preview: {answer[:250]}...")
        print(f"Matched keywords: {matched_keywords}")
        print(f"Keyword score: {score:.2f}")
        print(f"Has sources: {has_sources}")
        print(f"Expected sources: {should_have_sources}")
        print(f"Passed: {passed}")

        results.append({
            "id": question_id,
            "domain": domain,
            "question": question,
            "answer": answer,
            "expected_keywords": ", ".join(expected_keywords),
            "matched_keywords": ", ".join(matched_keywords),
            "keyword_score": round(score, 2),
            "has_sources": has_sources,
            "should_have_sources": should_have_sources,
            "source_check_passed": source_check_passed,
            "passed": passed,
            "source_count": len(sources)
        })

    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "domain",
                "question",
                "answer",
                "expected_keywords",
                "matched_keywords",
                "keyword_score",
                "has_sources",
                "should_have_sources",
                "source_check_passed",
                "passed",
                "source_count"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    passed_count = sum(1 for row in results if row["passed"])
    avg_score = sum(row["keyword_score"] for row in results) / total if total > 0 else 0

    print("\n" + "=" * 80)
    print("Evaluation complete.")
    print(f"Total questions: {total}")
    print(f"Passed: {passed_count}/{total}")
    print(f"Average keyword score: {avg_score:.2f}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()