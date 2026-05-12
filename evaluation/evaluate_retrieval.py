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

from src.helper import download_embeddings, format_source_documents


load_dotenv()

INDEX_NAME = "domain-rag-assistant"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "retrieval_results.csv"


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def is_question_allowed_for_domain(question: str, selected_domain: str) -> bool:
    """
    Same basic domain guard used in the app.
    This prevents wrong-domain questions from being evaluated as valid retrieval.
    """
    question = question.lower()

    medical_keywords = [
        "disease", "symptom", "treatment", "medicine", "medical", "health",
        "doctor", "patient", "diagnosis", "infection", "pain", "skin",
        "acne", "asthma", "cancer", "diabetes", "blood", "heart", "fever",
        "virus", "bacteria", "drug", "therapy", "injury", "body", "organ"
    ]

    machine_learning_keywords = [
        "machine learning", "ml", "gradient descent", "loss function",
        "cost function", "model", "training", "dataset", "feature",
        "label", "overfitting", "underfitting", "regression", "classification",
        "neural network", "deep learning", "backpropagation", "optimizer",
        "parameter", "weight", "bias", "accuracy", "precision", "recall",
        "supervised", "unsupervised", "reinforcement learning"
    ]

    llm_keywords = [
        "llm", "large language model", "generative ai", "transformer",
        "attention", "self-attention", "prompt", "prompting", "rag",
        "retrieval augmented generation", "embedding", "vector database",
        "token", "context window", "agent", "fine-tuning", "in-context learning",
        "chatbot", "language model"
    ]

    domain_keywords = {
        "medical": medical_keywords,
        "machine_learning": machine_learning_keywords,
        "llm": llm_keywords
    }

    allowed_keywords = domain_keywords.get(selected_domain, medical_keywords)

    return any(keyword in question for keyword in allowed_keywords)


def keyword_score(text: str, expected_keywords: list[str]):
    text_lower = text.lower()

    matched_keywords = [
        keyword for keyword in expected_keywords
        if keyword.lower() in text_lower
    ]

    matched_count = len(matched_keywords)
    total_count = len(expected_keywords)

    score = matched_count / total_count if total_count > 0 else 0

    return matched_count, total_count, score, matched_keywords


def main():
    questions = load_questions()

    print("Loading embeddings...")
    embeddings = download_embeddings()

    print("Connecting to Pinecone...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
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

        allowed = is_question_allowed_for_domain(question, domain)

        if not allowed:
            retrieved_text = ""
            sources = []
            matched_count, total_count, score, matched_keywords = keyword_score(
                f"outside {domain} domain",
                expected_keywords
            )

            has_sources = False
            source_check_passed = has_sources == should_have_sources
            passed = source_check_passed

            print("Domain guard: blocked wrong-domain question")
            print(f"Passed: {passed}")

            results.append({
                "id": question_id,
                "domain": domain,
                "question": question,
                "domain_guard_allowed": allowed,
                "expected_keywords": ", ".join(expected_keywords),
                "matched_keywords": ", ".join(matched_keywords),
                "keyword_score": round(score, 2),
                "has_sources": has_sources,
                "should_have_sources": should_have_sources,
                "source_check_passed": source_check_passed,
                "passed": passed,
                "source_count": len(sources),
                "top_sources": ""
            })

            continue

        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": {"domain": domain}
            }
        )

        docs = retriever.invoke(question)

        retrieved_text = "\n\n".join(doc.page_content for doc in docs)
        sources = format_source_documents(docs)

        matched_count, total_count, score, matched_keywords = keyword_score(
            retrieved_text,
            expected_keywords
        )

        has_sources = len(sources) > 0
        source_check_passed = has_sources == should_have_sources

        # For retrieval evaluation, 0.4 is okay because retrieved chunks may contain related context
        passed = score >= 0.4 and source_check_passed

        top_sources = [
            f"{source['source']} page {source['page']} chunk {source['chunk_id']}"
            for source in sources
        ]

        print(f"Retrieved docs: {len(docs)}")
        print(f"Matched keywords: {matched_keywords}")
        print(f"Keyword score: {score:.2f}")
        print(f"Has sources: {has_sources}")
        print(f"Expected sources: {should_have_sources}")
        print(f"Passed: {passed}")

        results.append({
            "id": question_id,
            "domain": domain,
            "question": question,
            "domain_guard_allowed": allowed,
            "expected_keywords": ", ".join(expected_keywords),
            "matched_keywords": ", ".join(matched_keywords),
            "keyword_score": round(score, 2),
            "has_sources": has_sources,
            "should_have_sources": should_have_sources,
            "source_check_passed": source_check_passed,
            "passed": passed,
            "source_count": len(sources),
            "top_sources": " | ".join(top_sources)
        })

    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "domain",
                "question",
                "domain_guard_allowed",
                "expected_keywords",
                "matched_keywords",
                "keyword_score",
                "has_sources",
                "should_have_sources",
                "source_check_passed",
                "passed",
                "source_count",
                "top_sources"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    passed_count = sum(1 for row in results if row["passed"])
    avg_score = sum(row["keyword_score"] for row in results) / total if total > 0 else 0

    print("\n" + "=" * 80)
    print("Retrieval evaluation complete.")
    print(f"Total questions: {total}")
    print(f"Passed: {passed_count}/{total}")
    print(f"Average keyword score: {avg_score:.2f}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()