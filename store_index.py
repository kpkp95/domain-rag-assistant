from dotenv import load_dotenv
import os
import argparse
from collections import Counter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from src.helper import (
    add_chunk_ids,
    clean_text,
    download_embeddings,
    load_pdf_files,
    filter_to_minimal_docs,
    text_split
)


load_dotenv()


def get_args():
    """
    Read domain name from command line.

    Examples:
        python store_index.py --domain medical
        python store_index.py --domain machine_learning
        python store_index.py --domain llm
    """
    parser = argparse.ArgumentParser(
        description="Upload domain-specific PDF documents to Pinecone."
    )

    parser.add_argument(
        "--domain",
        type=str,
        default="medical",
        choices=["medical", "machine_learning", "llm"],
        help="Domain folder to index: medical, machine_learning, or llm"
    )

    return parser.parse_args()


def validate_chunks(text_chunks):
    """
    Check that every chunk has valid string content before uploading to Pinecone.
    """
    bad_chunks = []

    for i, chunk in enumerate(text_chunks):
        if not isinstance(chunk.page_content, str) or not chunk.page_content.strip():
            bad_chunks.append((i, type(chunk.page_content), chunk.page_content))

    print("Bad chunks found:", len(bad_chunks))

    if bad_chunks:
        print("First bad chunk:", bad_chunks[0])
        raise ValueError("Invalid chunks found. Fix text cleaning before uploading.")


def upload_documents_in_batches(vectorstore, text_chunks, batch_size=50):
    """
    Upload documents to Pinecone in smaller batches.

    Multiple PDFs/books are already supported because DirectoryLoader loads
    every PDF inside the selected domain folder.
    """
    total_chunks = len(text_chunks)

    for start in range(0, total_chunks, batch_size):
        end = min(start + batch_size, total_chunks)
        batch = text_chunks[start:end]

        clean_batch = []

        for doc in batch:
            content = clean_text(doc.page_content)

            if not content:
                continue

            clean_batch.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "domain": doc.metadata.get("domain"),
                        "chunk_id": doc.metadata.get("chunk_id")
                    }
                )
            )

        if not clean_batch:
            print(f"Skipping empty batch {start} to {end}")
            continue

        print(f"Uploading chunks {start} to {end}...")

        try:
            vectorstore.add_documents(clean_batch)

        except Exception as e:
            print(f"\nFailed batch: {start} to {end}")

            for i, doc in enumerate(clean_batch):
                print("Local chunk index:", start + i)
                print("Content type:", type(doc.page_content))
                print("Content length:", len(doc.page_content))
                print("Content preview:", repr(doc.page_content[:300]))
                print("Metadata:", doc.metadata)
                print("-" * 80)

            raise e


def main():
    args = get_args()

    domain = args.domain
    data_path = f"data/{domain}"
    index_name = "domain-rag-assistant"

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is missing from .env")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing from .env")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    print(f"Selected domain: {domain}")
    print(f"Loading PDFs from: {data_path}")

    extracted_data = load_pdf_files(data_path)
    
    if len(extracted_data) == 0:
        raise ValueError(f"No PDF files found inside: {data_path}")

    print(f"Pages loaded: {len(extracted_data)}")
    page_counts = Counter(doc.metadata.get("source", "Unknown source") for doc in extracted_data)

    print("\nPDF files loaded:")
    for source, count in page_counts.items():
        print(f"- {source}: {count} pages")
    filtered_data = filter_to_minimal_docs(extracted_data, domain)

    if len(filtered_data) == 0:
        raise ValueError(f"No valid text content found inside PDFs in: {data_path}")

    text_chunks = text_split(filtered_data)
    text_chunks = add_chunk_ids(text_chunks)
    chunk_counts = Counter(chunk.metadata.get("source", "Unknown source") for chunk in text_chunks)

    print("\nChunks created per PDF:")
    for source, count in chunk_counts.items():
        print(f"- {source}: {count} chunks")
        
    if len(text_chunks) == 0:
        raise ValueError("No chunks created. Check PDF content and text splitting.")

    print(f"Chunks created: {len(text_chunks)}")
    print("Sample metadata:", text_chunks[0].metadata)
    print("Sample content:", text_chunks[0].page_content[:300])

    validate_chunks(text_chunks)

    embedding = download_embeddings()

    pc = Pinecone(api_key=pinecone_api_key)

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created Pinecone index: {index_name}")
    else:
        print(f"Pinecone index already exists: {index_name}")

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )

    upload_documents_in_batches(
        vectorstore=vectorstore,
        text_chunks=text_chunks,
        batch_size=50
    )

    print(f"Successfully uploaded {domain} documents to Pinecone.")


if __name__ == "__main__":
    main()