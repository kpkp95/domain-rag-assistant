from dotenv import load_dotenv
import os

from src.helper import (
    add_chunk_ids,
    download_embeddings,
    load_pdf_files,
    filter_to_minimal_docs,
    text_split
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env")


DOMAIN = "medical"
DATA_PATH = f"data/{DOMAIN}"
INDEX_NAME = "domain-rag-assistant"


print(f"Loading PDFs from: {DATA_PATH}")

extracted_data = load_pdf_files(DATA_PATH)
print(f"Pages loaded: {len(extracted_data)}")

filter_data = filter_to_minimal_docs(extracted_data, DOMAIN)

texts_chunk = text_split(filter_data)
texts_chunk_id = add_chunk_ids(texts_chunk)

print(f"Chunks created: {len(texts_chunk_id)}")
print("Sample metadata:", texts_chunk_id[0].metadata)

embedding = download_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created Pinecone index: {INDEX_NAME}")
else:
    print(f"Pinecone index already exists: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk_id,
    embedding=embedding,
    index_name=INDEX_NAME
)

print("Documents uploaded to Pinecone successfully.")