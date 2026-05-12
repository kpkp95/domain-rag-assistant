from typing import List
from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


def clean_text(text) -> str:
    """
    Convert text to a clean string and remove problematic Unicode/control characters
    that can break embedding tokenizers.
    """
    if text is None:
        return ""

    text = str(text)

    # Remove invalid surrogate unicode characters such as \ud835
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    # Remove control characters except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdf_files(data_path: str) -> List[Document]:
    """
    Load all PDF files from a given folder.
    This supports multiple books/PDFs inside the same domain folder.
    """
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document], domain: str) -> List[Document]:
    """
    Keep page content and useful metadata for citations and domain filtering.
    Removes empty or invalid pages.
    """
    minimal_docs: List[Document] = []

    for doc in docs:
        content = clean_text(doc.page_content)

        if not content:
            continue

        minimal_docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "domain": domain
                }
            )
        )

    return minimal_docs


def text_split(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for RAG retrieval.
    Removes empty or invalid chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = text_splitter.split_documents(docs)

    clean_chunks: List[Document] = []

    for chunk in text_chunks:
        content = clean_text(chunk.page_content)

        if not content:
            continue

        clean_chunks.append(
            Document(
                page_content=content,
                metadata=chunk.metadata
            )
        )

    return clean_chunks


def add_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Add a unique chunk_id to every chunk.
    """
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def download_embeddings():
    """
    Load the HuggingFace embedding model.

    all-MiniLM-L6-v2 returns vectors with 384 dimensions.
    Pinecone index dimension must match this value.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )

    return embeddings


def format_source_documents(docs: List[Document]) -> List[dict]:
    """
    Format retrieved source documents for displaying citations in the UI.
    Removes duplicate source/page/chunk combinations.
    """
    sources = []
    seen_sources = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        domain = doc.metadata.get("domain", "Unknown domain")
        chunk_id = doc.metadata.get("chunk_id", "Unknown chunk")

        source_name = Path(source).name if source else "Unknown source"
        source_key = (source_name, page, chunk_id)

        if source_key not in seen_sources:
            seen_sources.add(source_key)

            sources.append(
                {
                    "source": source_name,
                    "page": page,
                    "domain": domain,
                    "chunk_id": chunk_id
                }
            )

    return sources