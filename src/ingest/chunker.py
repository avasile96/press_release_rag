# src/ingest/chunkers.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import settings
from src.models.schemas import Document, Chunk

def chunk_documents(docs: list[Document]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks: list[Chunk] = []
    for d in docs:
        for j, piece in enumerate(splitter.split_text(d.text)):
            chunks.append(Chunk(
                doc_id=d.id, chunk_id=f"{d.id}::c{j}",
                text=piece, meta={"source": d.source}
            ))
    return chunks
