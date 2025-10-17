from src.config.settings import settings
from src.models.schemas import Document, Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    for d in docs:
        if len(d.text) <= 400:
            chunks.append(Chunk(doc_id=d.id, chunk_id=f"{d.id}::c0",
                                text=d.text, meta={"source": d.source, "title": d.title}))
        else:
            for j, piece in enumerate(splitter.split_text(d.text)):
                chunks.append(Chunk(doc_id=d.id, chunk_id=f"{d.id}::c{j}",
                                    text=piece, meta={"source": d.source, "title": d.title}))
    return chunks
