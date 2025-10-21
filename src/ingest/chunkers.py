from src.config.settings import settings
from src.models.schemas import Document, Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Split documents into chunks suitable for embedding and indexing.

    Parameters
    ----------
    docs : list[Document]
        List of Document objects to split into smaller chunks. Uses
        `RecursiveCharacterTextSplitter` and the `chunk_size` and
        `chunk_overlap` values from configuration.

    Returns
    -------
    list[Chunk]
        A list of Chunk objects, each containing `doc_id`, `chunk_id`, `text`,
        and `meta` fields derived from the source Document.
    """

    chunks: list[Chunk] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    for d in docs:

        text = (d.text or "").strip()
        if not text:
            continue

        # Atomic short entries
        if len(text.split()) <= settings.short_entry_words:
            pieces = [text]
        else:
            print(text)
            pieces = [p for p in splitter.split_text(text) if p and p.strip()]
        
        # Base meta
        base_meta = {}
        if hasattr(d, "meta") and isinstance(d.meta, dict):
            base_meta.update(d.meta)
        if getattr(d, "source", None) is not None:
            base_meta.setdefault("source", d.source)

        for j, piece in enumerate(pieces):
            meta = dict(base_meta)
            meta.update({
                "chunk_index": j,
                "total_chunks": len(pieces),
                "word_count": len(piece.split()),
            })
            chunks.append(
                Chunk(
                    doc_id=d.id,
                    chunk_id=f"{d.id}::c{j}",
                    text=piece,
                    meta=meta,
                )
            )
    return chunks
