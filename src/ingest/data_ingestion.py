from pathlib import Path
from src.models.schemas import Document

"""
Load raw data, normalize into Document
"""

def load_plaintext(file_path: Path) -> list[Document]:
    text = Path(file_path).read_text(encoding="utf-8")
    
    docs = []
    for i, block in enumerate(text.split("\n\n\n")):
        if not block.strip():
            continue
        docs.append(Document(id=f"doc-{i:04d}", text=block, source=str(file_path)))
    return docs
