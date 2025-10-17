from pathlib import Path
from src.models.schemas import Document

"""
Load raw data, normalize into Document
"""

def load_plaintext(file_path: Path) -> list[Document]:
    text = Path(file_path).read_text(encoding="utf-8")
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    docs = []
    for i, p in enumerate(paras):
        title = p.split(".")[0][:120] if "." in p else p[:120]
        docs.append(Document(id=f"pr-{i:04d}", text=p, title=title, source=str(file_path)))
    return docs
