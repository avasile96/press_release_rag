from pathlib import Path
from src.models.schemas import Document

"""
Load raw data, normalize into Document
"""

def load_plaintext(file_path: Path) -> list[Document]:
    """Load a plaintext file and convert paragraphs into Document objects.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to a plaintext file where press releases are separated by
        one or more blank lines.

    Returns
    -------
    list[Document]
        A list of `Document` objects created from each paragraph. Each
        Document will have an auto-generated id, title (first sentence or
        truncated), text, and source (the input filename).
    """

    text = Path(file_path).read_text(encoding="utf-8")
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    docs = []
    for i, p in enumerate(paras):
        title = p.split(".")[0][:120] if "." in p else p[:120]
        docs.append(Document(id=f"pr-{i:04d}", text=p, title=title, source=str(file_path)))
    return docs
