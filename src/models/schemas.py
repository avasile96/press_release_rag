from pydantic import BaseModel
from typing import List, Optional

"""
Data contracts for clarity
"""

class Document(BaseModel):
    id: str
    text: str
    source: Optional[str] = None
    date: Optional[str] = None

class Chunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    meta: dict

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    question: str
    answer: str
    sources: List[dict]  # [{source, score, text}]
