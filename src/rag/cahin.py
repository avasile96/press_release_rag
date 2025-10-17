from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from src.retrieval.retriever import get_retriever
from src.retrieval.prompt import SYSTEM_PROMPT
from src.llm.chat import get_chat

"""
“retrieve → stuff → generate” (simple, dependable).
"""

def format_docs(docs):
    return "\n\n".join(
        f"[{i}] {d.metadata.get('doc_id')} — {d.page_content}"
        for i, d in enumerate(docs, 1)
    )

def build_rag_chain():
    retriever = get_retriever()
    chat = get_chat()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])

    chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnableMap({"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]})
        | prompt
        | chat
        | StrOutputParser()
    )
    return chain
