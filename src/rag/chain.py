from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from src.retrieval.retriever import get_retriever
from src.retrieval.prompt import SYSTEM_PROMPT
from src.llm.chat import get_chat

def _format_docs(docs):
    out = []
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title") or d.metadata.get("doc_id", "")
        out.append(f"[{i}] {title} (`{d.metadata.get('doc_id')}`) — {d.page_content}")
    return "\n\n".join(out)

def build_rag_chain():
    retriever = get_retriever()
    chat = get_chat()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])
    chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnableMap({"context": lambda x: _format_docs(x["docs"]), "question": lambda x: x["question"]})
        | prompt
        | chat
        | StrOutputParser()
    )
    return chain
