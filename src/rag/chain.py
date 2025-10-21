from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from src.retrieval.retriever import get_retriever
from src.retrieval.prompt import SYSTEM_PROMPT
from src.llm.chat import get_chat

def _format_docs(docs):
    """Format retrieved documents into a human-readable context string.

    Parameters
    ----------
    docs : Iterable
        Iterable of document-like objects returned by the retriever. Each
        item is expected to have `metadata` and `page_content` attributes.

    Returns
    -------
    str
        A single string that enumerates documents with their
        snippets, suitable for insertion into a prompt.
    """

    out = []
    for i, d in enumerate(docs, 1):
        text = d.metadata.get("text") or d.metadata.get("doc_id", "")
        out.append(f"[{i}] {text} (`{d.metadata.get('doc_id')}`) — {d.page_content}")
    return "\n\n".join(out)

def build_rag_chain():
    """Build a retrieval-augmented generation (RAG) runnable chain.

    The returned chain composes a retriever, prompt template, and chat
    model into a runnable pipeline that accepts a `question` and returns a
    parsed string answer.

    Returns
    -------
    Runnable
        A LangChain runnable pipeline ready to be invoked with inputs such
        as `{"question": "..."}`.
    """

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
