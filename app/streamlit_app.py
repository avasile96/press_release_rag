import streamlit as st
from src.rag.chain import build_rag_chain
from src.retrieval.retriever import get_retriever

"""
One box, one answer, sources under it.
"""
st.set_page_config(page_title="Telekom RAG", layout="wide")
st.title("Deutsche Telekom — Press Releases RAG")

question = st.text_input("Ask a question about the announcements:")
k = st.slider("Top-k passages", 2, 10, 4)

if "chain" not in st.session_state:
    st.session_state.chain = build_rag_chain()

if st.button("Search") and question.strip():
    # temporarily override k at runtime
    retriever = get_retriever()
    retriever.search_kwargs["k"] = k
    chain = build_rag_chain()

    with st.spinner("Thinking…"):
        answer = chain.invoke(question)

    st.markdown("### Answer")
    st.write(answer)

    # Show sources
    with st.expander("Show retrieved context"):
        for d in retriever.get_relevant_documents(question):
            st.markdown(f"**{d.metadata.get('doc_id','?')}** — {d.metadata.get('source','')}")
            st.write(d.page_content[:800] + ("…" if len(d.page_content) > 800 else ""))
            st.divider()
