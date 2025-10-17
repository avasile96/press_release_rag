import streamlit as st
from src.rag.chain import build_rag_chain
from src.retrieval.retriever import get_retriever

st.set_page_config(page_title="Nimbus RAG", layout="wide")
st.title("Nimbus Mobile — Press Releases RAG")

question = st.text_input("Ask a question:")
k = st.slider("Top-k passages", 2, 12, 6)

if st.button("Search") and question.strip():
    retriever = get_retriever()
    retriever.search_kwargs["k"] = k
    chain = build_rag_chain()
    with st.spinner("Thinking…"):
        answer = chain.invoke(question)
    st.markdown("### Answer")
    st.write(answer)
    with st.expander("Show retrieved context"):
        docs = retriever.invoke(question)
        for d in docs:
            title = d.metadata.get("title") or d.metadata.get("doc_id","?")
            st.markdown(f"**{title}**  \n`{d.metadata.get('doc_id','?')}` — {d.metadata.get('source','')}")
            st.write(d.page_content)
            st.divider()
