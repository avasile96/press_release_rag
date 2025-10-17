import sys
from src.rag.chain import build_rag_chain

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What did Nimbus do for students?"
    chain = build_rag_chain()
    print(chain.invoke(q))
