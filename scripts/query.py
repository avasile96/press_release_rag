import sys
from src.rag.chain import build_rag_chain

"""
Quick terminal check.
"""

chain = build_rag_chain()
print(chain.invoke(" ".join(sys.argv[1:])))
