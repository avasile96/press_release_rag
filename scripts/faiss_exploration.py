import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

with open("data/vectorstore/index.pkl", "rb") as f:
    mapping = pickle.load(f)
type(mapping), len(mapping)
# e.g., print a few entries
if isinstance(mapping, dict):
    for k in list(mapping)[:5]:
        print(k, mapping[k])
else:
    print(mapping[:5])
model = SentenceTransformer("all-MiniLM-L6-v2")   # MUST match the embedding model used previously
q = "Latest Nimbus Mobile 5G rollout"
emb = model.encode([q], convert_to_numpy=True)
emb = emb.astype("float32")
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

D, I = index.search(emb, k=5)  # distances, indices