from datasets import load_dataset
from src.rag.chain import build_rag_chain
from src.retrieval.retriever import get_retriever

ragas_data = load_dataset("aurelio-ai/ai-arxiv2-ragas-mixtral", split="train")
# print(ragas_data)

# question = "What does Nimbus Mobile do in eastern europe?"

retriever = get_retriever()
# docs = retriever.invoke(question)

retriever = get_retriever()
retriever.search_kwargs["k"] = 2
chain = build_rag_chain()
# answer = chain.invoke(question)
# docs = retriever.invoke(question)

import pandas as pd
from tqdm.auto import tqdm

df = pd.DataFrame({
    "question": [],
    "contexts": [],
    "answer": [],
    "ground_truth": []
})


from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
)

eval_data = Dataset.from_dict(df)
print(eval_data)

from ragas import evaluate

result = evaluate(
    dataset=eval_data,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness,
    ],
)
result = result.to_pandas()

print(result)