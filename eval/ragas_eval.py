import os, json, argparse, asyncio
from pathlib import Path
from datasets import load_dataset, Dataset as HFDataset
import evaluate as hf_eval
from langchain_core.prompts import ChatPromptTemplate
from ragas.metrics import faithfulness, answer_relevancy
try:
    from ragas.llms.base import LangchainLLMWrapper
except Exception:
    from ragas.llms import LangchainLLM as LangchainLLMWrapper
from src.config.settings import settings
from src.llm.chat import get_chat
from src.llm.embeddings import get_embeddings

os.environ.setdefault("OLLAMA_HOST", settings.ollama_host)

def pick_cols(cols):
    # Heuristic column picker: choose likely columns for question, ground-truth and contexts
    q  = next(c for c in cols if c in ("question","input","prompt","instruction","text"))
    gt = next((c for c in cols if c in ("ground_truth","ground_truths","answers","target","output")), None)
    gc = next((c for c in cols if c in ("ground_truth_context","contexts","context")), None)
    return q, gt, gc

def as_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, str) and x.strip().startswith("["):
        try: return json.loads(x)
        except: pass
    return [x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aurelio-ai/ai-arxiv2-ragas-mixtral")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ctx_chars", type=int, default=500)
    ap.add_argument("--out", default="ragas_eval_out/predictions.jsonl")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    q_col, gt_col, gc_col = pick_cols(ds.column_names)

    chat = get_chat()                                # Ollama chat
    judge = LangchainLLMWrapper(chat)                # bind judge to Ollama
    emb = get_embeddings()                           # Ollama embeddings
    faithfulness.llm = judge; faithfulness.embeddings = emb
    answer_relevancy.llm = judge; answer_relevancy.embeddings = emb

    prompt = ChatPromptTemplate.from_messages([("human","Question: {q}\n\nContext:\n{ctx}\n\nAnswer:")])

    rows, preds, refs = [], [], []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for i, s in enumerate(ds):
            if i >= args.limit: break
            q = s[q_col] if not isinstance(s[q_col], list) else "\n".join(s[q_col])
            ctxs = [c[:args.ctx_chars] for c in as_list(s.get(gc_col))][:args.k]  # oracle contexts (no FAISS)
            pred = chat.invoke(prompt.format(q=q, ctx="\n\n".join(ctxs))).content
            gt = None if gt_col is None else (s[gt_col][0] if isinstance(s[gt_col], list) else s[gt_col])
            rows.append({"user_input": q, "response": pred, "retrieved_contexts": ctxs})
            if gt is not None: preds.append(pred); refs.append(gt)
            f.write(json.dumps({"input": q, "ground_truth": gt, "prediction": pred}, ensure_ascii=False) + "\n")

    async def score(r):
        fa = await faithfulness._ascore(r, callbacks=None)
        ar = await answer_relevancy._ascore(r, callbacks=None)
        return fa, ar

    # Run scoring concurrently and handle empty row set
    if rows:
        async def _all_scores(rs):
            return await asyncio.gather(*(score(r) for r in rs))

        fa_ar_pairs = asyncio.run(_all_scores(rows))
        fa_list, ar_list = zip(*fa_ar_pairs)
    else:
        fa_list, ar_list = [], []

    bleu = hf_eval.load("bleu").compute(predictions=preds, references=refs)["bleu"] if refs else float("nan")
    rougeL = hf_eval.load("rouge").compute(predictions=preds, references=refs, use_stemmer=True).get("rougeL", float("nan")) if refs else float("nan")

    HFDataset.from_list([
        {"question": r["user_input"], "answer": r["response"], "contexts": r["retrieved_contexts"],
         "faithfulness": (fa_list[i] if fa_list else float("nan")), "answer_relevancy": (ar_list[i] if ar_list else float("nan"))}
        for i, r in enumerate(rows)
    ]).to_pandas().to_csv(Path(args.out).with_name("per_sample_scores.csv"), index=False)

    # Compute averages while excluding NaNs from both numerator and denominator
    valid_fa = [x for x in fa_list if x == x]
    valid_ar = [x for x in ar_list if x == x]
    agg = {
        "samples": len(rows),
        "BLEU": round(bleu, 4),
        "ROUGE-L": round(rougeL, 4),
        "faithfulness_avg": round(sum(valid_fa)/len(valid_fa), 4) if valid_fa else float("nan"),
        "answer_relevancy_avg": round(sum(valid_ar)/len(valid_ar), 4) if valid_ar else float("nan"),
    }
    Path(Path(args.out).parent, "aggregate_scores.json").write_text(json.dumps(agg, indent=2), "utf-8")
    print(f'Samples:{agg["samples"]} | BLEU:{agg["BLEU"]:.4f} | ROUGE-L:{agg["ROUGE-L"]:.4f} | '
          f'Faithfulness:{agg["faithfulness_avg"]:.4f} | AnswerRel:{agg["answer_relevancy_avg"]:.4f}')
    print(f"Saved predictions → {args.out}")

if __name__ == "__main__":
    main()
