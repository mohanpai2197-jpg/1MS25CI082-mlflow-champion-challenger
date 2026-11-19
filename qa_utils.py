"""Metrics and the MLflow pyfunc wrapper for extractive question answering."""
import re
import string
from collections import Counter

import mlflow.pyfunc
import pandas as pd


# ---------------- SQuAD v2 metrics (same logic as the official script) ----------------
def normalize_answer(text: str) -> str:
    """Lower-case, drop punctuation, articles and extra whitespace."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    # Unanswerable question: both must be empty to score 1
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_references(prediction: str, references: list[str]) -> tuple[float, float]:
    """SQuAD scores against every gold answer and keeps the best one."""
    if not references:
        references = [""]
    em = max(exact_match(prediction, r) for r in references)
    f1 = max(f1_score(prediction, r) for r in references)
    return em, f1


# ---------------- MLflow pyfunc wrapper (Lab 9 style) ----------------
class HFQuestionAnsweringModel(mlflow.pyfunc.PythonModel):
    """Wraps a Hugging Face QA pipeline so it can be served from MLflow.

    Input : DataFrame with columns `question` and `context`
    Output: DataFrame with `question`, `answer`, `score`, `is_unanswerable`
    """

    def load_context(self, context):
        from transformers import pipeline  # imported here so callers never touch HF

        cfg = context.model_config or {}
        self.settings = {
            "max_seq_len": int(cfg.get("max_seq_len", 384)),
            "doc_stride": int(cfg.get("doc_stride", 128)),
            "max_answer_len": int(cfg.get("max_answer_len", 30)),
            "handle_impossible_answer": bool(cfg.get("handle_impossible_answer", True)),
        }
        self.qa = pipeline(
            "question-answering",
            model=context.artifacts["hf_model"],
            tokenizer=context.artifacts["hf_model"],
            device=-1,
        )

    def predict(self, context, model_input, params=None):
        df = pd.DataFrame(model_input)
        rows = []
        for q, c in zip(df["question"], df["context"]):
            out = self.qa(question=str(q), context=str(c), **self.settings)
            answer = out["answer"].strip()
            rows.append(
                {
                    "question": q,
                    "answer": answer,
                    "score": round(float(out["score"]), 4),
                    "is_unanswerable": answer == "",
                }
            )
        return pd.DataFrame(rows)
