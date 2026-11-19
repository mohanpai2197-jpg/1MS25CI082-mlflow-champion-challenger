"""Run the Champion (A) and Challenger (B) QA models on the same SQuAD v2 subset
and log everything to MLflow.

Runs created (one experiment):
  compare_qa             parent / summary run with the decision
    A_qa_minilm-squad2   nested run for Model A
    B_qa_roberta-base-squad2 nested run for Model B

Usage:
  python evaluate.py
"""
import gc
import json
import os
import platform
import random
import time
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "1000")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

import config as C
from qa_utils import HFQuestionAnsweringModel, best_over_references

os.makedirs(C.DATA_DIR, exist_ok=True)
os.makedirs(C.PLOT_DIR, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- data
def build_eval_set() -> pd.DataFrame:
    """Pick a fixed, seeded subset of SQuAD v2 validation and save it."""
    ds = load_dataset(C.DATASET_NAME, split=C.DATASET_SPLIT)
    ds = ds.shuffle(seed=C.SEED).select(range(C.EVAL_SAMPLES))
    rows = []
    for ex in ds:
        refs = list(dict.fromkeys(ex["answers"]["text"]))  # unique, keep order
        rows.append(
            {
                "id": ex["id"],
                "title": ex["title"],
                "question": ex["question"],
                "context": ex["context"],
                "references": json.dumps(refs, ensure_ascii=False),
                "is_answerable": len(refs) > 0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.DATA_DIR, "eval_inputs.csv"), index=False)
    return df


# ---------------------------------------------------------------- inference
def run_model(model_cfg: dict, eval_df: pd.DataFrame):
    """Download the model once, run it on every eval row, return predictions."""
    local_dir = os.path.join("hf_models", model_cfg["short_name"])
    if not os.path.isdir(local_dir):
        AutoTokenizer.from_pretrained(model_cfg["hf_model_id"]).save_pretrained(local_dir)
        AutoModelForQuestionAnswering.from_pretrained(model_cfg["hf_model_id"]).save_pretrained(local_dir)
    qa = pipeline("question-answering", model=local_dir, tokenizer=local_dir, device=-1)
    settings = dict(
        max_seq_len=C.MAX_SEQ_LEN,
        doc_stride=C.DOC_STRIDE,
        max_answer_len=C.MAX_ANSWER_LEN,
        handle_impossible_answer=C.HANDLE_IMPOSSIBLE_ANSWER,
    )
    qa(question="warm up?", context="This is a warm up call.", **settings)  # not timed
    rows, latencies = [], []
    for _, r in eval_df.iterrows():
        t0 = time.perf_counter()
        out = qa(question=r["question"], context=r["context"], **settings)
        latencies.append((time.perf_counter() - t0) * 1000)
        pred = out["answer"].strip()
        refs = json.loads(r["references"])
        em, f1 = best_over_references(pred, refs)
        rows.append(
            {
                "id": r["id"],
                "question": r["question"],
                "references": " | ".join(refs) if refs else "<no answer>",
                "prediction": pred if pred else "<no answer>",
                "score": round(float(out["score"]), 4),
                "is_answerable": r["is_answerable"],
                "exact_match": em,
                "f1": round(f1, 4),
                "latency_ms": round(latencies[-1], 2),
            }
        )
    del qa  # free memory before the next model is loaded
    gc.collect()
    return pd.DataFrame(rows), latencies, local_dir


def compute_metrics(pred_df: pd.DataFrame, latencies: list[float]) -> dict:
    has = pred_df[pred_df["is_answerable"]]
    no = pred_df[~pred_df["is_answerable"]]
    return {
        "exact_match": round(100 * pred_df["exact_match"].mean(), 2),
        "f1": round(100 * pred_df["f1"].mean(), 2),
        "has_ans_f1": round(100 * has["f1"].mean(), 2) if len(has) else 0.0,
        "no_ans_accuracy": round(100 * no["exact_match"].mean(), 2) if len(no) else 0.0,
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "total_inference_s": round(float(np.sum(latencies)) / 1000, 3),
    }


# ---------------------------------------------------------------- plots
def plot_metric_bars(metrics_a, metrics_b, path):
    keys = ["exact_match", "f1", "has_ans_f1", "no_ans_accuracy"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, [metrics_a[k] for k in keys], 0.4, label=f"A {C.MODEL_A['short_name']}")
    ax.bar(x + 0.2, [metrics_b[k] for k in keys], 0.4, label=f"B {C.MODEL_B['short_name']}")
    ax.set_xticks(x, ["Exact match", "F1", "F1 (answerable)", "No-answer acc."])
    ax.set_ylabel("Score (%) - higher is better")
    ax.set_ylim(0, 100)
    ax.set_title("Champion vs Challenger on every quality metric")
    ax.legend()
    for i, k in enumerate(keys):
        ax.text(i - 0.2, metrics_a[k] + 1, f"{metrics_a[k]:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, metrics_b[k] + 1, f"{metrics_b[k]:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def plot_f1_distribution(pred_a, pred_b, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 11)
    ax.hist(
        [pred_a["f1"], pred_b["f1"]],
        bins=bins,
        label=[f"A {C.MODEL_A['short_name']}", f"B {C.MODEL_B['short_name']}"],
    )
    ax.set_xlabel("Per-question F1")
    ax.set_ylabel("Number of questions")
    ax.set_title("Per-example F1 distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def plot_latency(lat_a, lat_b, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([lat_a, lat_b], showfliers=False)
    ax.set_xticks([1, 2], [f"A {C.MODEL_A['short_name']}", f"B {C.MODEL_B['short_name']}"])
    ax.set_ylabel("Latency per question (ms) - lower is better")
    ax.set_title("Inference latency on CPU")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


# ---------------------------------------------------------------- decision
def decide(metrics_a, metrics_b) -> dict:
    """Apply the Champion rule from config.py / README (fixed before the run)."""
    delta = round(metrics_b[C.PRIMARY_METRIC] - metrics_a[C.PRIMARY_METRIC], 2)
    latency_ratio = round(metrics_b["avg_latency_ms"] / metrics_a["avg_latency_ms"], 2)
    if latency_ratio > C.LATENCY_GUARDRAIL_X:
        winner, reason = "A", f"Challenger rejected by latency guardrail ({latency_ratio}x > {C.LATENCY_GUARDRAIL_X}x)"
    elif delta >= C.MARGIN:
        winner, reason = "B", f"Challenger beats Champion by {delta} F1 points (>= margin {C.MARGIN})"
    elif delta > -C.MARGIN:
        # within margin: tie-breaker = prefer the faster model
        faster = "A" if metrics_a["avg_latency_ms"] <= metrics_b["avg_latency_ms"] else "B"
        winner, reason = faster, f"Difference {delta} is within the {C.MARGIN}-point margin; tie-breaker picks the faster model"
    else:
        winner, reason = "A", f"Challenger is worse by {abs(delta)} F1 points"
    return {"winner": winner, "delta": delta, "latency_ratio": latency_ratio, "reason": reason}


def disagreements(pred_a, pred_b, n=5) -> pd.DataFrame:
    merged = pred_a[["id", "question", "references", "prediction", "f1"]].merge(
        pred_b[["id", "prediction", "f1"]], on="id", suffixes=("_A", "_B")
    )
    diff = merged[merged["prediction_A"].str.lower() != merged["prediction_B"].str.lower()].copy()
    diff["f1_gap"] = (diff["f1_B"] - diff["f1_A"]).abs()
    diff = diff.sort_values("f1_gap", ascending=False).head(n)
    diff["comment"] = ""  # filled in by hand after reading the cases (see README)
    return diff.drop(columns=["f1_gap"])


# ---------------------------------------------------------------- MLflow logging
def log_model_run(model_cfg, pred_df, metrics, local_dir, eval_df, common_tags):
    run_name = f"{model_cfg['label']}_{C.TASK}_{model_cfg['short_name']}"
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params(
            {
                "hf_model_id": model_cfg["hf_model_id"],
                "task": "question-answering",
                "role": f"{model_cfg['role']} candidate",
                "max_seq_len": C.MAX_SEQ_LEN,
                "doc_stride": C.DOC_STRIDE,
                "max_answer_len": C.MAX_ANSWER_LEN,
                "handle_impossible_answer": C.HANDLE_IMPOSSIBLE_ANSWER,
                "eval_samples": len(eval_df),
                "seed": C.SEED,
                "device": "cpu",
            }
        )
        mlflow.set_tags({**common_tags, "hf_model_id": model_cfg["hf_model_id"], "model_label": model_cfg["label"]})
        mlflow.log_metrics(metrics)
        pred_path = os.path.join(C.OUT_DIR, f"predictions_{model_cfg['short_name']}.csv")
        pred_df.to_csv(pred_path, index=False)
        metrics_path = os.path.join(C.OUT_DIR, f"metrics_summary_{model_cfg['short_name']}.json")
        with open(metrics_path, "w") as f:
            json.dump({"hf_model_id": model_cfg["hf_model_id"], **metrics}, f, indent=2)
        mlflow.log_artifact(pred_path, artifact_path="outputs")
        mlflow.log_artifact(metrics_path, artifact_path="outputs")
        mlflow.log_artifact(os.path.join(C.DATA_DIR, "eval_inputs.csv"), artifact_path="data")
        mlflow.log_table(pred_df, artifact_file="tables/predictions.json")
        # Explicit signature, so MLflow does not reload the model just to infer it
        signature = infer_signature(
            eval_df[["question", "context"]].head(2),
            pd.DataFrame({"question": ["q"], "answer": ["a"], "score": [0.5], "is_unanswerable": [False]}),
        )
        mlflow.pyfunc.log_model(
            name="model",
            python_model=HFQuestionAnsweringModel(),
            artifacts={"hf_model": local_dir},
            code_paths=["qa_utils.py"],
            signature=signature,
            model_config={
                "max_seq_len": C.MAX_SEQ_LEN,
                "doc_stride": C.DOC_STRIDE,
                "max_answer_len": C.MAX_ANSWER_LEN,
                "handle_impossible_answer": C.HANDLE_IMPOSSIBLE_ANSWER,
                "hf_model_id": model_cfg["hf_model_id"],
            },
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"transformers=={transformers.__version__}",
                f"torch=={torch.__version__.split('+')[0]}",
                "pandas",
            ],
        )
        return run.info.run_id


def main():
    set_seed(C.SEED)
    mlflow.set_tracking_uri(C.TRACKING_URI)
    mlflow.set_experiment(C.EXPERIMENT_NAME)
    print(f"Loading {C.EVAL_SAMPLES} questions from {C.DATASET_NAME} ({C.DATASET_SPLIT}) ...")
    eval_df = build_eval_set()
    print(f"Running Model A: {C.MODEL_A['hf_model_id']}")
    pred_a, lat_a, dir_a = run_model(C.MODEL_A, eval_df)
    print(f"Running Model B: {C.MODEL_B['hf_model_id']}")
    pred_b, lat_b, dir_b = run_model(C.MODEL_B, eval_df)
    m_a, m_b = compute_metrics(pred_a, lat_a), compute_metrics(pred_b, lat_b)
    decision = decide(m_a, m_b)
    common_tags = {
        "student_usn": C.STUDENT_USN,
        "dataset_name": f"{C.DATASET_NAME}:{C.DATASET_SPLIT}",
        "library_versions": f"transformers={transformers.__version__}, torch={torch.__version__}, mlflow={mlflow.__version__}",
        "python_version": platform.python_version(),
    }
    with mlflow.start_run(run_name=f"compare_{C.TASK}") as parent:
        run_a = log_model_run(C.MODEL_A, pred_a, m_a, dir_a, eval_df, common_tags)
        run_b = log_model_run(C.MODEL_B, pred_b, m_b, dir_b, eval_df, common_tags)
        # Summary of the decision
        mlflow.log_params(
            {
                "primary_metric": C.PRIMARY_METRIC,
                "margin": C.MARGIN,
                "latency_guardrail_x": C.LATENCY_GUARDRAIL_X,
                "tie_breaker": "faster model (lower avg_latency_ms)",
                "champion_model_id": C.MODEL_A["hf_model_id"],
                "challenger_model_id": C.MODEL_B["hf_model_id"],
                "eval_samples": len(eval_df),
                "seed": C.SEED,
            }
        )
        winner_cfg = C.MODEL_A if decision["winner"] == "A" else C.MODEL_B
        mlflow.set_tags(
            {
                **common_tags,
                "winner": f"{winner_cfg['label']}_{C.TASK}_{winner_cfg['short_name']}",
                "winner_label": decision["winner"],
                "winner_hf_model_id": winner_cfg["hf_model_id"],
                "decision_reason": decision["reason"],
                "run_id_A": run_a,
                "run_id_B": run_b,
            }
        )
        mlflow.log_metrics(
            {
                "delta_primary_metric": decision["delta"],
                "latency_ratio_B_over_A": decision["latency_ratio"],
                **{f"A_{k}": v for k, v in m_a.items()},
                **{f"B_{k}": v for k, v in m_b.items()},
            }
        )
        # Shared artifacts
        summary = {
            "task": "extractive question answering (SQuAD v2)",
            "eval_samples": len(eval_df),
            "metric_direction": {
                "exact_match": "higher is better",
                "f1": "higher is better (PRIMARY)",
                "has_ans_f1": "higher is better",
                "no_ans_accuracy": "higher is better",
                "avg_latency_ms": "lower is better",
                "total_inference_s": "lower is better",
            },
            "A": {"hf_model_id": C.MODEL_A["hf_model_id"], **m_a},
            "B": {"hf_model_id": C.MODEL_B["hf_model_id"], **m_b},
            "decision": decision,
        }
        with open(os.path.join(C.OUT_DIR, "metrics_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(os.path.join(C.OUT_DIR, "metrics_summary.json"))
        figs = {
            "metric_comparison.png": plot_metric_bars(m_a, m_b, os.path.join(C.PLOT_DIR, "metric_comparison.png")),
            "f1_distribution.png": plot_f1_distribution(pred_a, pred_b, os.path.join(C.PLOT_DIR, "f1_distribution.png")),
            "latency_boxplot.png": plot_latency(lat_a, lat_b, os.path.join(C.PLOT_DIR, "latency_boxplot.png")),
        }
        for name, fig in figs.items():
            mlflow.log_figure(fig, f"plots/{name}")
            plt.close(fig)
        dis = disagreements(pred_a, pred_b)
        dis_path = os.path.join(C.OUT_DIR, "disagreements.csv")
        if not os.path.exists(dis_path):  # keep hand-written comments on reruns
            dis.to_csv(dis_path, index=False)
        mlflow.log_artifact(dis_path, artifact_path="outputs")
        mlflow.log_artifact(os.path.join(C.DATA_DIR, "eval_inputs.csv"), artifact_path="data")
        if os.path.exists("README.md"):
            mlflow.log_artifact("README.md")
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt")
        print("\n=========== RESULTS ===========")
        print(pd.DataFrame({"A (champion)": m_a, "B (challenger)": m_b}).to_string())
        print(f"\nDecision: winner = Model {decision['winner']} -> {decision['reason']}")
        print(f"Parent run: compare_{C.TASK} ({parent.info.run_id})")
        print("Next: python register_champion.py")


if __name__ == "__main__":
    main()
