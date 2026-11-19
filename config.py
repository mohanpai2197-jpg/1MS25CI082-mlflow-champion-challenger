"""Shared settings for the Champion-Challenger QA project.

Everything that might change between machines lives here, so no script
has hard-coded run IDs or paths.
"""
import os

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "1000")

STUDENT_USN = os.getenv("STUDENT_USN", "1MS25CI082")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "hf-qa-champion-challenger"
REGISTERED_MODEL_NAME = "qa-model"

TASK = "qa"
DATASET_NAME = "rajpurkar/squad_v2"
DATASET_SPLIT = "validation"
EVAL_SAMPLES = int(os.getenv("EVAL_SAMPLES", "100"))
SEED = 42

# Model A = current Champion, Model B = Challenger (pairing 7 vs 6 in the brief)
MODEL_A = {
    "label": "A",
    "role": "champion",
    "hf_model_id": "deepset/minilm-uncased-squad2",
    "short_name": "minilm-squad2",
}
MODEL_B = {
    "label": "B",
    "role": "challenger",
    "hf_model_id": "deepset/roberta-base-squad2",
    "short_name": "roberta-base-squad2",
}

# Inference settings, identical for both models
MAX_SEQ_LEN = 384
DOC_STRIDE = 128
MAX_ANSWER_LEN = 30
HANDLE_IMPOSSIBLE_ANSWER = True  # SQuAD v2 has unanswerable questions

# ---- Champion rule (written BEFORE the final run, do not edit after) ----
PRIMARY_METRIC = "f1"  # SQuAD v2 token-level F1, higher is better
MARGIN = 2.0  # Challenger must win by >= 2.0 F1 points
LATENCY_GUARDRAIL_X = 5.0  # reject Challenger if > 5x Champion latency

DATA_DIR = "data"
OUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
