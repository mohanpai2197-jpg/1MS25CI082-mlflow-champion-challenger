"""Load whichever model holds the `champion` alias and answer new questions.

This script never imports transformers: prediction goes through MLflow only.
It does not know which version is the Champion; the registry tells it.

Usage:
  python predict_from_mlflow.py                 # champion
  python predict_from_mlflow.py --alias challenger
"""
import argparse
import os
import mlflow
import pandas as pd
from mlflow import MlflowClient

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args()

    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()

    model_uri = f"models:/{C.REGISTERED_MODEL_NAME}@{args.alias}"
    mv = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, args.alias)
    hf_id = mv.tags.get("hf_model_id", "unknown")

    print(f"Model URI         : {model_uri}")
    print(f"Registry version  : {mv.version}")
    print(f"Source run        : {mv.tags.get('source_run', mv.run_id)}")
    print(f"Hugging Face ID   : {hf_id}")

    model = mlflow.pyfunc.load_model(model_uri)
    new_df = pd.read_csv(os.path.join(C.DATA_DIR, "new_data.csv"))
    preds = model.predict(new_df[["question", "context"]])

    out = preds.copy()
    out["answer"] = out["answer"].replace("", "<no answer>")
    out.insert(0, "alias", args.alias)
    out["model_version"] = mv.version
    out["hf_model_id"] = hf_id

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 160)
    print("\nPredictions on new data:")
    print(out[["question", "answer", "score"]].to_string(index=False))

    os.makedirs(C.OUT_DIR, exist_ok=True)
    name = "new_data_predictions.csv" if args.alias == "champion" else f"new_data_predictions_{args.alias}.csv"
    out.to_csv(os.path.join(C.OUT_DIR, name), index=False)
    print(f"\nSaved {os.path.join(C.OUT_DIR, name)}")


if __name__ == "__main__":
    main()
