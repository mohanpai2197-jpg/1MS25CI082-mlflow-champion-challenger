"""Register Model A and Model B under one registered model name and point the
`champion` / `challenger` aliases at the right versions.

Nothing is hard-coded: the script finds the latest compare_qa run by name,
reads the winner tag that evaluate.py wrote, and finds the two child runs.

Usage:
  python register_champion.py          # follow the decision from evaluate.py
  python register_champion.py --flip   # stretch goal: swap the two aliases
"""
import argparse
import mlflow
from mlflow import MlflowClient

import config as C


def latest_run(client, experiment_id, run_name, parent_id=None):
    filt = f"attributes.run_name = '{run_name}'"
    if parent_id:
        filt += f" and tags.mlflow.parentRunId = '{parent_id}'"
    runs = client.search_runs(
        [experiment_id],
        filter_string=filt,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"No run named {run_name!r} found. Run evaluate.py first.")
    return runs[0]


def find_version(client, run_id):
    for mv in client.search_model_versions(f"name = '{C.REGISTERED_MODEL_NAME}'"):
        if mv.run_id == run_id:
            return mv
    return None


def register(client, run, model_cfg):
    """Register the run's model once (re-running the script will not duplicate it)."""
    mv = find_version(client, run.info.run_id)
    if mv is None:
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", C.REGISTERED_MODEL_NAME)
    client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "hf_model_id", model_cfg["hf_model_id"])
    client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "model_label", model_cfg["label"])
    client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "source_run", run.info.run_name)
    client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "f1", str(run.data.metrics.get("f1")))
    return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip", action="store_true", help="swap champion/challenger aliases")
    args = ap.parse_args()

    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()

    exp = client.get_experiment_by_name(C.EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit("Experiment not found. Run evaluate.py first.")

    if args.flip:
        champ = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, "champion")
        chall = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, "challenger")
        client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "champion", chall.version)
        client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "challenger", champ.version)
        print(f"Flipped: champion -> v{chall.version}, challenger -> v{champ.version}")
        return

    parent = latest_run(client, exp.experiment_id, f"compare_{C.TASK}")
    run_a = latest_run(client, exp.experiment_id, f"A_{C.TASK}_{C.MODEL_A['short_name']}", parent.info.run_id)
    run_b = latest_run(client, exp.experiment_id, f"B_{C.TASK}_{C.MODEL_B['short_name']}", parent.info.run_id)

    try:
        client.create_registered_model(
            C.REGISTERED_MODEL_NAME,
            description="Extractive QA (SQuAD v2). Champion-Challenger: MiniLM vs RoBERTa-base.",
        )
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    mv_a = register(client, run_a, C.MODEL_A)
    mv_b = register(client, run_b, C.MODEL_B)

    winner = parent.data.tags.get("winner_label", "A")
    champ, chall = (mv_a, mv_b) if winner == "A" else (mv_b, mv_a)

    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "champion", champ.version)
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "challenger", chall.version)

    print(f"Registered model : {C.REGISTERED_MODEL_NAME}")
    print(f"  version {mv_a.version} <- {run_a.info.run_name} ({C.MODEL_A['hf_model_id']})")
    print(f"  version {mv_b.version} <- {run_b.info.run_name} ({C.MODEL_B['hf_model_id']})")
    print(f"Decision tag     : {parent.data.tags.get('decision_reason')}")
    print(f"Alias champion   -> version {champ.version}")
    print(f"Alias challenger -> version {chall.version}")


if __name__ == "__main__":
    main()
