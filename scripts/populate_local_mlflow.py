"""Частина 1 ЛР4: локальне логування результатів навчання YOLO у MLflow (SQLite).

Повторює логіку notebook lab04_mlflow.ipynb як автоматизований скрипт:
створює експеримент Construction-PPE-YOLO-Notebook і реєструє два Run-и
(з різними гіперпараметрами), логуючи параметри, теги, метрики за епохами та артефакти.
"""
import math
import subprocess
from pathlib import Path

import pandas as pd
import mlflow

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

MLFLOW_DIR = ROOT / "data" / "mlflow-local"
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = MLFLOW_DIR / "mlflow.db"
ARTIFACT_PATH = MLFLOW_DIR / "artifacts"
ARTIFACT_PATH.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")

EXPERIMENT = "Construction-PPE-YOLO-Notebook"
if mlflow.get_experiment_by_name(EXPERIMENT) is None:
    mlflow.create_experiment(name=EXPERIMENT, artifact_location=ARTIFACT_PATH.as_uri())
mlflow.set_experiment(EXPERIMENT)
print("Tracking URI:", mlflow.get_tracking_uri())


def git_info(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


GIT_COMMIT = git_info("rev-parse", "HEAD")
GIT_BRANCH = git_info("branch", "--show-current")
GIT_DIRTY = bool(git_info("status", "--porcelain"))

METRIC_PREFIXES = {
    "precision": "metrics/precision",
    "recall": "metrics/recall",
    "mAP50": "metrics/mAP50(",
    "mAP50_95": "metrics/mAP50-95",
}


def log_run(run_name, run_dir, params):
    run_dir = Path(run_dir)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.set_tags({
            "dataset.name": "Construction-PPE-small",
            "dataset.revision": GIT_COMMIT,
            "git.commit": GIT_COMMIT,
            "git.branch": GIT_BRANCH,
            "git.dirty": str(GIT_DIRTY).lower(),
            "model.family": "YOLO11",
            "task": "object_detection",
            "environment": "jupyter",
        })
        run_id = run.info.run_id

        df = pd.read_csv(run_dir / "results.csv")
        df.columns = df.columns.str.strip()
        columns = {}
        for name, prefix in METRIC_PREFIXES.items():
            matches = [c for c in df.columns if c.lower().startswith(prefix.lower())]
            if not matches:
                raise ValueError(f"Metric column not found: {name}")
            columns[name] = matches[0]

        for index, row in df.iterrows():
            epoch = int(row["epoch"]) if "epoch" in df.columns else index + 1
            for name, column in columns.items():
                value = float(row[column])
                if math.isfinite(value):
                    mlflow.log_metric(name, value, step=epoch)
        mlflow.log_metric("best_mAP50_95", float(df[columns["mAP50_95"]].max()))

        files = {
            "results.csv": "metrics",
            "results.png": "plots",
            "confusion_matrix.png": "plots",
            "confusion_matrix_normalized.png": "plots",
            "args.yaml": "metadata",
        }
        for filename, destination in files.items():
            path = run_dir / filename
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=destination)
        best_model = run_dir / "weights" / "best.pt"
        if best_model.is_file():
            mlflow.log_artifact(str(best_model), artifact_path="model")

        print(f"Logged run '{run_name}'  id={run_id}  best_mAP50_95={df[columns['mAP50_95']].max():.4f}")
        return run_id


if __name__ == "__main__":
    import sys
    run1_dir = sys.argv[1]
    run2_dir = sys.argv[2]
    log_run("yolo11n-notebook-01", run1_dir,
            {"model": "yolo11n.pt", "epochs": 10, "imgsz": 320, "batch": 4, "device": "cpu"})
    log_run("yolo11n-notebook-02", run2_dir,
            {"model": "yolo11n.pt", "epochs": 10, "imgsz": 416, "batch": 2, "device": "cpu"})
    print("Done. Experiment:", EXPERIMENT)
