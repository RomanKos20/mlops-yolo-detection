from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import math
import os
import platform
import subprocess
import sys

import pandas as pd
import mlflow
from ultralytics import YOLO
from ultralytics import settings


def parse_args():
    parser = ArgumentParser(description="Automated YOLO training with MLflow tracking")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base YOLO model or path to weights")
    parser.add_argument("--data", type=str, default="datasets/construction-ppe-small/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Training device: cpu or GPU index")
    parser.add_argument("--project", type=str, default="runs/lab03_script", help="Directory for experiment results")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (timestamp-based if omitted)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    return parser.parse_args()


def create_experiment_name(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{Path(model_name).stem}_construction_ppe_{timestamp}"


def save_run_metadata(run_dir: Path, args) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "run_dir": str(run_dir),
        "best_model_path": str(run_dir / "weights" / "best.pt"),
        "last_model_path": str(run_dir / "weights" / "last.pt"),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


# ---- MLflow integration (lab 04) ----

def log_yolo_metrics(run_dir: Path) -> None:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()

    prefixes = {
        "precision": "metrics/precision",
        "recall": "metrics/recall",
        "mAP50": "metrics/mAP50(",
        "mAP50_95": "metrics/mAP50-95",
    }

    columns = {}
    for metric_name, prefix in prefixes.items():
        matches = [col for col in df.columns if col.lower().startswith(prefix.lower())]
        if not matches:
            raise ValueError(f"Metric {metric_name} not found in results.csv")
        columns[metric_name] = matches[0]

    for index, row in df.iterrows():
        epoch = int(row["epoch"]) if "epoch" in df.columns else index + 1
        for name, column in columns.items():
            value = float(row[column])
            if math.isfinite(value):
                mlflow.log_metric(name, value, step=epoch)

    best_map = df[columns["mAP50_95"]].max()
    mlflow.log_metric("best_mAP50_95", float(best_map))


def log_yolo_artifacts(run_dir: Path) -> None:
    artifact_files = {
        "results.csv": "metrics",
        "results.png": "plots",
        "confusion_matrix.png": "plots",
        "confusion_matrix_normalized.png": "plots",
        "args.yaml": "metadata",
        "run_metadata.json": "metadata",
    }
    for filename, destination in artifact_files.items():
        path = run_dir / filename
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path=destination)

    for filename in ["best.pt", "last.pt"]:
        path = run_dir / "weights" / filename
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path="model")


def get_git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return "unknown"


def get_git_dirty() -> str:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.returncode != 0:
        return "unknown"
    return str(bool(result.stdout.strip())).lower()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    if args.name is None:
        args.name = create_experiment_name(args.model)

    # Вимикаємо автоматичну інтеграцію Ultralytics, логування реалізуємо самостійно
    settings.update({"mlflow": False})

    # Адреса MLflow Server із змінної середовища
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Construction-PPE-YOLO-Automated")

    with mlflow.start_run(run_name=args.name):
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "patience": args.patience,
        })

        mlflow.set_tags({
            "dataset.name": "Construction-PPE-small",
            "dataset.revision": get_git_commit(),
            "git.commit": get_git_commit(),
            "git.dirty": get_git_dirty(),
            "model.family": "YOLO11",
            "task": "object_detection",
            "environment": "python-script",
        })

        model = YOLO(args.model)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
        )

        run_dir = Path(model.trainer.save_dir)
        save_run_metadata(run_dir, args)
        log_yolo_metrics(run_dir)
        log_yolo_artifacts(run_dir)

        print("Training completed")
        print("Results:", run_dir)
        print("MLflow Run ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
