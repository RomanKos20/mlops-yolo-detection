from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import platform
import sys

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="Automated YOLO training for Construction-PPE-small dataset")
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


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML file not found: {data_path}")
    if args.name is None:
        args.name = create_experiment_name(args.model)

    print("Starting YOLO training")
    print(f"Model: {args.model}  Dataset: {args.data}  Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}  Batch: {args.batch}  Device: {args.device}  Name: {args.name}")

    model = YOLO(args.model)
    model.train(data=str(data_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=args.project, name=args.name, patience=args.patience)

    run_dir = Path(model.trainer.save_dir)
    save_run_metadata(run_dir, args)
    print("\nTraining completed")
    print(f"Results saved to: {run_dir}")
    print(f"Best model: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
