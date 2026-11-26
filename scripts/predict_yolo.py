from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="Run YOLO inference on a set of images")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image file or directory with images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--project", type=str, default="runs/lab03_predict", help="Directory for prediction output")
    parser.add_argument("--name", type=str, default="predictions", help="Prediction run name")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    print(f"Loading model: {args.model}")
    model = YOLO(str(model_path))
    results = model.predict(source=args.source, conf=args.conf, save=True,
                            project=args.project, name=args.name, exist_ok=True)

    total = 0
    for result in results:
        total += len(result.boxes)
    print("Prediction completed")
    print(f"Images processed: {len(results)}  Detected objects: {total}")
    print(f"Annotated images saved to: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
