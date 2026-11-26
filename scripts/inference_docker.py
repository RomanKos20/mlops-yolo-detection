from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = ArgumentParser(description="YOLO Batch Inference in Docker")
    parser.add_argument("--model", default="/app/models/best.pt", help="Path to trained YOLO model")
    parser.add_argument("--source", default="/app/input", help="Input image or directory")
    parser.add_argument("--output", default="/app/output", help="Directory for inference results")
    parser.add_argument("--imgsz", type=int, default=416, help="Target input image size")
    parser.add_argument("--device", default="cpu", help="Execution device")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    return parser.parse_args()


def validate_inputs(model_path: Path, source_path: Path) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")
    if source_path.is_dir():
        images = [p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            raise ValueError(f"No supported images found in: {source_path}")


def count_images(source_path: Path) -> int:
    if source_path.is_file():
        return 1
    return len([p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def main():
    args = parse_args()
    model_path = Path(args.model)
    source_path = Path(args.source)
    output_path = Path(args.output)

    validate_inputs(model_path, source_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("YOLO Batch Inference")
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")

    load_start = perf_counter()
    model = YOLO(str(model_path))
    model_loading_seconds = perf_counter() - load_start

    inference_start = perf_counter()
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        save=True,
        project=str(output_path),
        name="predict",
        exist_ok=True,
    )
    inference_seconds = perf_counter() - inference_start

    images_processed = len(results)
    detections = sum(len(r.boxes) for r in results)

    speed_keys = ("preprocess", "inference", "postprocess")
    speed_totals = {k: 0.0 for k in speed_keys}
    for r in results:
        for k in speed_keys:
            speed_totals[k] += float(r.speed.get(k, 0.0))
    speed_avg = {k: (speed_totals[k] / images_processed if images_processed else 0.0) for k in speed_keys}

    throughput = images_processed / inference_seconds if inference_seconds > 0 else 0.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(args.model),
        "source": str(args.source),
        "output": str(output_path / "predict"),
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "images_processed": images_processed,
        "detections": detections,
        "model_loading_seconds": round(model_loading_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        "throughput_images_per_second": round(throughput, 4),
        "average_preprocess_ms": round(speed_avg["preprocess"], 3),
        "average_model_inference_ms": round(speed_avg["inference"], 3),
        "average_postprocess_ms": round(speed_avg["postprocess"], 3),
    }

    summary_path = output_path / "inference_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print("\nInference completed")
    print(f"Images processed: {images_processed}")
    print(f"Detections: {detections}")
    print(f"Model loading: {model_loading_seconds:.3f} s")
    print(f"Inference job: {inference_seconds:.3f} s")
    print(f"Throughput: {throughput:.3f} images/s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
