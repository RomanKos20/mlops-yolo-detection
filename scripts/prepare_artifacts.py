from argparse import ArgumentParser
from pathlib import Path
import shutil


def parse_args():
    parser = ArgumentParser(description="Collect training artifacts for DVC versioning")
    parser.add_argument("--run-dir", type=str, required=True, help="Training run directory (contains weights/, results.csv)")
    parser.add_argument("--prediction-dir", type=str, default=None, help="Directory with annotated prediction images")
    parser.add_argument("--output-dir", type=str, default="artifacts/lab03", help="Destination directory for artifacts")
    parser.add_argument("--max-predictions", type=int, default=5, help="Max number of prediction examples to copy")
    return parser.parse_args()


def copy_if_exists(src: Path, dst_dir: Path) -> bool:
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        print(f"  copied {src.name}")
        return True
    return False


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "model"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metadata_dir = output_dir / "metadata"
    predictions_dir = output_dir / "predictions"

    print("Collecting model weights")
    copy_if_exists(run_dir / "weights" / "best.pt", model_dir)
    copy_if_exists(run_dir / "weights" / "last.pt", model_dir)

    print("Collecting metrics")
    copy_if_exists(run_dir / "results.csv", metrics_dir)

    print("Collecting plots")
    copy_if_exists(run_dir / "results.png", plots_dir)
    for plot in run_dir.glob("confusion_matrix*.png"):
        copy_if_exists(plot, plots_dir)
    for plot in run_dir.glob("*_curve.png"):
        copy_if_exists(plot, plots_dir)

    print("Collecting metadata")
    copy_if_exists(run_dir / "args.yaml", metadata_dir)
    copy_if_exists(run_dir / "run_metadata.json", metadata_dir)

    if args.prediction_dir:
        pred_src = Path(args.prediction_dir)
        if pred_src.exists():
            print("Collecting prediction examples")
            predictions_dir.mkdir(parents=True, exist_ok=True)
            images = sorted([p for p in pred_src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            for image in images[: args.max_predictions]:
                shutil.copy2(image, predictions_dir / image.name)
                print(f"  copied {image.name}")

    print("\nArtifacts prepared in:", output_dir)
    for sub in [model_dir, metrics_dir, plots_dir, metadata_dir, predictions_dir]:
        if sub.exists():
            count = len(list(sub.iterdir()))
            print(f"  {sub.relative_to(output_dir)}/: {count} files")


if __name__ == "__main__":
    main()
