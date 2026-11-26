from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = ArgumentParser(description="Analyze YOLO training metrics from results.csv")
    parser.add_argument("--results", type=str, required=True, help="Path to results.csv produced by training")
    parser.add_argument("--output-dir", type=str, default="reports/lab03_metrics", help="Directory for analysis output")
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def find_column(df: pd.DataFrame, *candidates) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of the columns {candidates} found in results. Available: {list(df.columns)}")


def build_metric_summary(df: pd.DataFrame) -> dict:
    epoch_col = find_column(df, "epoch")
    precision_col = find_column(df, "metrics/precision(B)", "metrics/precision")
    recall_col = find_column(df, "metrics/recall(B)", "metrics/recall")
    map50_col = find_column(df, "metrics/mAP50(B)", "metrics/mAP50")
    map5095_col = find_column(df, "metrics/mAP50-95(B)", "metrics/mAP50-95")

    best_idx = df[map5095_col].idxmax()
    best = df.loc[best_idx]
    return {
        "total_epochs": int(df[epoch_col].max()) + 1,
        "best_epoch": int(best[epoch_col]),
        "best_precision": round(float(best[precision_col]), 4),
        "best_recall": round(float(best[recall_col]), 4),
        "best_map50": round(float(best[map50_col]), 4),
        "best_map50_95": round(float(best[map5095_col]), 4),
        "final_precision": round(float(df[precision_col].iloc[-1]), 4),
        "final_recall": round(float(df[recall_col].iloc[-1]), 4),
        "final_map50": round(float(df[map50_col].iloc[-1]), 4),
        "final_map50_95": round(float(df[map5095_col].iloc[-1]), 4),
        "columns": {
            "epoch": epoch_col, "precision": precision_col, "recall": recall_col,
            "map50": map50_col, "map50_95": map5095_col,
        },
    }


def plot_precision_recall(df, cols, output_path):
    plt.figure(figsize=(9, 5))
    plt.plot(df[cols["epoch"]], df[cols["precision"]], marker="o", label="Precision")
    plt.plot(df[cols["epoch"]], df[cols["recall"]], marker="s", label="Recall")
    plt.xlabel("Epoch"); plt.ylabel("Value"); plt.title("Precision and Recall per epoch")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(output_path, dpi=150); plt.close()


def plot_map(df, cols, output_path):
    plt.figure(figsize=(9, 5))
    plt.plot(df[cols["epoch"]], df[cols["map50"]], marker="o", label="mAP@50")
    plt.plot(df[cols["epoch"]], df[cols["map50_95"]], marker="s", label="mAP@50-95")
    plt.xlabel("Epoch"); plt.ylabel("mAP"); plt.title("Mean Average Precision per epoch")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(output_path, dpi=150); plt.close()


def main():
    args = parse_args()
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_columns(pd.read_csv(results_path))
    summary = build_metric_summary(df)
    cols = summary["columns"]

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)
    pd.DataFrame([{k: v for k, v in summary.items() if k != "columns"}]).to_csv(
        output_dir / "metrics_summary.csv", index=False)

    plot_precision_recall(df, cols, output_dir / "precision_recall_plot.png")
    plot_map(df, cols, output_dir / "map_plot.png")

    print("Metric analysis completed")
    print(f"Best epoch: {summary['best_epoch']}  mAP@50-95: {summary['best_map50_95']}  mAP@50: {summary['best_map50']}")
    print(f"Precision: {summary['best_precision']}  Recall: {summary['best_recall']}")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
