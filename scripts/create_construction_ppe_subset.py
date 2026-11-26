"""Створення зменшеної версії датасету Construction-PPE (construction-ppe-small).

Бере частину зображень із кожної підмножини (train/val/test) повного датасету,
копіює відповідні YOLO label-файли та формує окремий data.yaml.
Приклад:
    python scripts/create_construction_ppe_subset.py --train 100 --val 30 --test 30 --overwrite
"""
import argparse
import shutil
from pathlib import Path

SRC = Path("datasets/construction-ppe")
DST = Path("datasets/construction-ppe-small")
IMG_EXT = (".jpg", ".jpeg", ".png")

NAMES = {
    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles", 5: "Person",
    6: "no_helmet", 7: "no_goggle", 8: "no_gloves", 9: "no_boots", 10: "forklift", 11: "no_vest",
}


def copy_split(split: str, count: int) -> int:
    src_img = SRC / "images" / split
    src_lbl = SRC / "labels" / split
    dst_img = DST / "images" / split
    dst_lbl = DST / "labels" / split
    if not src_img.exists():
        print(f"  {split}: підмножина відсутня у датасеті — пропущено")
        return 0
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXT)
    selected = images[:count]
    copied = 0
    for img in selected:
        label = src_lbl / f"{img.stem}.txt"
        shutil.copy2(img, dst_img / img.name)
        if label.exists():
            shutil.copy2(label, dst_lbl / label.name)
        copied += 1
    print(f"  {split}: скопійовано {copied} зображень (із {len(images)})")
    return copied


def write_yaml():
    lines = [
        "path: datasets/construction-ppe-small",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for i, name in NAMES.items():
        lines.append(f"  {i}: {name}")
    (DST / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  створено {DST/'data.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=100)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"Повний датасет не знайдено: {SRC}")
    if DST.exists():
        if args.overwrite:
            shutil.rmtree(DST)
        else:
            raise SystemExit(f"{DST} вже існує. Використайте --overwrite.")

    print("Створення construction-ppe-small:")
    copy_split("train", args.train)
    copy_split("val", args.val)
    copy_split("test", args.test)
    write_yaml()
    print("Готово.")


if __name__ == "__main__":
    main()
