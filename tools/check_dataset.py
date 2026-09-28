import argparse
from pathlib import Path
import yaml


def count(folder):
    return len([p for p in Path(folder).glob("*") if p.is_file()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset")
    args = parser.parse_args()

    root = Path(args.dataset)
    required = [
        root / "images/train",
        root / "images/val",
        root / "labels/train",
        root / "labels/val",
    ]

    print("Dataset:", root.resolve())
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing folders:")
        for p in missing:
            print(" -", p)
        return

    for folder in required:
        print(f"{folder}: {count(folder)} files")

    print("\nBasic structure looks valid.")
    print("For best results, verify every image has a matching YOLO .txt label.")


if __name__ == "__main__":
    main()
