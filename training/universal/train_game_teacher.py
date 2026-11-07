"""Trains a game's own teacher for label_frames.py --game-teacher: the COCO YOLO26s fine-tuned on a set of that game's
screenshots labelled by people, every class one class, player.

The set is laid out like the PC sets (<set>/train and <set>/valid, each with images and labels, as dragondata.py writes
it). Its pictures were stretched to a square by their publisher, so they are put back to 16:9 first, into
<work>/set. The teacher trains at 768 px, the size label_frames.py runs it at: at 960 px with a batch of 4 it ran out of
the memory of a 4 GB graphics card in its second epoch.

Usage: train_game_teacher.py <PC-style set> <work folder> [--base yolo26s.pt] [--epochs 25]
The weights end up in <work>/run/weights/best.pt.
"""
import argparse
import pathlib

import cv2
from ultralytics import YOLO

TEACHER_SIZE = 768   # label_frames.GAME_TEACHER_SIZE


def write_one_class_set(source, out):
    """Copies the train and valid splits at 16:9 with every class as class 0, and returns the data.yaml path."""
    for split, ours in (("train", "train"), ("valid", "val")):
        for folder in ("images", "labels"):
            (out / folder / ours).mkdir(parents=True, exist_ok=True)
        for image_path in sorted((source / split / "images").glob("*.jpg")):
            image = cv2.imread(str(image_path))
            height = image.shape[0]
            image = cv2.resize(image, (int(round(height * 16 / 9)), height), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out / "images" / ours / image_path.name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label = source / split / "labels" / f"{image_path.stem}.txt"
            lines = [line.split() for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
            # coordinates are shares of the picture, so they hold at any width
            (out / "labels" / ours / f"{image_path.stem}.txt").write_text(
                "".join("0 " + " ".join(parts[1:5]) + "\n" for parts in lines))
    data = out / "data.yaml"
    data.write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: player\n",
                    encoding="utf-8")
    return data


def main():
    """Prepares the set and trains one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("set", help="the game's screenshots, laid out like the PC sets")
    parser.add_argument("work", help="folder for the 16:9 copy of the set and the run")
    parser.add_argument("--base", default="yolo26s.pt", help="the checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    work = pathlib.Path(args.work).resolve()
    data = write_one_class_set(pathlib.Path(args.set), work / "set")
    YOLO(args.base).train(
        data=str(data), imgsz=TEACHER_SIZE, epochs=args.epochs, batch=4, workers=3, device=0, seed=20261106,
        optimizer="AdamW", lr0=0.001, cos_lr=True, patience=10, amp=True, fliplr=0.5, mosaic=1.0, close_mosaic=10,
        hsv_s=0.5, hsv_v=0.4, project=str(work), name="run", exist_ok=False, plots=True)


# Ultralytics starts its data loaders as new processes, which on Windows import this file again
if __name__ == "__main__":
    main()
