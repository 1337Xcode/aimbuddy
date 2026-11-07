"""Turns DragonData-4 (Standoff 2 screenshots labelled by people, Roboflow export) into a set laid out like the PC sets.

The screenshots were taken over ten days, a few matches a day, and Roboflow split them picture by picture, so pictures a
few seconds apart sit in its train and test splits. This puts whole days in one split instead: the test days keep only
their unaltered pictures (Roboflow's valid and test splits), since its train split holds cropped and noised copies.
Both classes (ct, tr) are players; outlines drawn as polygons become the box round them.

Usage: dragondata.py <extracted DragonData-4 folder> <out folder>
"""
import pathlib
import shutil
import sys

TEST_DAYS = ("2026-09-13", "2026-09-15", "2026-09-19")
VALID_DAYS = ("2026-09-20",)


def day_of(path):
    """The day a screenshot was taken, from its name: screenshot_<yyyy-mm-dd>_<time>_png.rf.<hash>.jpg."""
    return path.name.split("_")[1]


def box_line(line):
    """One YOLO label line as a box line; a polygon (class then x y pairs) becomes the box round it."""
    values = line.split()
    cls, points = values[0], [float(v) for v in values[1:]]
    if len(points) == 4:
        return line.strip()
    xs, ys = points[0::2], points[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return f"{cls} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


def split_of(source_split, path):
    """Our split for one picture, or None when it is left out (an altered copy from a test day)."""
    day = day_of(path)
    if day in TEST_DAYS:
        return None if source_split == "train" else "test"
    return "valid" if day in VALID_DAYS else "train"


def main():
    """Copies every kept picture with its box labels into <out>/<split>/{images,labels} and writes data.yaml."""
    source, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    counts = {}
    for source_split in ("train", "valid", "test"):
        for image in sorted((source / source_split / "images").glob("*.jpg")):
            split = split_of(source_split, image)
            if split is None:
                continue
            for folder in ("images", "labels"):
                (out / split / folder).mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, out / split / "images" / image.name)
            label = source / source_split / "labels" / f"{image.stem}.txt"
            lines = [box_line(line) for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
            (out / split / "labels" / f"{image.stem}.txt").write_text("".join(f"{line}\n" for line in lines))
            counts[split] = counts.get(split, 0) + 1
    # names one a line, as sources.read_names reads them
    (out / "data.yaml").write_text("names:\n- ct\n- tr\nnc: 2\ntrain: train/images\nval: valid/images\n"
                                   "test: test/images\n", encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
