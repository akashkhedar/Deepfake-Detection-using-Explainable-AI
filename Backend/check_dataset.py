# check_dataset.py
# Usage: python check_dataset.py --root dataset --sample_per_class 12 --out summary.json
import os
import argparse
import json
from PIL import Image, ImageOps
from tqdm import tqdm
import math


def is_image_ok(path):
    try:
        with Image.open(path) as im:
            im.verify()  # verify integrity
        # reopen to get size safely
        with Image.open(path) as im:
            w, h = im.size
        return True, (w, h)
    except Exception as e:
        return False, str(e)


def make_grid(img_paths, save_path, thumb_size=(128, 128), cols=8):
    if not img_paths:
        return
    rows = math.ceil(len(img_paths) / cols)
    grid_w = cols * thumb_size[0]
    grid_h = rows * thumb_size[1]
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    x, y = 0, 0
    for i, p in enumerate(img_paths):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail(thumb_size, Image.LANCZOS)
                # center the thumbnail in the cell
                bg = Image.new("RGB", thumb_size, (255, 255, 255))
                offset = (
                    (thumb_size[0] - im.size[0]) // 2,
                    (thumb_size[1] - im.size[1]) // 2,
                )
                bg.paste(im, offset)
                grid.paste(bg, (x * thumb_size[0], y * thumb_size[1]))
        except Exception:
            pass
        x += 1
        if x >= cols:
            x = 0
            y += 1
    grid.save(save_path)


def scan_split(split_path, sample_per_class=12, check_duplicates=False):
    summary = {"counts": {}, "corrupt": [], "sizes": {}, "samples": {}}
    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue
        files = []
        for root, _, fnames in os.walk(cls_path):
            for f in fnames:
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp", "tiff")
                ):
                    files.append(os.path.join(root, f))
        summary["counts"][cls] = len(files)
        sample_paths = []
        for p in tqdm(files, desc=f"Scanning {split_path}/{cls}", unit="img"):
            ok, info = is_image_ok(p)
            if not ok:
                summary["corrupt"].append({"path": p, "error": info})
            else:
                w, h = info
                summary["sizes"].setdefault(f"{w}x{h}", 0)
                summary["sizes"][f"{w}x{h}"] += 1
            if len(sample_paths) < sample_per_class:
                sample_paths.append(p)
        summary["samples"][cls] = sample_paths
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="dataset", help="dataset root with train/val/test"
    )
    parser.add_argument("--sample_per_class", type=int, default=12)
    parser.add_argument("--out", type=str, default="dataset_summary.json")
    parser.add_argument("--make_sample_grid", action="store_true")
    args = parser.parse_args()

    splits = ["train", "validation", "test"]
    final = {"root": args.root, "splits": {}}
    for s in splits:
        split_path = os.path.join(args.root, s)
        if not os.path.exists(split_path):
            print(f"[WARN] {split_path} does not exist. Skipping.")
            continue
        print(f"\n--- Scanning split: {s} ---")
        summary = scan_split(split_path, sample_per_class=args.sample_per_class)
        final["splits"][s] = summary
        if args.make_sample_grid:
            out_grid = os.path.join(args.root, f"sample_grid_{s}.jpg")
            # flatten sample images across classes
            flat = []
            for cls, paths in summary["samples"].items():
                flat.extend(paths)
            make_grid(flat, out_grid, thumb_size=(128, 128), cols=8)
            final["splits"][s]["sample_grid"] = out_grid
            print(f"Saved sample grid to: {out_grid}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"\nSummary written to {args.out}")
    total_corrupt = sum(len(final["splits"][s]["corrupt"]) for s in final["splits"])
    print("Total corrupt images found:", total_corrupt)
    print("Counts per split and per class printed in summary file.")


if __name__ == "__main__":
    main()
