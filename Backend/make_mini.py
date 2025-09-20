# make_mini.py
import os, random, shutil, argparse


def make_mini(
    src_root, dst_root, per_class=2000, splits=("train", "validation", "test")
):
    os.makedirs(dst_root, exist_ok=True)
    for s in splits:
        src = os.path.join(src_root, s)
        dst = os.path.join(dst_root, s)
        if not os.path.exists(src):
            print("Skipping missing split", s)
            continue
        for cls in os.listdir(src):
            src_cls = os.path.join(src, cls)
            if not os.path.isdir(src_cls):
                continue
            dst_cls = os.path.join(dst, cls)
            os.makedirs(dst_cls, exist_ok=True)
            files = [
                os.path.join(root, f)
                for root, _, fnames in os.walk(src_cls)
                for f in fnames
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            random.shuffle(files)
            chosen = files[: min(len(files), per_class)]
            for p in chosen:
                shutil.copy2(p, os.path.join(dst_cls, os.path.basename(p)))
            print(f"Copied {len(chosen)} for {s}/{cls}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="dataset")
    parser.add_argument("--dst", default="dataset_mini")
    parser.add_argument("--per_class", type=int, default=2000)
    args = parser.parse_args()
    make_mini(args.src, args.dst, args.per_class)
