import os
import shutil
import random
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)

SRC = Path("datasets/Pepper Belly Crop DS")
DST = Path("datasets")

CLASS_MAP = {
    "Pepper,_bell___Bacterial_spot": "Bacterial_Spot",
    "Pepper,_bell___healthy": "Healthy",
}

for old_name, new_name in CLASS_MAP.items():
    src_dir = SRC / old_name
    images = [f for f in os.listdir(src_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    splits = [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]

    for split_name, file_list in splits:
        dst_dir = DST / split_name / new_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in file_list:
            shutil.copy2(src_dir / fname, dst_dir / fname)

    print(f"{new_name}: {n} images "
          f"→ train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

print("\nDone. Structure:")
for split in ["train", "val", "test"]:
    for cls in CLASS_MAP.values():
        path = DST / split / cls
        count = len(list(path.iterdir())) if path.exists() else 0
        print(f"  datasets/{split}/{cls}/  →  {count} files")
