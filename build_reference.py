import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import glob
import io

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/custom_cnn_best.keras"
GALLERY_DIRS = ["datasets/train", "datasets/val", "datasets/test"]
OUT_PATH = "models/leaf_reference.npz"
BATCH = 64


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    return arr[None, ...]


def load_extractor():
    model = tf.keras.models.load_model(MODEL_PATH)
    extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    return model, extractor


def image_paths(directory):
    patterns = ("*.JPG", "*.jpg", "*.jpeg", "*.png")
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
    return sorted(paths)


def _read_image(path):
    with open(path, "rb") as fh:
        return preprocess(fh.read())


def embeddings(paths, extractor):
    feats = []
    total = len(paths)
    for i in range(0, total, BATCH):
        batch = paths[i : i + BATCH]
        arrs = np.concatenate([_read_image(p) for p in batch], axis=0)
        feats.append(extractor.predict(arrs, verbose=0))
        print(f"  embedded {min(i + BATCH, total)}/{total}")
    return np.concatenate(feats, axis=0).astype(np.float32)


def normalize(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb / norms


def leave_one_out_nn(emb, chunk=64):
    nn = np.empty(emb.shape[0], dtype=np.float32)
    n = emb.shape[0]
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        d = np.linalg.norm(emb[i:j, None, :] - emb[None, :, :], axis=-1)
        d[np.arange(j - i), np.arange(i, j)] = np.inf
        nn[i:j] = d.min(axis=1)
    return nn


def nearest_neighbor_distances(query, ref, chunk=64):
    nn = np.empty(query.shape[0], dtype=np.float32)
    for i in range(0, query.shape[0], chunk):
        j = min(i + chunk, query.shape[0])
        d = np.linalg.norm(query[i:j, None, :] - ref[None, :, :], axis=-1)
        nn[i:j] = d.min(axis=1)
    return nn


def synthetic_non_leaves(seed=0):
    rng = np.random.default_rng(seed)
    flat = np.full((128, 128, 3), 128, dtype=np.float32)
    noise = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8).astype(np.float32)
    grad = np.linspace(0, 255, 128)[None, :, None] * np.ones((128, 128, 3), dtype=np.float32)
    checker = ((np.indices((128, 128)).sum(axis=0) // 16) % 2).astype(np.float32)
    checker = np.broadcast_to(checker[:, :, None], (128, 128, 3)) * 255.0
    return {"flat gray": flat, "random noise": noise, "gradient": grad, "checkerboard": checker}


def main():
    model, extractor = load_extractor()
    print("Extractor output shape:", extractor.output.shape)

    gallery_paths = []
    for directory in GALLERY_DIRS:
        paths = image_paths(directory)
        gallery_paths.extend(paths)
        print(f"  {directory}: {len(paths)} images")
    print(f"Gallery images: {len(gallery_paths)}")

    emb = normalize(embeddings(gallery_paths, extractor))
    print("Embedding matrix:", emb.shape)

    nn = leave_one_out_nn(emb)
    threshold = float(np.percentile(nn, 99))
    print(
        f"Leaf-to-leaf NN distance  min={nn.min():.4f} "
        f"median={np.median(nn):.4f} p99={threshold:.4f} max={nn.max():.4f}"
    )

    rng = np.random.default_rng(0)
    held_idx = rng.choice(len(emb), size=40, replace=False)
    keep = np.ones(len(emb), dtype=bool)
    keep[held_idx] = False
    held_nn = nearest_neighbor_distances(emb[held_idx], emb[keep])
    rejected = int((held_nn > threshold).sum())
    print(f"Held-out leaves (40, not in gallery) rejected: {rejected}/40")

    np.savez(OUT_PATH, embeddings=emb, threshold=np.float32(threshold))
    print(f"Saved -> {OUT_PATH}")

    for name, arr in synthetic_non_leaves().items():
        feats = normalize(extractor.predict(arr[None, ...], verbose=0))[0]
        d = float(np.linalg.norm(feats[None, :] - emb, axis=1).min())
        verdict = "REJECTED (not a leaf)" if d > threshold else "NOT REJECTED"
        print(f"[{name:<14}] nearest-leaf={d:.4f}  threshold={threshold:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
