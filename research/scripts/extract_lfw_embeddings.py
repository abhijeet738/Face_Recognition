"""T04: extract ArcFace embeddings for the LFW 10-fold pairs (PLAN sections 10.1 and 10.2).

What it does
    1. Finds (or downloads) the LFW images and the official `pairs.txt` via scikit-learn's cache.
    2. Reads the 6,000 labelled pairs straight from `pairs.txt`, so every image gets a real
       identity name. (`fetch_lfw_pairs` would load all images into RAM, about 9 GB: too
       much for an 8 GB Mac, so images are read one at a time here instead.)
    3. Runs SCRFD + ArcFace (`buffalo_l`) on every unique image and keeps the face closest to the
       image centre. If no face is found it retries with det_size=(320, 320); if that also fails it
       embeds a centre crop and records the failure.
    4. Saves research/data/lfw_embeddings.npz (git-ignored). Run this once and reuse the file.

Usage (from the repo root, with the research venv active)
    python research/scripts/extract_lfw_embeddings.py --limit 200    # quick test
    python research/scripts/extract_lfw_embeddings.py                # full run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # research/
DEFAULT_OUT = ROOT / "data" / "lfw_embeddings.npz"
N_PAIRS = 6000
FOLD_SIZE = 600
CROP_FRACTION = 0.6  # fallback centre crop covers 60% of the shorter image side

Key = tuple[str, int]  # (person name, image number), as written in pairs.txt


def locate_lfw() -> tuple[Path, Path]:
    """Return (images_dir, pairs_file), downloading through scikit-learn if needed."""
    from sklearn.datasets import get_data_home

    home = Path(get_data_home()) / "lfw_home"
    images, pairs = home / "lfw", home / "pairs.txt"
    if not (images.is_dir() and pairs.is_file()):
        print("LFW not found locally, downloading (about 200 MB)...")
        try:
            # Private scikit-learn helper (Verify: it exists in your scikit-learn version).
            from sklearn.datasets._lfw import _check_fetch_lfw

            _check_fetch_lfw(funneled=False, download_if_missing=True)
        except Exception as exc:  # noqa: BLE001 - any failure gets the same manual-download advice
            sys.exit(
                f"LFW download failed ({exc}).\n"
                "Download the LFW archive (lfw.tgz) and pairs.txt from a public mirror such as "
                "Kaggle, extract so that these exist, then rerun:\n"
                f"  {images}/<Person_Name>/*.jpg\n  {pairs}"
            )
    if not (images.is_dir() and pairs.is_file()):
        sys.exit(f"Expected {images} and {pairs} after download, but they are missing.")
    return images, pairs


def parse_pairs(path: Path) -> list[tuple[Key, Key, bool]]:
    """Parse pairs.txt: 3 fields = same person, 4 fields = different people. Header is skipped."""
    pairs: list[tuple[Key, Key, bool]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            name, i, j = parts
            pairs.append(((name, int(i)), (name, int(j)), True))
        elif len(parts) == 4:
            n1, i, n2, j = parts
            pairs.append(((n1, int(i)), (n2, int(j)), False))
    return pairs


def build_index(pairs: list[tuple[Key, Key, bool]], selected: np.ndarray) -> dict:
    """Map the selected pairs to unique images, identities and folds."""
    key_to_id: dict[Key, int] = {}
    pair_idx = np.zeros((len(selected), 2), dtype=np.int32)
    for row, p in enumerate(selected):
        for col in (0, 1):
            key = pairs[p][col]
            pair_idx[row, col] = key_to_id.setdefault(key, len(key_to_id))
    keys = list(key_to_id)
    names = sorted({k[0] for k in keys})
    name_to_id = {n: i for i, n in enumerate(names)}
    return {
        "keys": keys,
        "pair_idx": pair_idx,
        "issame": np.array([pairs[p][2] for p in selected], dtype=bool),
        "folds": (selected // FOLD_SIZE).astype(np.int16),
        "identity": np.array([name_to_id[k[0]] for k in keys], dtype=np.int32),
        "names": np.array([k[0] for k in keys]),
    }


def load_models():
    """SCRFD + ArcFace (buffalo_l), CPU only. Downloads the models on first use."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def centre_face(faces, shape):
    """Keep the face whose box centre is closest to the image centre (LFW images are centred)."""
    cy, cx = shape[0] / 2, shape[1] / 2
    return min(
        faces,
        key=lambda f: (
            ((f.bbox[0] + f.bbox[2]) / 2 - cx) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2 - cy) ** 2
        ),
    )


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def embed_image(app, img: np.ndarray) -> tuple[np.ndarray, str]:
    """Return (unit embedding, status) where status is 'ok', 'retry' or 'failed'."""
    import cv2

    faces = app.get(img)
    status = "ok"
    if not faces:
        # Retry with a smaller detector input (Verify: attribute name in your insightface version).
        det = app.det_model
        old = det.input_size
        det.input_size = (320, 320)
        try:
            faces = app.get(img)
        finally:
            det.input_size = old
        status = "retry"
    if faces:
        return unit(centre_face(faces, img.shape).normed_embedding), status

    h, w = img.shape[:2]
    s = int(CROP_FRACTION * min(h, w))
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = cv2.resize(img[y0 : y0 + s, x0 : x0 + s], (112, 112))
    feat = app.models["recognition"].get_feat(crop)  # Verify: returns shape (1, 512)
    return unit(feat), "failed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--limit", type=int, default=0, help="use only N evenly spaced pairs (quick test)"
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output .npz path")
    args = ap.parse_args()

    import cv2
    from tqdm import tqdm

    images_dir, pairs_file = locate_lfw()
    pairs = parse_pairs(pairs_file)
    if len(pairs) != N_PAIRS:
        sys.exit(f"Expected {N_PAIRS} pairs in {pairs_file}, found {len(pairs)}.")
    print(f"{len(pairs)} pairs, {sum(p[2] for p in pairs)} same-person")

    selected = np.arange(N_PAIRS)
    if args.limit:
        selected = np.linspace(0, N_PAIRS - 1, args.limit).astype(int)
    index = build_index(pairs, selected)
    keys = index["keys"]
    print(f"{len(keys)} unique images to process")

    app = load_models()
    emb = np.zeros((len(keys), 512), dtype=np.float32)
    det_failed = np.zeros(len(keys), dtype=bool)
    retried = 0
    start = time.time()
    for n, (name, num) in enumerate(tqdm(keys, unit="img")):
        path = images_dir / name / f"{name}_{num:04d}.jpg"
        img = cv2.imread(str(path))  # uint8 BGR, which is what insightface expects
        if img is None:
            sys.exit(f"Cannot read {path}")
        emb[n], status = embed_image(app, img)
        det_failed[n] = status == "failed"
        retried += status == "retry"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        emb=emb,
        pair_idx=index["pair_idx"],
        issame=index["issame"],
        folds=index["folds"],
        identity=index["identity"],
        det_failed=det_failed,
        names=index["names"],
    )
    print(f"\nSaved {args.out} in {(time.time() - start) / 60:.1f} min")
    norms = np.linalg.norm(emb, axis=1)
    print(f"emb {emb.shape}, pairs {index['pair_idx'].shape}")
    print(f"identities: {index['identity'].max() + 1}")
    print(f"detection retried with 320: {retried}")
    print(f"DETECTION FAILURES (centre crop used): {det_failed.sum()}")
    print(f"embedding norms: min {norms.min():.3f}, max {norms.max():.3f}")


if __name__ == "__main__":
    main()