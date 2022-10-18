# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
from typing import ItemsView, List

import git
import numpy as np
import pandas as pd
from matplotlib import cm
from PIL import Image
from pykanto.utils.compute import with_pbar
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs
from sklearn.model_selection import train_test_split

# ──── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def save_songs(folder: Path, specs: List[Path]) -> None:
    """
    Save song spectrograms as .jpg images to folder.

    Args:
        folder (Path): Path to destination folder.
        specs (List[Path]): List of spectrogram paths.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        img = np.load(spec)
        img *= 255.0 / (img + img.min()).max()
        img = np.invert(np.flipud(np.floor(img).astype(int))) + 256
        img = Image.fromarray(np.uint8(cm.magma(img) * 255)).convert("RGB")
        img.save(folder / f"{spec.stem}.jpg")

def save_subset(
    train_dir: Path,
    test_dir: Path,
    dname: str,
    to_export: ItemsView[str, List[Path]],
) -> None:
    """
    Save train and test subsets of dataset to disk as .jpg images (in folders
    correspoding to class labels).

    Args:
        train_dir (Path): Destination folder for training data.
        test_dir (Path): Destination folder for test data.
        dname (str): Name of subset.
        to_export (ItemsView[str, List[Path]]): Subset of dataset to export.
    """
    for song_class, specs in with_pbar(to_export, total=len(to_export)):
        folder = (train_dir if dname == "train" else test_dir) / song_class
        save_songs(folder, specs)


# ──── SETTINGS ─────────────────────────────────────────────────────────────────

DATASET_ID = "pykanto-example"

# Minimum sample size to include a bird in the model:
min_sample = 10

# Start a ProjDirs object for the project
project_root = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir  # type: ignore
)
segmented_dir = project_root / "data" / "segmented" / DATASET_ID
DIRS = ProjDirs(project_root, segmented_dir, DATASET_ID, mkdir=True)

# ──── LOAD DATASET ─────────────────────────────────────────────────────────────

# Open an existing dataset
out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(out_dir, DIRS)

# ──── SUBSAMPLE DATASET FOR MODEL TRAINING ─────────────────────────────────────

"""
This will create a unique song class label for each vocalisation in the dataset
(a combination of the ID and the label).
"""

# Remove rows from song types with fewer than 10 songs, then sample 10 songs per
# type and bird

df = (
    dataset.data.query("noise == False")
    .groupby(["ID", "class_label"])
    .filter(lambda x: len(x) >= min_sample)
    .copy()
)

df_sub = pd.concat(
    [data.sample(n=min_sample) for _, data in df.groupby(["ID", "class_label"])]
)

# Remove songs labelled as noise (-1)
df_sub = df_sub.loc[df_sub["class_label"] != "-1"]

# Add new unique song type ID and add spectrogram files
df_sub["song_class"] = df_sub["ID"] + "_" + df_sub["class_label"]
df_sub["spectrogram"] = dataset.files["spectrogram"]

# Print info
n_rem = len(set(dataset.data["ID"])) - len(set(df_sub["ID"]))
print(f"Removed {n_rem} birds (no songs types with < {min_sample} examples)")


# ──── TRAIN / TEST SPLIT AND EXPORT ────────────────────────────────────────────

train, test = train_test_split(
    df_sub,
    test_size=0.3,
    shuffle=True,
    stratify=df_sub["song_class"],
    random_state=42,
)

out_dir = dataset.DIRS.DATASET.parent / "ML"
train_dir, test_dir = out_dir / "train", out_dir / "test"

for dset, dname in zip([train, test], ["train", "test"]):
    to_export = (
        dset.groupby("song_class")["spectrogram"]  # type: ignore
        .apply(list)
        .to_dict()
        .items()
    )
    save_subset(train_dir, test_dir, dname, to_export)
