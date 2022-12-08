# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path

import git
import pandas as pd
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs

# ──── SETTINGS ─────────────────────────────────────────────────────────────────

DATASET_ID = "pykanto-example"

# Start a ProjDirs object for the project
project_root = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)
segmented_dir = project_root / "data" / "segmented" / DATASET_ID
DIRS = ProjDirs(project_root, segmented_dir, DATASET_ID, mkdir=True)

# ──── LOAD DATASET ─────────────────────────────────────────────────────────────

# Open an existing dataset
out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(out_dir, DIRS)

# Check an example: (uncomment if running interactively)
# dataset.plot(str(dataset.data.index[100]), segmented=True)

# Save dataset as csv
dataset.to_csv(DIRS.DATASET.parent, timestamp=False)

# ──── CHECK OR LOAD LABELS ─────────────────────────────────────────────────────

"""
Now you can either use the interactive app or load existing labels:
using existing labels by default. App user guide in the docs.
"""

use_app = False

if use_app:
    dataset.open_label_app()
    dataset = dataset.reload()
else:
    df = pd.read_csv(DIRS.DATASET.parent / "checked_labels.csv", index_col=0)
    dataset.data.insert(2, "class_label", df.class_label.astype("string"))
    dataset.save_to_disk()
