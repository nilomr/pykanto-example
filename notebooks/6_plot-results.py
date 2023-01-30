# ──── IMPORTS ──────────────────────────────────────────────────────────────────


from pathlib import Path

import git
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pykanto.utils.compute import with_pbar
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs

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
train_path = DIRS.DATASET.parent / "ML" / "train"

# ──── LOAD DATASET ─────────────────────────────────────────────────────────────


vector_dir = DIRS.DATASET.parent / "ML" / "output" / "feat_vectors.csv"
feat_vec = pd.read_csv(vector_dir, index_col=0)
fnames = np.unique(feat_vec.index.values)

# Open an existing pykanto dataset
out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(out_dir, DIRS)

all_dir = DIRS.DATA / "datasets" / DATASET_ID / "distances.csv"
all_df = pd.read_csv(all_dir, index_col=None)

umap_dir = DIRS.DATA / "datasets" / DATASET_ID / "embedding.csv"

# read colours column as tuple:
umap_df = pd.read_csv(umap_dir, index_col=None, converters={"colours": eval})


# ──── PLOT SOME MATCHING SONGS ─────────────────────────────────────────────────

# Find rows where type is 'top hits':

toplot = (
    all_df[all_df["type"] == "top hits"][["class1", "class2"]]
    .sample(10, random_state=42)
    .values
)

# Get labels
imgpaths = [i for i in train_path.glob("*/*.jpg")]
fnames = np.unique(feat_vec.index.values)
labels = []

for fname in with_pbar(fnames, total=len(fnames)):
    for p in imgpaths:
        if fname == p.stem:
            labels.append(p.parent.name)

for i, bird in enumerate(toplot):
    for idx in bird:
        key = fnames[labels.index(idx)]
        x = dataset.plot(key)
        plt.savefig(DIRS.FIGURES / f"match_{i}_{idx}.png")
        plt.close()


# ──── PLOT DISTRIBUTION OF SIMILARITY SCORES ───────────────────────────────────

sns.set_style("ticks")
fig = plt.figure()
ax = fig.add_subplot(111, aspect=3)
sns.stripplot(
    ax=ax, data=all_df, x="type", y="dist", color="k", size=2, alpha=0.5
)
sns.despine(offset=10, trim=True, bottom=True)
plt.tight_layout()
plt.show()

# ──── PLOT UMAP ────────────────────────────────────────────────────────────────

# Plot UMAP projection
plt.rcParams.update(
    {
        "ytick.color": "None",
        "xtick.color": "None",
        "axes.labelcolor": "None",
        "axes.edgecolor": "white",
        "axes.facecolor": ".18",
        "legend.fontsize": 12,
        "legend.labelspacing": 0.7,
        "legend.labelcolor": "white",
        "legend.edgecolor": "None",
    }
)

fig, ax = plt.subplots(figsize=(15, 15), facecolor=".18")

for label in set(umap_df.labs):
    mask = umap_df.labs == label
    for marker in set(umap_df[mask].markers):
        m = umap_df[mask].markers == marker
        ax.scatter(
            umap_df[mask][m].x,
            umap_df[mask][m].y,
            c=umap_df[mask][m].colours,
            marker=marker,
            label=label,
            s=200,
            alpha=0.6,
        )

ax.set_aspect("equal")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

handles, labels = ax.get_legend_handles_labels()
handles = [handles[i] for i in range(0, len(handles), 2)]
labels = [labels[i] for i in range(0, len(labels), 2)]

ax.legend(handles, labels, loc="center", bbox_to_anchor=(1.1, 0.5))
ax.set_title(
    "\nUMAP projection of 2020 (+) and 2021 (o) songs\n",
    fontsize=16,
    color="white",
)
plt.show()
