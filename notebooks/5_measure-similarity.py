# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from collections import Counter
from pathlib import Path

import git
import numpy as np
import pandas as pd
import seaborn as sns
from pykanto.utils.compute import with_pbar
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs
from sklearn.metrics import pairwise_distances
from umap import UMAP

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
bird_data = pd.read_csv(DIRS.DATASET.parent / f"bird_data.csv", index_col=0)
# Open an existing pykanto dataset
out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(out_dir, DIRS)

# ──── PREPARE LABELS ───────────────────────────────────────────────────────────

# Get labels
imgpaths = [i for i in train_path.glob("*/*.jpg")]
fnames = np.unique(feat_vec.index.values)

imnames = [i.stem for i in imgpaths]

# fnames not in imnames:

not_in = [i for i in fnames if i not in imnames]
len(not_in)

labels = []
for fname in with_pbar(fnames, total=len(fnames)):
    for p in imgpaths:
        if fname == p.stem:
            labels.append(p.parent.name)

# Add labels to feature vectors
feat_vec.index, vecdf = labels, feat_vec
vecmed, labs = vecdf.to_numpy(), vecdf.index.values

# Add repertoire size column to bird data
pnums = [l.split("_")[0] for l in labs]
bird_data["repertoire_size"] = pd.Series(Counter(pnums))


# ──── DERIVE DISTANCES ─────────────────────────────────────────────────────────

metric = "cosine"

# Calculate pairwise distances
sim_mat = pairwise_distances(vecmed, metric=metric)
mx = 1 - sim_mat
mx = np.round((mx - np.min(mx)) / np.ptp(mx), 5)


# ──── WRANGLE RESULTS ──────────────────────────────────────────────────────────

# Consolidate discrete sharing matrices (by year)
birdict = bird_data.father.to_dict()
bird_id = [
    f'{birdict[idx.split("_")[0]]}_{idx.split("_")[1]}' for idx in labels
]

mat_df = pd.DataFrame(mx, index=vecdf.index, columns=vecdf.index)
tall_mat = (
    mat_df.stack()
    .reset_index()
    .rename(columns={"level_0": "class1", "level_1": "class2", 0: "dist"})
)

tall_mat["bird1"] = [f'{birdict[idx.split("_")[0]]}' for idx in tall_mat.class1]
tall_mat["bird2"] = [f'{birdict[idx.split("_")[0]]}' for idx in tall_mat.class2]
tall_mat["year1"] = tall_mat.class1.apply(lambda x: x[:4])
tall_mat["year2"] = tall_mat.class2.apply(lambda x: x[:4])
tall_mat = tall_mat.query("dist != 1")


# ──── COMPARE SIMILARITY BETWEEN AND WITHIN BIRDS ──────────────────────────────

n = 1000  # Size of subset to take (for comp. eff.)

# Most similar songs in a different year
df_most_sim = tall_mat.query("year1 != year2")
idx = (
    df_most_sim.groupby(["bird1"])["dist"].transform(max) == df_most_sim["dist"]
)
df_most_sim = df_most_sim[idx].drop_duplicates(
    subset=["dist", "bird2", "bird1"]
)

print(
    f"{len(df_most_sim.query('bird1 == bird2'))} out of {len(df_most_sim)}"
    " most similar to themselves\n"
)

# Same year and song type
df_same_class = tall_mat.query("year1 == year2 & class1 == class2").sample(
    n, random_state=42
)

# Most similar by same bird in different year
df_same = tall_mat.query(
    "year1 != year2 & bird1 == bird2"
    "& ((class1 in @df_most_sim.class1 & class2 in @df_most_sim.class2)"
    "| (class2 in @df_most_sim.class1 & class1 in @df_most_sim.class2)) "
)

# Any other bird in any year
df_diff_bird = tall_mat.query("bird1 != bird2").sample(n, random_state=42)


# Consolidate and export
all_df = pd.concat([df_same_class, df_same, df_most_sim, df_diff_bird])
all_df["type"] = (
    ["same year"] * len(df_same_class)
    + ["different year"] * len(df_same)
    + ["top hits"] * len(df_most_sim)
    + ["different bird"] * len(df_diff_bird)
)

all_dir = DIRS.DATA / "datasets" / DATASET_ID / "distances.csv"
all_df.to_csv(all_dir, index=False)


# ──── UMAP PROJECTION ──────────────────────────────────────────────────────────

# We can project the similarity matrix to a lower dimension representation
# using, e.g., UMAP:

reducer = UMAP(
    n_neighbors=100, n_components=2, min_dist=0.2, metric="precomputed"
)
embedding = reducer.fit_transform(sim_mat)

# Prepare data for later plotting:
labvalues = (
    tall_mat.groupby("class1").first().reset_index()[["class1", "bird1"]].values
)
labdict = {v[0]: v[1] for v in labvalues}
idlabs = [labdict[l] for l in labs]
nlabs = len(set(idlabs))
colordict = dict(
    zip(
        set(idlabs),
        sns.color_palette("Set3" if nlabs <= 12 else "tab20", nlabs),
    )
)
colours = [colordict[k] for k in idlabs]
markers = ["+" if "2020" in lab else "o" for lab in labs]

# Combine embedding, colours and markers variables into dataframe:
df = pd.DataFrame(embedding, columns=["x", "y"])
df["colours"] = colours
df["markers"] = markers
df["labs"] = idlabs

# Save df to file:
umap_dir = DIRS.DATA / "datasets" / DATASET_ID / "embedding.csv"
df.to_csv(umap_dir, index=False)
