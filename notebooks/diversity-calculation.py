# %%
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import git
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from pkg_resources import load_entry_point
from pykanto.utils.compute import with_pbar
from pykanto.utils.paths import ProjDirs, link_project_data
from pynndescent import NNDescent
from scipy.spatial import distance_matrix

# %%
# MDS
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from tqdm import tqdm

# %%
# Umap of song type median positions
# from cuml import UMAP as cumlUMAP
from umap import UMAP

# %% [markdown]
# # Project setup

# %%
PROJECT_ROOT = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)
DATASET_ID = "GRETI_2021-22"
years = ["2020", "2021"]

data_path = PROJECT_ROOT / "data" / "datasets" / DATASET_ID / "ML"
train_path, test_path = data_path / "train", data_path / "test"
vector_dir = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / DATASET_ID
    / "ML"
    / "output"
    / "feat_vectors.csv"
)

# %%
# DATA_LOCATION = Path("/data/zool-songbird/shil5293/data/wytham-great-tit")
# link_project_data(DATA_LOCATION, PROJECT_ROOT / "data")

segmented_dir = PROJECT_ROOT / "data"  # / "segmented" / DATASET_ID.lower()[:-3]
DIRS = ProjDirs(PROJECT_ROOT, segmented_dir, DATASET_ID, mkdir=True)

# %% [markdown]
# # Data ingest

# %% [markdown]
# ## Breeding data

# %%
# Read in bird breeding data
fs = [DIRS.RESOURCES / "bird_data" / f"bird_data_{year}.csv" for year in years]
bird_data = pd.concat([pd.read_csv(f, index_col=0) for f in fs])
bird_data["year"] = bird_data["year"].apply(str)

# Prepare spatial information
coord_df = bird_data[["x", "y"]].copy()
coord_df["xy"] = coord_df[["x", "y"]].values.tolist()
coord_df = pd.DataFrame(coord_df["xy"], index=bird_data.index, columns=["xy"])

# Get box infomation
fb = DIRS.RESOURCES / "bird_data" / "nestbox_coordinates.csv"
box_data = pd.read_csv(fb)
box_data["box"] = box_data["box"].str.upper()

# %% [markdown]
# ## Feature vectors

# %%
# Read in feature vectors
feat_vec = pd.read_csv(vector_dir, index_col=0)

# Remove broken bird (20211O115; something physically wrong with the poor thing)

# Looking at field notes and sharing, decided to remove the following:
# 20211W56 : same bird as 20211W51, both no ID, failed - possibly a second attempt
# 20211W36 : same bird as 20211W37, second no ID
# 20211O75B : same bird as 20211O75D, no ID, failed - possibly a second attempt
# 20211W79 : same bird as 20211W80, boxes very close so possibly same bird.
#            W80 more amplitude so assigned to that

remove = ["20211O115", "20211W56", "20211W36", "20211O75B", "20211W79"]
feat_vec = feat_vec[~feat_vec.index.str.contains("|".join(remove))]

# Get labels
with open(vector_dir.parent / "train_labels.pkl", "rb") as f:
    labels = pickle.load(f)
    labels = [l for l in labels if l.split("_")[0] not in remove]

# Add labels and calculate class means
feat_vec.index = labels
vecdf = feat_vec.groupby(feat_vec.index).mean()
vecmed = vecdf.to_numpy()

# Get all class labels
labs = vecdf.index.values

# Add repertoire size column to bird data
pnums = [l.split("_")[0] for l in labs]
bird_data["repertoire_size"] = pd.Series(Counter(pnums))

# How many birds are in >1yr?
allrings = bird_data.query(
    "father == father and repertoire_size == repertoire_size"
).father.tolist()
print(f"{len(allrings) - len(set(allrings)) } were recorded in both years")

# %%
# Build discrete song sharing matrix (by year)

# Functions
def acoustic_knn(
    feat_df: pd.DataFrame, metric: str = "cosine", ac_knn: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns indexes and acoustic distances of knns in acoustic space

    Args:
        feat_df (pd.DataFrame): One row per bird, one column per feature
        metric (str, optional): Distance to use. Defaults to "cosine".
        ac_knn (int, optional): knns not including self. Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple with indexes and distances
    """
    sim_mat = pairwise_distances(feat_df.to_numpy(), metric=metric)
    mx = 1 - sim_mat
    mx = np.round((mx - np.min(mx)) / np.ptp(mx), 5)
    nnindex = NNDescent(mx, metric=metric)
    indx, dists = nnindex.query(mx, k=ac_knn + 1)
    return indx[:, 1:], dists[:, 1:]


def get_shared_songs(
    bird_data: pd.DataFrame,
    feat_df: pd.DataFrame,
    k: int,
    year: str,
    metric: str = "cosine",
):
    # Get coordinates of birds for which we have enough data
    coordata = bird_data.query(
        f"year == '{year}' and repertoire_size == repertoire_size"
    )[["x", "y"]]

    labs = feat_df.index.values

    # calculate pairwise distances
    indx, dists = acoustic_knn(feat_df, metric, k)

    # Build binary matrix (yes/no song among knn)
    wl = len(feat_df)
    m = np.zeros(shape=(wl, wl))
    for query in with_pbar(feat_df.index):
        i = np.where(labs == query)[0][0]
        _, ne = dists[i], labs[indx[i]]
        m[i, np.in1d(feat_df.index, ne).nonzero()[0]] = 1

    labs_dict = {
        pnum: [l for l in labs if pnum == l.split("_")[0]]
        for pnum in coordata.index
    }

    nn_sharing = []
    for pnum1 in coordata.index:
        rep1 = labs_dict[pnum1]
        idx1 = [np.where(labs == i)[0][0] for i in rep1]
        for pnum2 in coordata.index:
            rep2 = labs_dict[pnum2]
            idx2 = [np.where(labs == i)[0][0] for i in rep2]
            shared = m[np.ix_(idx1, idx2)]
            # print(pd.DataFrame(shared, index=rep1, columns=rep2))
            nn_sharing.append(
                [pnum1, pnum2, 2 * int(np.sum(shared)), len(rep1) + len(rep2)]
            )  # to calculate Song Sharing Index (McGregor and Krebs 1982)

    # Return as dataframe
    nn_sharing_df = pd.DataFrame(
        nn_sharing, columns=["bird1", "bird2", "shared", f"total_rep"]
    )

    return nn_sharing_df


def drop_swap_duplicates(df, colnames: List[str]):
    df[colnames] = np.sort(df[colnames].values, axis=1)
    df = df.drop_duplicates(colnames)
    return df


def imm_disp_rate(bird_data, coords, box_distmat):
    """
    calculate proportion of knn that are immigrants and
    mean dispersal distance for birds that were born in the population

    Args:
        bird_data (_type_): _description_
        coords (_type_): _description_
        box_distmat (_type_): _description_

    Returns:
        _type_: _description_
    """
    imm, disp = [], []
    for pnum in with_pbar(coords.index):
        residents = []
        dispdist = []
        for pnum1 in coords.loc[pnum, "spatial_nn"]:
            ims = bird_data.loc[pnum1, "wytham_born"]
            residents.append(ims)
            if ims:
                nat = bird_data.loc[pnum1, "natal_box"]
                dispdist.append(box_distmat.loc[nat, pnum1[5:]])
        imm.append(1 - np.mean(residents))
        disp.append(np.round(np.mean(dispdist), decimals=1))

    dfd = {"prop_immigrants": imm, "mean_dispersal": disp}
    imm_disp_df = pd.DataFrame(dfd, index=coords.index)
    return imm_disp_df


# #%%
# # Only with k-spatial-n
# knn_sharing = []

# for year in years:
#     # Add labels and calculate class means
#     metric = "cosine"
#     feat_df = vecdf[vecdf.index.str.contains(year)]
#     labs = feat_df.index.values

#     # calculate pairwise distances
#     indx, dists = acoustic_knn(feat_df, metric, ac_knn)

#     wl = len(feat_df)
#     m = np.zeros(shape=(wl, wl))
#     for query in with_pbar(feat_df.index):
#         i = np.where(labs == query)[0][0]
#         ds, ne = dists[i], labs[indx[i]]
#         m[i, np.in1d(feat_df.index, ne).nonzero()[0]] = 1

#     # pd.DataFrame(m, index=feat_df.index, columns=feat_df.index)

#     # get nearest spatial neighbours
#     coordata = bird_data.query(
#         f"year == '{year}' and repertoire_size == repertoire_size"
#     )[["x", "y"]]

#     # Calculate sharing with nearest spatial neighbours
#     labs_dict = {
#         pnum: [l for l in labs if pnum == l.split("_")[0]]
#         for pnum in coordata.index
#     }

#     nn_sharing = []
#     for pnum1 in tqdm(coordata.index):
#         rep1 = labs_dict[pnum1]
#         idx1 = [np.where(labs == i)[0][0] for i in rep1]
#         for pnum2 in coordata.loc[pnum1, "spatial_nn"]:
#             rep2 = labs_dict[pnum2]
#             idx2 = [np.where(labs == i)[0][0] for i in rep2]
#             shared = m[np.ix_(idx1, idx2)]
#             # if np.sum(shared) > 1:
#             #     print(pd.DataFrame(shared, index=rep1, columns=rep2))
#             nn_sharing.append(
#                 [pnum1, pnum2, 2 * int(np.sum(shared)), len(rep1) + len(rep2)]
#             )  # to calculate Song Sharing Index (McGregor and Krebs 1982)

#     nn_sharing_df = pd.DataFrame(
#         nn_sharing, columns=["bird1", "bird2", "shared", f"total_nn"]
#     )
#     nn_sharing_sum_df = nn_sharing_df.groupby(["bird1"]).sum()
#     nn_sharing_sum_df.index.rename("pnum", inplace=True)
#     nn_sharing_sum_df = nn_sharing_sum_df.reset_index(level=0)
#     knn_sharing.append(nn_sharing_sum_df)


# knn_sharing_df = pd.concat(knn_sharing)
# bird_data_sharing = bird_data.merge(
#     knn_sharing_df, left_on="pnum", right_on="pnum", how="outer"
# )

# 20211B108 is a good show bird

#%%


#%%

# ──── CALCULATE PAIRWISE SONG SHARING ──────────────────────────────────────────

# 'song sharing' here means that a song is among
# the k-nearest neighbours of another bird's song

# Settings
ac_knn = 15  # How many neighbors to use for acoustic similarity
sp_knn = 10  # How many spatial neighbors to use
metric = "cosine"

nn_sharing = [
    get_shared_songs(bird_data, vecdf, ac_knn, year, metric=metric)
    for year in years
]
nn_sharing_df = pd.concat(nn_sharing).query("bird1!=bird2")
# nn_sharing_df = drop_swap_duplicates(nn_sharing_df, ["bird1", "bird2"])

# %%
# ──── BUILD PAIRWISE DISTANCE DATAFRAME FOR ALL CLASSES ───────────────────────

# calculate pairwise distances
mx = 1 - pairwise_distances(vecmed, metric=metric)
mx = np.round((mx - np.min(mx)) / np.ptp(mx), 5)
mat_df = pd.DataFrame(mx, index=vecdf.index, columns=vecdf.index)

tall_mat = (
    mat_df.stack()
    .reset_index()
    .rename(columns={"level_0": "class1", "level_1": "class2", 0: "ac_sim"})
)
tall_mat = tall_mat.query("ac_sim != 1")
tall_mat["bird1"] = tall_mat.class1.apply(lambda x: x.split("_")[0])
tall_mat["bird2"] = tall_mat.class2.apply(lambda x: x.split("_")[0])
tall_mat["year1"] = tall_mat.class1.apply(lambda x: x[:4])
tall_mat["year2"] = tall_mat.class2.apply(lambda x: x[:4])

# Add spatial distances to similarity dataframe
selfsim_mean = tall_mat.query("bird1==bird2").groupby(["bird1", "year1"]).mean()
selfsim_mean["cat"] = "self"
selfsim_mean.reset_index(level=1, drop=True, inplace=True)
selfsim_mean.index = selfsim_mean.index.set_names(["pnum"])

# Add self-similarity to data
bird_data["self_similarity"] = selfsim_mean["ac_sim"]

# Get all pairwise mean similarities
pairwise = []
for year in years:
    pair_df = (
        tall_mat.query(f"bird2!=bird1 and year1=='{year}' and year2=='{year}'")
        .groupby(["bird1", "bird2"], as_index=False)
        .mean()
    )
    pair_df = pair_df.merge(
        coord_df[["xy"]].add_suffix("_1"), right_index=True, left_on="bird1"
    ).merge(
        coord_df[["xy"]].add_suffix("_2"), right_index=True, left_on="bird2"
    )
    pair_df["spatial_dist"] = np.linalg.norm(
        np.array([*pair_df["xy_1"].values])
        - np.array([*pair_df["xy_2"].values]),
        axis=1,
    ).round(decimals=2)
    pairwise.append(pair_df)


pairwise_df = pd.concat(pairwise).query("bird1!=bird2")
# pairwise_df = drop_swap_duplicates(pairwise_df, ["bird1", "bird2"])

#%%

# ──── JOIN SIMILARITY AND REPERTOIRE-SHARING DFS ───────────────────────────────

pair_sharing_df = nn_sharing_df.merge(pairwise_df, on=["bird1", "bird2"])
pair_sharing_df["sharing_index"] = (
    pair_sharing_df["shared"] / pair_sharing_df["total_rep"]
)

# Relationship between similarity and repertoire-sharing
sns.lmplot(
    data=pair_sharing_df.query("sharing_index >0"),
    x="ac_sim",
    y="sharing_index",
    scatter_kws={"alpha": 0.05},
)

# %%


# Add sharing with all other birds to bird data
bird_data = (
    bird_data.reset_index()
    .merge(
        pair_sharing_df.groupby(["bird1"]).mean()[["ac_sim"]],
        left_on="pnum",
        right_on="bird1",
        how="outer",
    )
    .set_index("pnum")
)

# get nearest spatial neighbours for each bird and how many songs they share
def kn_spatial_n(bird_data: pd.DataFrame, year: str, sp_knn: int = 5):
    coordata = bird_data.query(
        f"year=='{year}' and repertoire_size == repertoire_size"
    )[["x", "y"]]
    tree = BallTree(coordata.values, metric="euclidean")
    _, indices = tree.query(coordata.values, k=sp_knn + 1)
    coordata["spatial_nn"] = [coordata.index[i].values[1:] for i in indices]
    return coordata


coordata = pd.concat([kn_spatial_n(bird_data, year, sp_knn) for year in years])

nns = []
cols = [
    "bird1",
    "bird2",
    "shared",
    "total_rep",
    "ac_sim",
    "spatial_dist",
    "sharing_index",
]
# TODO: get proportion of neighbours that are immigrant for each bird
for pnum in with_pbar(coordata.index):
    for pnum1 in coordata.loc[pnum, "spatial_nn"]:
        nns.append(
            pair_sharing_df.query("bird1 == @pnum and bird2 == @pnum1")[
                cols
            ].values.tolist()[0]
        )
nn_sharing_df = pd.DataFrame(
    nns,
    columns=cols,
)

nn_sharing_summary_df = nn_sharing_df.groupby("bird1").sum()
nn_sharing_summary_df["mean_spat_dist"] = nn_sharing_df.groupby("bird1").mean()[
    "spatial_dist"
]
nn_sharing_summary_df["mean_ac_sim"] = nn_sharing_df.groupby("bird1").mean()[
    "ac_sim"
]

bird_data = (
    bird_data.reset_index()
    .merge(
        nn_sharing_summary_df[
            ["shared", "total_rep", "mean_spat_dist", "mean_ac_sim"]
        ],
        left_on="pnum",
        right_on="bird1",
        how="outer",
    )
    .set_index("pnum")
)

#%%

# Build spatial distance matrix for boxes
box_distmat = pd.DataFrame(
    distance_matrix(box_data[["x", "y"]].values, box_data[["x", "y"]].values),
    index=box_data.box,
    columns=box_data.box,
).round(decimals=1)

# TODO: get mean dispersal distance / prop immigrants for neighbourhood
sp_knn = 10  # twice the acoustic neighbors to avoid tiny sample

immdisp = []
for year in years:
    coords = bird_data.query(f"year=='{year}' and father==father")[["x", "y"]]
    tree = BallTree(coords.values, metric="euclidean")
    _, indices = tree.query(coords.values, k=sp_knn + 1)
    coords["spatial_nn"] = [coords.index[i].values[1:] for i in indices]

    # calculate neighbourhood immigration and dispersal levels
    immdisp.append(imm_disp_rate(bird_data, coords, box_distmat))

bird_data_complete = bird_data.join(pd.concat(immdisp))


#%%

# ──── SAVE DATA ────────────────────────────────────────────────────────────────


bird_data_complete.to_csv(
    DIRS.RESOURCES / "bird_data" / f"full_dataset_{'-'.join(years)}.csv"
)


#%%
pnum = "20201B11"
pair_sharing_df.loc[pnum]


pair_sharing_df.groupby(["bird1"])

nn_sharing_df = pd.DataFrame(
    nn_sharing, columns=["bird1", "bird2", "shared", f"total_nn"]
)
nn_sharing_sum_df = nn_sharing_df.groupby(["bird1"]).sum()
nn_sharing_sum_df.index.rename("pnum", inplace=True)
nn_sharing_sum_df = nn_sharing_sum_df.reset_index(level=0)
knn_sharing.append(nn_sharing_sum_df)


# TODO: get mean dispersal distance / prop immigrants for neighbourhood
for pnum in tqdm(coordata.index):
    pair_sharing_df.loc[coordata.loc[pnum, "spatial_nn"]]


# %%


#%%


sns.catplot(
    data=bird_data,
    x="year",
    y="repertoire_size",
    hue="wytham_born",
    kind="boxen",
)

sns.catplot(data=bird_data, x="wytham_born", y="repertoire_size", kind="violin")

kk = bird_data_complete.copy()
kk["sharing_index"] = kk["shared"] / kk["total_rep"]

sns.lmplot(
    data=kk,
    y="repertoire_size",
    x="dispersal_index",
    scatter_kws={"alpha": 0.05},
)
plt.yscale("log")

mean_spat_dist

sns.relplot(
    x="x",
    y="y",
    hue="mean_spat_dist",
    size="mean_spat_dist",
    sizes=(40, 400),
    alpha=0.5,
    height=6,
    data=bird_data_complete,
)


sns.lmplot(
    data=nn_sharing_summary_df,
    x="mean_spat_dist",
    y="mean_ac_sim",
    scatter_kws={"alpha": 0.05},
    lowess=True,
)


# %% [markdown]
# # Plot query results

# %%
# Plot closest song types


def plot_knn(folder: List[Path], query, neighbours, distances):
    lab_folders = {p.stem: p for p in list(folder.glob("*"))}
    images = [list(lab_folders[query].glob("*.jpg"))[0]] + [
        list(lab_folders[l].glob("*.jpg"))[0] for l in neighbours
    ]
    fig = plt.figure(figsize=(30, 30))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, len(images)),
        axes_pad=0.1,
    )

    for i, (ax, path) in enumerate(zip(grid, images)):
        # Iterating over the grid returns the Axes.
        name = path.parent.name
        if i == 0:
            ax.set_title(f"Target: {name}", fontsize=22)
        else:
            ax.set_title(f"{name} ({distances[i-1]:.2f})", fontsize=22)
        ax.imshow(Image.open(path))
        ax.axis("off")
    plt.show()


for query in labs[0:3]:
    # query = '20211W64A_2'
    loc = np.where(labs == query)[0][0]
    distances, neighbours = dists[loc], labs[indx[loc]]
    plot_knn(train_path, query, neighbours, distances)

# %% [markdown]
# # UMAP projection


reducer = UMAP(
    n_neighbors=10, n_components=2, min_dist=0.3, metric="precomputed"
)
embedding = reducer.fit_transform(sim_mat)
allrings = bird_data.query(
    "father == father and repertoire_size == repertoire_size"
).father.tolist()
print(f"{len(allrings) - len(set(allrings)) } were recorded in both years")

# %%
plt.figure(figsize=(10, 10))

colours = ["blue" if "2020" in lab else "red" for lab in labs]
plt.scatter(embedding[:, 0], embedding[:, 1], c=colours, s=10, alpha=0.5)
plt.gca().set_aspect("equal", "datalim")


mds = MDS(
    metric=True,
    dissimilarity="precomputed",
    random_state=0,
    n_jobs=-1,
    verbose=1,
)
# Get the embeddings
pts = mds.fit_transform(sim_mat)


for query in labs[0:3]:
    # query = '20211W64A_2'
    # plot_knn(train_path, query, neighbours, distances)
    loc = np.where(labs == query)[0][0]
    distances, neighbours = dists[loc], labs[indx[loc]]
    plt.figure(figsize=(10, 10))
    colours = ["blue" if lab in neighbours else "grey" for lab in labs]
    plt.scatter(pts[:, 0], pts[:, 1], c=colours, s=20, alpha=0.5)
    plt.gca().set_aspect("equal", "datalim")
    plt.show()

# %%
# Consolidate information for each bird

# %%
# %%
# get kn(spatial)n


k = 5
year = 2020
coordata = bird_data.query(
    "year==@year and repertoire_size == repertoire_size"
)[["x", "y"]]

tree = BallTree(coordata.values, metric="euclidean")
distances, indices = tree.query(coordata.values, k=k + 1)
coordata["spatial_nn"] = [coordata.index[i].values[1:] for i in indices]

labs_dict = {pnum: [l for l in labs if pnum in l] for pnum in coordata.index}


# calculate acoustic neigbours for this year only

# Add labels and calculate class means
metric = "cosine"
feat_vec.index = labels
feat_vec_year = feat_vec[feat_vec.index.str.contains(str(year))]
vecdf = feat_vec_year.groupby(feat_vec_year.index).mean()

vecmed = vecdf.to_numpy()
sim_mat = pairwise_distances(vecmed, metric=metric)
mx = 1 - sim_mat
mx = np.round((mx - np.min(mx)) / np.ptp(mx), 5)

# Get k nearest acoustic neighbours for each bird

# nn queries
indx, dists = nnindex.query(mx, k=4)
indx = indx[:, 1:]
dists = dists[:, 1:]

kk = []
for pnum in with_pbar(coordata.index):
    l = labs_dict[pnum]
    nl = len(l)

    for query in l:
        loc = np.where(labs == query)[0][0]
        distances, neighbours = dists[loc], labs[indx[loc]]

    kk.append([pnum, neighbours])


for pnum in with_pbar(coordata.index):
    coordata.loc[pnum, "spatial_nn"]
    labs_dict[pnum]
    for query in labs_dict[pnum]:
        loc = np.where(labs == query)[0][0]
        distances, neighbours = dists[loc], labs[indx[loc]]


for name, d, ind in zip(coordata.index, distances, indices):
    print(f"NAME {name} closest matches:")
    for i, index in enumerate(ind):
        print(f"\t{bird_data.index[index]} with distance {d[i]} m")


bird_data.query("pnum == '20211O115'")

# %%
# Similarity with natal vs territory neighbours

# Similarity with own year vs previous year males

# Mean similarity with neighbours for resident and immigrants

# mean similarity between own songs vs random set of equal size from rest of the population

# delete members of own class (also works on full matrix)
knnidx = []
knndists = []
for i, d, l in zip(idx, dists, labels):
    rmidx = np.where(np.array(labels) == l)[0]
    idxs = np.isin(i, rmidx)
    knnidx.append(np.delete(i, idxs, axis=0))
    knndists.append(np.delete(d, idxs, axis=0))

class_medians = np.median(dists, axis=1)
mindist = np.argmin(class_medians)
maxdist = np.argmax(class_medians)

ulabs = list(dict.fromkeys(labels))
print(f"{ulabs[mindist] = }, {ulabs[maxdist] = }")

groups = (
    pd.Series(range(len(median_dst)))
    .groupby(labels, sort=False)
    .apply(list)
    .tolist()
)
class_mx = [np.mean([median_dst[i] for i in idx]) for idx in groups]

diag = pd.Series(
    np.diag(class_dist_mat),
    index=[class_dist_mat.index, class_dist_mat.columns],
)


l = "B227_1"
rmidx = np.where(np.array(labels) == l)[0]
index.query(mx[rmidx], k=10)


# NOTE: array will be irregular if not all own labels
# are among the knn

median_dst = [np.median(i) for i in knndists]
groups = (
    pd.Series(range(len(median_dst)))
    .groupby(labels, sort=False)
    .apply(list)
    .tolist()
)
class_medians = [np.median([median_dst[i] for i in idx]) for idx in groups]

mindist = np.argmin(class_medians)
maxdist = np.argmax(class_medians)

ulabs = list(dict.fromkeys(labels))
print(f"{ulabs[mindist] = }, {ulabs[maxdist] = }")


class_medians[mindist]
class_medians[maxdist]

sim_mat_class = pd.DataFrame(mx, columns=labels, index=labels)

sim_mat.index.values[mindist]
sim_mat.index.values[maxdist]


labels[mindist]
labels[maxdist]

median_dst[mindist]
median_dst[maxdist]

# %%


# %%


# %%
