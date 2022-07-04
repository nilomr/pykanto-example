#%%

import os
import sys
import warnings
from pathlib import Path

import git

# functions
import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.signal.analysis import (
    approximate_minmax_frequency,
    get_mean_sd_mfcc,
    get_peak_freqs,
    spec_centroid_bandwidth,
)

# Feature extraction
from pykanto.signal.filter import mels_to_hzs
from pykanto.signal.spectrogram import retrieve_spectrogram
from pykanto.utils.compute import with_pbar
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs, link_project_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import MDS
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%
# Dataset to load
DATASET_ID = "GRETI_2021"

# Create a ProjDirs object for the project, including location of raw data to
# segment
PROJECT_ROOT = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)

RAW_DATA = PROJECT_ROOT / "data" / "raw" / DATASET_ID
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, mkdir=True)


out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
out_dir = Path(
    "/media/nilomr/SONGDATA/wytham-great-tit/datasets/GRETI_2021/GRETI_2021.db"
)  # provisional
dataset = load_dataset(out_dir, DIRS)

# Get manually checked labels
recover_csv = pd.read_csv(
    dataset.DIRS.DATASET.parent / "GRETI_2021_VOCS_20220601.csv", index_col=0
)

dataset.vocs.insert(
    2, "type_label", recover_csv.type_label.astype("Int64").astype(str)
)

# Get n random songs per song type and bird
nsamples: int = 20

# functions


def get_peak_freq(
    dataset: KantoData,
    S: np.ndarray,
    melscale: bool = True,
    threshold: float = 0.3,
):

    minfreq = dataset.parameters.lowcut
    min_db = -dataset.parameters.top_dB

    if melscale:
        hz_freq = mels_to_hzs(dataset)
        result = (
            hz_freq[np.argmax(np.max(S, axis=1))]
            if (max(np.max(S, axis=1)) > min_db * (1 - threshold))
            else -1
        )

        return result

    else:
        return np.array(minfreq + np.argmax(np.max(S, axis=1)))
        # REVIEW did not test for melscale = False


# main script

features_df = (
    dataset.vocs.groupby(["ID", "type_label"])
    .filter(lambda x: len(x) > nsamples)
    .groupby(["ID", "type_label"])
    .sample(nsamples, random_state=123)
)[:2000]


minmax, mfcc, peakfreq, m_sd_cent_bw = [], [], [], []
for key in with_pbar(features_df.index):
    spec = retrieve_spectrogram(dataset.vocs.at[key, "spectrogram_loc"])
    features_df.iloc[0].ID

    minfreqs, maxfreqs = approximate_minmax_frequency(
        dataset, spec=spec, roll_percents=[0.2, 0.7]
    )

    mean_sd = np.array(
        [[np.nanmean(i), np.nanstd(i)] for i in [minfreqs, maxfreqs]]
    ).flatten()
    minmax.append(mean_sd)

    mfcc.append(get_mean_sd_mfcc(spec, 25))
    peakfreq.append(get_peak_freq(dataset, spec))

features_dict = {
    "class_label": features_df.ID + features_df.type_label,
    "mean_unit_duration": [np.mean(r[:-1]) for r in features_df.unit_durations],
    "std_unit_duration": [np.std(r[:-1]) for r in features_df.unit_durations],
    "mean_silence_duration": [
        np.mean(r[:-1]) for r in features_df.silence_durations
    ],
    "std_silence_duration": [
        np.std(r[:-1]) for r in features_df.silence_durations
    ],
    "ioi": [np.mean((r - np.append(r[1:], 0))) for r in features_df.onsets],
    "ioi_std": [np.std((r - np.append(r[1:], 0))) for r in features_df.onsets],
    "minmax_freq": minmax,
    "mfcc": mfcc,
    "peak_freq": peakfreq,
}


features = pd.DataFrame(features_dict, index=features_df.index)

arraycols = [
    col for col in features.columns if isinstance(features[col][0], np.ndarray)
]

dfs = []
for col in arraycols:
    tmpdf = features[col].apply(pd.Series)
    dfs.append(tmpdf.rename(columns=lambda x: f"{col}_{str(x)}"))

features = pd.concat(dfs + [features], axis=1).drop(columns=arraycols)

# random forest
features.dropna(inplace=True)
features.groupby("class_label").size()

y = features.class_label.values
X = features.drop(["class_label"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)


#  Scale features
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(sc.transform(X_test), columns=X.columns)

# Train and fir Random Forest
randforest = RandomForestClassifier(
    n_estimators=1000, random_state=123, max_features="sqrt", min_samples_leaf=1
)
randforest.fit(X_train, y_train)
y_pred = randforest.predict(X_test)

accuracy_score(y_test, y_pred)

#%%

classification_report(
    y_test, y_pred, target_names=np.unique(y_test), output_dict=True
)

SW832
MP521

dataset.plot(dataset.vocs.query('ID=="SW82" and type_label=="2"').index[0])
dataset.plot(dataset.vocs.query('ID=="MP52" and type_label=="1"').index[0])


def proximityMatrix(model, X, normalize=True):

    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:, 0]
    proxMat = 1 * np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:, i]
        proxMat += 1 * np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat


pmatrix = proximityMatrix(randforest, X_train, normalize=True)


mds = MDS(dissimilarity="precomputed", random_state=0)
X_transform = mds.fit_transform(pmatrix)


sns.scatterplot(
    x=X_transform[:, :1].flatten(), y=X_transform[:, 1:].flatten(), hue=y_train
)

del pmatrix

idx = 0
y_train[idx]
y_train[np.argpartition(pmatrix[idx], 20)[-20:][10]]

dataset.plot(dataset.vocs.query('ID=="B32" and type_label=="0"').index[0])
dataset.plot(dataset.vocs.query('ID=="B35" and type_label=="2"').index[0])


#%%
conf_mat = confusion_matrix(y_test, y_pred)[:30, :30]
labels = [x.title().replace("_", " ") for x in np.unique(y_test)][:30]

figsize = (10, 10)
fig, ax = plt.subplots(figsize=figsize)
sns.set(font_scale=1.4)  # for label size
sns.heatmap(
    data=conf_mat,
    annot=True,
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 9},
    square=True,
    fmt=".2f",
    cbar=False,
    cmap="BuPu",
)

ax.set_xlabel("\nPredicted", fontsize=16)
ax.set_ylabel("True label\n", fontsize=16)
plt.xticks(rotation=45, ha="right")

plt.show()


# Get and save feature importance from RF runs
feature_importances = np.array(randforest.feature_importances_)
feats = [
    (X_train.columns[i], feature_importances[i])
    for i in range(len(X_train.columns))
]
feats_df = pd.DataFrame(feats, columns=["feature", "value"]).explode(
    column="value"
)


# Quick plot of mean feature importance
# Sort the feature importance in descending order

sorted_indices = np.argsort(feature_importances)[::-1]

fig, ax = plt.subplots(figsize=(19, 5))
plt.title("Feature Importance")
plt.bar(
    range(X_train.shape[1]), feature_importances[sorted_indices], align="center"
)
plt.xticks(
    range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90
)
plt.tight_layout()
plt.show()


features.class_label.value_counts()
features_equalclass.ID.value_counts()
features_equalclass.group.value_counts()


minmax = approximate_minmax_frequency(
    dataset, spec=spec, roll_percents=[0.2, 0.7]
)
mfcc = get_mean_sd_mfcc(spec, 25)
peakfreq = get_peak_freq(dataset, spec)
centroid, bandwidth = spec_centroid_bandwidth(dataset, spec=spec, plot=True)
m_sd_cent_bw = np.array(
    [[np.nanmean(i), np.nanstd(i)] for i in [centroid, bandwidth]]
).flatten()


features_dict["mfcc"] = extract_mfccs(dataset, note_type=nt, n_mfcc=12)
features_dict["centroid_bw"] = extract_spectral_centroids_bw(
    dataset, note_type=nt
)
features_dict["minmax_freq"] = extract_minmax_frequencies(dataset, note_type=nt)


# features_dict["flatness"] = extract_flatness(
#     dataset, note_type=nt)


def get_spec(row):
    return len(retrieve_spectrogram(row.spectrogram_loc))


kk = features_df.apply(lambda row: get_spec(row), axis=1)


spec = retrieve_spectrogram(dataset.vocs.at[key, "spectrogram_loc"])
features_df.iloc[0].ID

minmax = approximate_minmax_frequency(
    dataset, spec=spec, roll_percents=[0.2, 0.7]
)
mfcc = get_mean_sd_mfcc(spec, 25)
peakfreq = get_peak_freq(dataset, spec)
centroid, bandwidth = spec_centroid_bandwidth(dataset, spec=spec, plot=True)
m_sd_cent_bw = np.array(
    [[np.nanmean(i), np.nanstd(i)] for i in [centroid, bandwidth]]
).flatten()


pd.DataFrame(features_dict)


def get_n_voc_samples(dataset, nsamples):
    ndrop = []
    for idx in dataset.vocs.groupby(["ID", "type_label"]).size().index:
        try:
            yield dataset.vocs.query(
                "ID==@idx[0] and type_label==@idx[1]"
            ).sample(nsamples, random_state=123)
        except ValueError as e:
            if str(e) == (
                "Cannot take a larger sample than "
                "population when 'replace=False'"
            ):
                ndrop.append(idx)
            else:
                raise e
    warnings.warn(
        f"Dropped {len(ndrop)} song types with sample "
        f"size < {nsamples} from {len(set([i[0] for i in ndrop]))} birds."
    )


def get_peak_freq(
    dataset: KantoData,
    S: np.ndarray,
    melscale: bool = True,
    threshold: float = 0.3,
):

    minfreq = dataset.parameters.lowcut
    min_db = -dataset.parameters.top_dB

    if melscale:
        hz_freq = mels_to_hzs(dataset)
        result = (
            hz_freq[np.argmax(np.max(S, axis=1))]
            if (max(np.max(S, axis=1)) > min_db * (1 - threshold))
            else -1
        )

        return result

    else:
        return np.array(minfreq + np.argmax(np.max(S, axis=1)))
        # REVIEW did not test for melscale = False


n = 0
for s in with_pbar(get_n_voc_samples(dataset, nsamples)):
    for key in s.index:
        spec = retrieve_spectrogram(dataset.vocs.at[key, "spectrogram_loc"])
        minmax = approximate_minmax_frequency(
            dataset, spec=spec, roll_percents=[0.2, 0.7]
        )
        mfcc = get_mean_sd_mfcc(spec, 25)
        peakfreq = get_peak_freq(dataset, spec)
        centroid, bandwidth = spec_centroid_bandwidth(
            dataset, spec=spec, plot=True
        )
        m_sd_cent_bw = np.array(
            [[np.nanmean(i), np.nanstd(i)] for i in [centroid, bandwidth]]
        ).flatten()

        n += 1
        print(n)
        if n > 7:
            raise IndexError

approximate_minmax_frequency(dataset, key=idx, plot=True)

plt.imshow(mfcc)


def extract_mfccs(
    dataset: SongDataset, note_type: str, n_mfcc: int = 20
) -> List[np.ndarray]:
    """
    Calculates the mean and SD for n MFCCs extracted from each note type in each
    song in the dataset. In the case of purr notes it return the mean mean and
    sd for all purr notes in a song.

    Args:
        dataset (SongDataset): Song dataset object.
        note_type (str): One of 'purr', 'breathe'.
        n_mfcc (int, optional): Number of MFCCs to return. Defaults to 20.

    Returns:
        List[np.ndarray]: List containing one array per song in the database
    """

    if not hasattr(dataset.DIRS, "UNITS"):
        raise KeyError(
            "This function requires the output of "
            "pykanto.SongDataset.get_units()"
        )

    idx = "-1" if note_type == "breathe" else "0:-1"
    # Iterates over IDs because data is stored gruped by ID to reduce I/O
    # bottleneck.
    keys = list(dataset.DIRS.UNITS)
    mean_sd_mfccs = []

    for key in tqdmm(keys, desc="Calculating MFCCs"):
        with open(dataset.DIRS.UNITS[key], "rb") as f:
            kk = pickle.load(f)

        notes = []
        for k, v in kk.items():
            notes.append(eval("v[" + idx + "]"))

        for note in notes:
            if note_type == "breathe":
                mean_sd = get_mean_sd_mfcc(note, n_mfcc)
            else:
                mean_sd_purrs = []
                for purr in note:
                    mean_sd_purr = get_mean_sd_mfcc(purr, n_mfcc)
                    mean_sd_purrs.append(mean_sd_purr)
                mean_sd = np.mean(mean_sd_purrs, axis=0)
            mean_sd_mfccs.append(mean_sd)

    return mean_sd_mfccs
