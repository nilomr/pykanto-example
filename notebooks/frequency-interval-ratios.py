import ast
from pathlib import Path

import git
import numpy as np
import pandas as pd
import seaborn as sns
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
from pykanto.signal.spectrogram import (
    get_vocalisation_units,
    retrieve_spectrogram,
)
from pykanto.utils.compute import with_pbar
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs, link_project_data, makedir

# ──── SETUP ────────────────────────────────────────────────────────────────────

DATASET_ID = "GRETI_2021"
years = ["2020", "2021"]
project_root = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)
data_path = project_root / "data" / "datasets" / DATASET_ID
segmented_dir = project_root / "data" / "segmented"
DIRS = ProjDirs(project_root, segmented_dir, DATASET_ID, mkdir=False)


out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"

dataset = load_dataset(out_dir, DIRS, relink_data=True)
dataset.files.average_units[0]


# ──── MAIN ───


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


# Get subset for each song type
nsamples = 10
dsubset = (
    dataset.data[dataset.data.class_label != "-1"]
    .groupby(["ID", "class_label"])
    .filter(lambda x: len(x) > nsamples)
    .groupby(["ID", "class_label"])
    .sample(nsamples, random_state=123)
)
dsubset["songid"] = dsubset.ID + "_" + dsubset.class_label

dflist = []
for idx in with_pbar(np.unique(dsubset.songid)[:2]):
    keys = dsubset[dsubset.songid == idx].index.values
    pfs = []
    for key in keys:
        f3notes = [v for k, v in get_vocalisation_units(dataset, key).items()][
            0
        ][:3]
        pf = [get_peak_freq(dataset, S) for S in f3notes]
        if -1 not in pf:
            dflist.append([idx, pf])


pd.DataFrame(dflist).to_csv(DIRS.DATA / "peak_freqs.csv")

# ──── EXPORT DATAFRAME ─────────────────────────────────────────────────────────

# (plotting will be done in R)
df = pd.DataFrame(dflist, columns=["ID", "frequency", "ratio"])

df = pd.merge(
    df,
    pd.DataFrame(df["iois"].values.tolist()).add_prefix("ioi_"),
    on=df.song_id,
).drop("key_0", axis=1)
df = pd.merge(
    df,
    pd.DataFrame(df["ratios"].values.tolist()).add_prefix("ratio_"),
    on=df.song_id,
).drop("key_0", axis=1)
df.to_csv(makedir(DIRS.DATA / "derived") / "peak_freqs.csv", index=False)


# Extract peak frequencies
key = str(dataset.data.index[0])
f3notes = [v for k, v in get_vocalisation_units(dataset, key).items()][0][:3]
[get_peak_freq(dataset, S) for S in f3notes]


features_df = (
    dataset.data.groupby(["ID", "type_label"])
    .filter(lambda x: len(x) > nsamples)
    .groupby(["ID", "type_label"])
    .sample(nsamples, random_state=123)
)[:2000]


minmax, mfcc, peakfreq, m_sd_cent_bw = [], [], [], []
for key in with_pbar(features_df.index):
    spec = retrieve_spectrogram(dataset.data.at[key, "spectrogram_loc"])
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


# ──── EXPORT DATAFRAME ─────────────────────────────────────────────────────────

# (plotting will be done in R)
df = pd.DataFrame(d, columns=["ID", "song_id", "iois", "ratios"])
df = pd.merge(
    df,
    pd.DataFrame(df["iois"].values.tolist()).add_prefix("ioi_"),
    on=df.song_id,
).drop("key_0", axis=1)
df = pd.merge(
    df,
    pd.DataFrame(df["ratios"].values.tolist()).add_prefix("ratio_"),
    on=df.song_id,
).drop("key_0", axis=1)
df.to_csv(makedir(DIRS.DATA / "derived") / "interval_ratios.csv", index=False)
