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

# Create symlink from project to data if it doesn't exist already:
DATA_LOCATION = Path("/media/nilomr/My Passport/SONGDATA/wytham-great-tit")
link_project_data(DATA_LOCATION, project_root / "data")

DIRS = ProjDirs(project_root, segmented_dir, DATASET_ID, mkdir=False)

# Load existing dataset
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
nsamples = 5
dsubset = (
    dataset.data[dataset.data.class_label != "-1"]
    .groupby(["ID", "class_label"])
    .filter(lambda x: len(x) > nsamples)
    .groupby(["ID", "class_label"])
    .sample(nsamples, random_state=123)
)
dsubset["songid"] = dsubset.ID + "_" + dsubset.class_label

dflist = []
for idx in with_pbar(np.unique(dsubset.songid)):
    keys = dsubset[dsubset.songid == idx].index.values
    pfs = []
    for key in keys:
        f3notes = [v for k, v in get_vocalisation_units(dataset, key).items()][
            0
        ][:3]
        pf = [get_peak_freq(dataset, S) for S in f3notes]
        if -1 not in pf:
            dflist.append([idx, pf])


# ──── EXPORT DATAFRAME ─────────────────────────────────────────────────────────

# (plotting will be done in R)

fdf = pd.DataFrame(dflist, columns=["songid", "freqs"])

fdf = (
    fdf.join(
        pd.DataFrame(
            fdf["freqs"].to_list(),
            columns=["freq_1", "freq_2", "freq_3"],
            index=fdf.index,
        )
    ).drop(["freqs"], axis=1)
).dropna(subset=["freq_1", "freq_2", "freq_3"])

fdf.to_csv(makedir(DIRS.DATA / "derived") / "peak_freqs.csv", index=False)
