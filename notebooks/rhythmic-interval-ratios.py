import ast
from pathlib import Path

import git
import numpy as np
import pandas as pd
import seaborn as sns
from pykanto.utils.compute import with_pbar
from pykanto.utils.paths import ProjDirs, makedir

# ──── SETUP ────────────────────────────────────────────────────────────────────

DATASET_ID = "GRETI_2021"
years = ["2020", "2021"]
project_root = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)
data_path = project_root / "data" / "datasets" / DATASET_ID
segmented_dir = project_root / "data" / "segmented"
DIRS = ProjDirs(project_root, segmented_dir, DATASET_ID, mkdir=False)


# ──── BUILD DATASET FROM A CSV FILE ────────────────────────────────────────────

dl = f"{DATASET_ID}_VOCS.csv"

# F defs
def from_np_array(array_string):
    array_string = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(array_string))


def str_to_path(path):
    return Path(path)


data = pd.read_csv(
    data_path / dl,
    index_col=0,
    dtype={"auto_class": object, "class_label": object},
    converters={
        "type_label": str,
        "unit_durations": from_np_array,
        "onsets": from_np_array,
        "offsets": from_np_array,
        "silence_durations": eval,
    },
).rename(columns={"type_label": "class_label"})

# ──── MAIN ─────────────────────────────────────────────────────────────────────

# Get subset for each song type
nsamples = 10
dsubset = (
    data[data.class_label != "-1"]
    .groupby(["ID", "class_label"])
    .filter(lambda x: len(x) > nsamples)
    .groupby(["ID", "class_label"])
    .sample(nsamples, random_state=123)
)

# Calculate inter onset intervals and ratios.
dsubset["songid"] = dsubset.ID + "_" + dsubset.class_label
dsubset["iois"] = [(np.append(r[1:], 0) - r)[:-1] for r in dsubset.onsets]
dsubset["ratios"] = [iois[:-1] / iois[1:] for iois in dsubset.iois.values]
dsubset["iois"] = [l[:3] for l in dsubset.iois]
dsubset["ratios"] = [l[:3] for l in dsubset.ratios]
dsubset = dsubset[["songid", "ratios", "iois"]]


# ──── EXPORT DATAFRAME ─────────────────────────────────────────────────────────

# (plotting will be done in R)

pdf = (
    dsubset.join(
        pd.DataFrame(
            dsubset["iois"].to_list(),
            columns=["ioi_1", "ioi_2", "ioi_3"],
            index=dsubset.index,
        )
    )
    .join(
        pd.DataFrame(
            dsubset["ratios"].to_list(),
            columns=["ratio_1", "ratio_2", "ratio_3"],
            index=dsubset.index,
        )
    )
    .drop(["iois", "ratios"], axis=1)
).dropna(subset=["ioi_1", "ioi_2", "ioi_3"])

pdf.to_csv(makedir(DIRS.DATA / "derived") / "interval_ratios.csv", index=False)


# iois = np.mean(np.stack([a[:4] for a in v]), axis=0)
# ratios = iois[:-1] / iois[1:]
# d.append(
#     [
#         data.loc[data.songid == ID, "ID"].values[0],
#         ID,
#         iois[:-1],
#         ratios,
#     ]
# )
