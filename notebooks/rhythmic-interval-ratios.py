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

# Calculate inter onset interval and get average per songid.

data["songid"] = data.ID + "_" + data.class_label
data["ioi"] = [(np.append(r[1:], 0) - r)[:-1] for r in data.onsets]

data.loc[data.songid == "B1191", "onsets"].values

dflist = []
drop = 0
for ID in with_pbar(data.songid.unique()):
    if "-" in ID:
        continue
    v = data.loc[data.songid == ID, "ioi"].values
    if len(v) > 5:
        v = [ioi for ioi in v if len(ioi) > 3]
        if len(v) > 5:
            dflist.append([[ID, a[:3]] for a in v][:5])
        else:
            drop += 1
    else:
        drop += 1
        continue
print(f"Dropped {drop} songs due to insufficient sample size")


pdf = pd.DataFrame(
    [item for sublist in dflist for item in sublist], columns=["songid", "iois"]
)
pdf["ratios"] = [iois[:-1] / iois[1:] for iois in pdf.iois.values]


# ──── EXPORT DATAFRAME ─────────────────────────────────────────────────────────

# (plotting will be done in R)

pdf = (
    pdf.join(
        pd.DataFrame(
            pdf["iois"].to_list(),
            columns=["ioi_1", "ioi_2", "ioi_3"],
            index=pdf.index,
        )
    )
    .join(
        pd.DataFrame(
            pdf["ratios"].to_list(),
            columns=["ratio_1", "ratio_2"],
            index=pdf.index,
        )
    )
    .drop(["iois", "ratios"], axis=1)
)

pdf.to_csv(makedir(DIRS.DATA / "derived") / "interval_ratios.csv", index=False)

pdf = pd.merge(
    pdf,
    pd.DataFrame(pdf["iois"].values.tolist()).add_prefix("ioi_"),
    on=pdf.songid,
).drop("key_0", axis=1)

pdf = pd.merge(
    pdf,
    pd.DataFrame(pdf["ratios"].values.tolist()).add_prefix("ratio_"),
    on=pdf.songid,
).drop("key_0", axis=1)

df.to_csv(makedir(DIRS.DATA / "derived") / "interval_ratios.csv", index=False)


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
