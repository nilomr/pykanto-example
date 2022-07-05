# ─── DESCRIPTION ─────────────────────────────────────────────────────────────

"""
Code to read and combine relevant information available for each breeding
attempt in a nest box at which we tried to record songs.
"""

# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import warnings
from datetime import datetime as dt
from pathlib import Path

import git
import numpy as np
import pandas as pd
from pykanto.utils.compute import with_pbar
from pykanto.utils.paths import (
    ProjDirs,
    get_file_paths,
    get_wavs_w_annotation,
    link_project_data,
)

# ──── SETTINGS ─────────────────────────────────────────────────────────────────

# Dataset to segment
DATASET_ID = "GRETI_2021"

# Where are the project and its data?
PROJECT_ROOT = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)
# DATA_LOCATION = Path("/media/nilomr/SONGDATA/wytham-great-tit")
# DATA_LOCATION = Path("/media/nilomr/My Passport/SONGDATA/wytham-great-tit")
DATA_LOCATION = Path("/data/zool-songbird/shil5293/data/wytham-great-tit")


# Create symlink from project to data if it doesn't exist already:
link_project_data(DATA_LOCATION, PROJECT_ROOT / "data")

# Create a ProjDirs object for the project, including location of raw data to
# segment
RAW_DATA = PROJECT_ROOT / "data" / "segmented" / DATASET_ID.lower()
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, DATASET_ID, mkdir=True)

# ──── MAIN ─────────────────────────────────────────────────────────────────────

# Get file w/ brood data for that year
brood_path = [
    file
    for file in (DIRS.RESOURCES / "bird_data").glob("*.csv")
    if f"ebmp_broods_{DATASET_ID.split('_')[1]}" in str(file)
][0]


broods = pd.read_csv(brood_path)
broods.columns = broods.columns.str.lower().str.replace(" ", "_")

cols = [
    "pnum",
    "species",
    "april_lay_date",
    "clutch_size",
    "num_fledglings",
    "father",
    "mother",
    "clear_date",
]

broods = broods[cols]
broods["box"] = broods.pnum.apply(lambda x: x[5:])
broods["year"] = broods.pnum.apply(lambda x: x[:4])

# Format date when the box was cleared
broods["clear_date"] = broods.clear_date.apply(
    lambda x: dt.strptime(x, "%d-%m-%Y") if isinstance(x, str) else None
)

# Some boxes are not closed: assign a closed date.
broods["clear_date"] = broods.clear_date.fillna(broods["clear_date"].max())


# ──── ASSIGN RECORDIGNS TO BREEDING ATTEMPTS ───────────────────────────────────

# Every year songs recorded at a nest are saved to a `year/boxnumber` folder.
# Once all the breeding data are in, we can link the nestboxes with their
# records. To make it easier to later combine data from multiple years, we will
# rename the folders from their simple names to their 'Pnum', which includes the
# year and the breeding attempt.

if "segmented" not in str(DIRS.RAW_DATA):
    extension = ".WAV"
    olddirs = [f for f in DIRS.RAW_DATA.iterdir() if f.is_dir()]

    d = {}
    for box in olddirs:
        box.stem
        fs = list(box.glob(f"*{extension}"))
        fs.sort()
        f, l = fs[0].stem, fs[-1].stem

        d[box.stem] = {
            k: dt.strptime(datime, "%Y%m%d_%H%M%S")
            for k, datime in zip(["first", "last"], [f, l])
        }

    # Works with box code or pnum
    d = {(k if len(k) < 5 else k[5:]): v for k, v in d.items()}

    d_pnum = {}
    for box, dates in d.items():
        boxes = broods.query("box == @box").copy()
        if len(boxes) > 1:
            boxes.sort_values("clear_date", inplace=True)
            k = 0
            while boxes.at[boxes.index[k], "clear_date"] < dates["last"]:
                k += 1
            pnum = boxes.at[boxes.index[k], "pnum"]
        else:
            pnum = boxes["pnum"].values[0]
        d_pnum[box] = pnum

    if len(olddirs) != len(d_pnum):
        raise IndexError("Number of boxes does not match number of pnums")
else:
    warnings.warn("Can only rename folders in /raw directory")

# Rename raw data folders
# WARNING - proceed with caution!
if "segmented" not in str(DIRS.RAW_DATA):
    for olddir in with_pbar(olddirs, total=len(olddirs)):
        if olddir.stem in d_pnum:
            newdir = DIRS.RAW_DATA / d_pnum[olddir.stem]
            olddir.rename(newdir)
else:
    warnings.warn("Can only rename folders in /raw directory")


# Rename existing segmented WAV and JSON files
wav_filepaths, json_filepaths = [
    get_file_paths(DIRS.SEGMENTED / ext.upper(), [f".{ext}"])
    for ext in ["wav", "JSON"]
]

# Update dictionary values
for f in with_pbar(json_filepaths):
    s = f.stem.split("_")[0]
    box = s[5:] if len(s) > 5 else s
    pnum = d_pnum[box]
    with open(f, "r") as fp:
        data = json.load(fp)
    for k, v in data.items():
        if isinstance(v, str) and pnum not in v:
            data[k] = v.replace(box, pnum)
    with open(f.as_posix(), "w") as fp:
        print(json.dumps(data, indent=2), file=fp)

# Rename wav and json files
all_paths = [*wav_filepaths, *json_filepaths]
for olddir in with_pbar(all_paths):
    box = olddir.stem.split("_")[0]
    if box in d_pnum:
        pnum = d_pnum[box]
        newdir = olddir.parent / olddir.name.replace(box, pnum)
        olddir.rename(newdir)


# ──── GET RINGING DATA ─────────────────────────────────────────────────────────

ring_files = list(
    (DIRS.RESOURCES / "bird_data" / "ringing_data").glob("*.csv")
)[0]
ring_data = pd.read_csv(ring_files)
ring_data.columns = ring_data.columns.str.lower().str.replace(" ", "_")
ring_data.pit_tag = ring_data.pit_tag.apply(lambda x: x.replace("'", ""))
ring_data.drop_duplicates(inplace=True)
ring_data.replace(r"^\s*$", np.nan, regex=True, inplace=True)
ring_data.bto_species_code = ring_data.bto_species_code.str.casefold()

# ──── COMBINE DATASETS ─────────────────────────────────────────────────────────

# Only keep great tits:
ring_data = ring_data.query("bto_species_code == 'greti'")
broods = broods.query("species == 'g'")

# Add 'immigration' status
fathers = broods.father.values
fathers = list(fathers[~pd.isnull(fathers)])
wytham_born = ring_data.query("bto_ring == @fathers and age==1").bto_ring.values
broods["wytham_born"] = broods.father.apply(
    lambda x: True if x in wytham_born else np.nan if pd.isnull(x) else False
)

# If bird was born in whytham, where?
natal_box = ring_data.query("bto_ring == @fathers and age==1")[
    ["bto_ring", "location"]
]
natal_box.location = natal_box.location.apply(lambda x: x[5:])
broods = (
    broods.merge(natal_box, left_on="father", right_on="bto_ring", how="outer")
    .drop_duplicates()
    .drop(columns=["bto_ring"])
    .rename(columns={"location": "natal_box"})
)

# Add bird age codes
all_pnums = list(broods.pnum.values)
ages = ring_data.query("location==@all_pnums and bto_ring == @fathers")[
    ["bto_ring", "age"]
]
broods = (
    broods.merge(ages, left_on="father", right_on="bto_ring", how="outer")
    .drop_duplicates()
    .drop(columns=["bto_ring"])
)

# If any duplicates, take the one with minimum age:
broods.drop_duplicates(subset=["pnum"], inplace=True)

# Add box coordinates:
coord_file = DIRS.RESOURCES / "bird_data" / "nestbox_coordinates.csv"
coords = pd.read_csv(coord_file).drop("section", axis=1)
coords.box = coords.box.str.upper()
broods = broods.merge(coords).drop_duplicates(subset=["pnum"])

# ──── SAVE DERIVED DATASET ─────────────────────────────────────────────────────

# Remove unwanted columns and save
broods.drop(columns=["clear_date", "species"], inplace=True)
broods.sort_values("pnum", inplace=True)
broods.to_csv(
    DIRS.RESOURCES / "bird_data" / f"bird_data_{DATASET_ID.split('_')[1]}.csv",
    index=False,
)

# Print some information:
print(f"{len(broods)} rows in dataset")
# Print proportion of broods that have IDs / were born in whytham:
print(f"{len(broods[broods['father'].notna()])} birds have ID")
print(
    f"Out of those {len(broods.query('wytham_born == True'))} "
    "were born in the population"
)
