from pathlib import Path

import git
from pykanto.signal.segment import segment_files_parallel
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.utils.paths import (
    ProjDirs,
    get_file_paths,
    get_wavs_w_annotation,
    link_project_data,
)
from pykanto.utils.write import make_tarfile

# Dataset to segment
DATASET_ID = "GRETI_2020"

# Where are the project and its data?
PROJECT_ROOT = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)
DATA_LOCATION = Path("/media/nilomr/SONGDATA/wytham-great-tit")

# Create symlink from project to data if it doesn't exist already:
link_project_data(DATA_LOCATION, PROJECT_ROOT / "data")

# Create a ProjDirs object for the project, including location of raw data to
# segment
RAW_DATA = PROJECT_ROOT / "data" / "wytham-great-tit" / "raw" / DATASET_ID
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, DATASET_ID, mkdir=True)

# Find files and annotations and segment
wav_filepaths, xml_filepaths = [
    get_file_paths(DIRS.RAW_DATA, [ext]) for ext in [".WAV", ".xml"]
]
files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)
segment_files_parallel(
    files_to_segment,
    DIRS,
    resample=22050,
    parser_func=parse_sonic_visualiser_xml,
    min_duration=1.5,
    min_freqrange=100,
    min_amplitude=5000,
    labels_to_ignore=["NOISE", "FIRST"],
)

# Compress segmented folder annotations to upload to cluster
out_dir = DIRS.SEGMENTED.parent / f"{DIRS.SEGMENTED.name}.tar.gz"
in_dir = DIRS.SEGMENTED
make_tarfile(in_dir, out_dir)
