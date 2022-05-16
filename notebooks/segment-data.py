from pathlib import Path
import git
from pykanto.utils.paths import link_project_data
from pykanto.utils.paths import ProjDirs
from pykanto.utils.paths import get_file_paths, get_wavs_w_annotation
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.signal.segment import segment_files_parallel

PROJECT_ROOT = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)

DATA_LOCATION = Path("/media/nilomr/My Passport/SONGDATA")
link_project_data(DATA_LOCATION, PROJECT_ROOT / "data")
DATASET_ID = "GRETI_2021"

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "wytham-great-tit" / DATASET_ID
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, mkdir=True)


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


from pykanto.utils.write import make_tarfile

out_dir = DIRS.SEGMENTED / "JSON.tar.gz"
in_dir = DIRS.SEGMENTED / "JSON"
make_tarfile(in_dir, out_dir)
