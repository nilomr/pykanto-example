import os
import sys
from pathlib import Path

import git
import ray
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.utils.paths import ProjDirs
from pykanto.utils.read import load_dataset

# Ray settings
redis_password = sys.argv[1]
ray.init(address=os.environ["ip_head"], _redis_password=redis_password)
print(ray.cluster_resources())

# Dataset to create
DATASET_ID = "GRETI_2021"

# Create a ProjDirs object for the project, including location of raw data to
# segment
PROJECT_ROOT = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)
RAW_DATA = Path(
    "/data/zool-songbird/shil5293/data/wytham-great-tit/segmented"
    / DATASET_ID.lower()
)
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, mkdir=True)


# Define parameters
params = Parameters(
    # Spectrogramming
    window_length=1024,
    hop_length=128,
    n_fft=1024,
    num_mel_bins=224,
    sr=22050,
    top_dB=65,  # top dB to keep
    lowcut=2000,
    highcut=10000,
    # Segmentation
    max_dB=-30,  # Max threshold for segmentation
    dB_delta=5,  # n thresholding steps, in dB
    silence_threshold=0.1,  # Between 0.1 and 0.3 tends to work
    max_unit_length=0.4,  # Maximum unit length allowed
    min_unit_length=0.02,  # Minimum unit length allowed
    min_silence_length=0.001,  # Minimum silence length allowed
    gauss_sigma=3,  # Sigma for gaussian kernel
    # general settings
    song_level=True,
    subset=None,
    verbose=False,
)
# np.random.seed(123)
# random.seed(123)
dataset = KantoData(
    DATASET_ID,
    DIRS,
    parameters=params,
    overwrite_dataset=True,
    random_subset=50,
    overwrite_data=True,
)


out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
dataset = load_dataset(out_dir)

dataset.segment_into_units()
dataset.get_units()
dataset.cluster_ids(min_sample=15)
dataset.prepare_interactive_data()

dataset.to_csv(dataset.DIRS.DATASET.parent)
