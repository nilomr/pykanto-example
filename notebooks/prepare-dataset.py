from pathlib import Path
import sys
import git
import ray


redis_password = sys.argv[1]
ray.init(address=os.environ["ip_head"], _redis_password=redis_password)
print(ray.cluster_resources())


PROJECT_ROOT = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)

"/data/zool-songbird/shil5293/data/wytham-great-tit/segmented/greti_2021"

DATA_LOCATION = Path("/media/nilomr/My Passport/SONGDATA")
link_project_data(DATA_LOCATION, PROJECT_ROOT / "data")
DATASET_ID = "GRETI_2021"

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "wytham-great-tit" / DATASET_ID
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
    # Segmentation,
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
    num_cpus=None,
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
