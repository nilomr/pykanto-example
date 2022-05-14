
import os
from pathlib import Path
import sys
from pykanto.utils.paths import ProjDirs
from pykanto.parameters import Parameters
from pykanto.dataset import KantoData
import ray


DATASET_ID = "GREAT_TIT"
DATA_PATH = Path('/data/zool-songbird/shil5293/projects/great-tit-song/data')
PROJECT = Path('/data/zool-songbird/shil5293/projects/greti-main')
RAW_DATA = DATA_PATH / "segmented" / "GRETI_2021"
DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)
print(DIRS)

redis_password = sys.argv[1]
ray.init(address=os.environ["ip_head"], _redis_password=redis_password)
print(ray.cluster_resources())

params = Parameters(dereverb=True, verbose=True)
dataset = KantoData(
    DATASET_ID,
    DIRS,
    parameters=params,
    overwrite_dataset=True,
    overwrite_data=True,
    random_subset=100
)

print('Done')