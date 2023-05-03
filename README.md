<br>
<br>
<div align='center'>

<a href="https://nilomr.github.io/pykanto">
    <img src="reports/figures/pykanto-logo-grey-04.svg" alt="pykanto logo" title="pykanto" height="80" style="padding-bottom:1em !important;" />
</a> 

<br>
<br>

![version](https://img.shields.io/badge/package_version-0.1.0-orange)
![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)
![license](https://img.shields.io/github/license/mashape/apistatus.svg)
![Open Source Love](https://img.shields.io/badge/open%20source-♡-lightgrey)
![Python 3.8](https://img.shields.io/badge/python->=3.8-blue.svg)

This is a reproducible example demonstrating how to use [**pykanto**](https://github.com/nilomr/pykanto), a python
library to manage and analyse animal vocalisations. We use a small sample
dataset to answer a real research question: can individual birds be recognised
by their song?

[Installation](#installation) •
[User guide](#user-guide) •
[Article](https://nilomr.github.io/pykanto-example)
#

</div>

> Note: this is a large repository (~600 MiB) including the data necessary to
> train a deep learning model and reproduce the results included in the paper.
> It might take a couple of minutes to download, or longer if you have a slow connection.
### Installation

1. Create a new environment, e.g. using miniconda:
```bash
conda create -n pykanto-example python=3.8
```
2. Install pykanto:
See [installing
pykanto](https://nilomr.github.io/pykanto/_build/html/contents/installation.html) for a
complete installation guide for the library, or just run `pip install
pykanto`.

3. Clone this repository to your computer, navigate to its root and install
using pip:

```bash
git clone https://github.com/nilomr/pykanto-example.git
cd pykanto-example
pip install .
  ```
<br>


##### GPU installation

One of the steps to reproduce this example involves training a deep neural
network, which requires compatible GPU resources. The repository includes the
feature vectors output by the model, so **this step can be**
safely **skipped** if you don't want to train the NN again. 

<details>
  <summary>Expand</summary>
<br>
If you do want to train the model yourself, you will need a few more libraries
that are not installed automatically with pykanto. The reason for this is that the are a bit finicky: which exact installation you need depends on which version of
CUDA you have and the like.

I recommend that, if this is the case, you first create a fresh environment with conda:

```bash
conda create -n pykanto-example python=3.8    
```         
And then install torch, pykanto and this example including the extra libraries.

```bash
conda install -c pytorch pytorch torchvision   
pip install pykanto
# Navigate to the root of this repository, then:
pip install ."[torch]"
```
</details>


##### Developer installation

You can also install in developer mode, and install along extra dependencies
that you might find useful: `pip install -e ."[extras]"`. See
`[project.optional-dependencies]` in the pyproject.toml file to see options for
`extras`.

<br>

### User guide


First, make sure that you have activated this project's environment (`conda
activate pykanto-example` if you followed the instructions above). Then,
navigate to ```/notebooks```. This is where the scripts are located. They can
all be run from the terminal, `python <script-name>`.

<details>
  <summary>Expand user guide</summary>
<br>

| Script                      | Description                                                         | Use                                                                                                                                                                                                                                                                                                 |
| --------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1_prepare-dataset.py`      | Ingests, creates spectrograms, and segments the dataset[^1]         | To run: `python 1_prepare-dataset.py -d pykanto-example -f pykanto-example`, to see options: `python 1_prepare-dataset.py --help`                                                                                                                                                                   |
| `2_interactive-labels.py`   | Opens an interactive app to check the automatically assigned labels | The latter step requires user input so it's disabled by default for reproducibility. If you want to test the app yourself set `use_app = False` in that script. To learn how to use it, see [using the interactive app](https://nilomr.github.io/pykanto/_build/html/contents/interactive-app.html) |
| `3_export-training-data.py` | Exports the data required to train the deep learning model          | `python 3_export-training-data.py`                                                                                                                                                                                                                                                                  |
| `4_train-model.ipynb`       | Model definition and training step                                  | A separate, self-contained jupyter notebook. This is to make it easier to run interactively on a GPU-enabled HPC. If you don't want to retrain the model, you can skip this step                                                                                                                    |
| `5_measure-similarity.py`   | Measures the similarity between songs across years and birds        | `python 5_measure-similarity.py`                                                                                                                                                                                                                                                                    |
| `6_plot-results.py`         | Plots the results.                                                  | `python 6_plot-results.py`: will output to graphics device but not save.                                                                                                                                                                                                                            |
| `6.1_publication-plots.R`   | Reproduce the exact plots included in the paper                     | Switch to R and run `Rscript -e 'renv::run("6.1_publication-plots.R")'` after [installing the R dependecies](https://rstudio.github.io/renv/articles/renv.html) via `renv::restore()`                                                                                                               |

[^1]: If you want to run this in a HPC you can use `pykanto`'s tool for this, which makes it very easy: (see [Docs](https://nilomr.github.io/pykanto/_build/html/contents/hpc.html) for more info)`pykanto-slaunch --exp BigBird2020 --p short --time 01:00:00 -n 1 --memory 40000 --gpu 0 -c "python 1_prepare-dataset.py -d pykanto-example -f pykanto-example" -env pykanto-example`

</details>

<br>

### Citation
If you use `pykanto` in your own work, please cite the associated article and/or
the repository:

[![DOI](https://zenodo.org/badge/239354937.svg)](https://zenodo.org/badge/latestdoi/239354937)
[![arXiv](https://img.shields.io/badge/arXiv-2302.10340-b31b1b.svg)](https://arxiv.org/abs/2302.10340)
#
<sub>© Nilo M. Recalde, 2021-present</sub>

