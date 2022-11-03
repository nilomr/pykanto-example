<br>
<br>
<div align='center'>

<a href="https://nilomr.github.io/pykanto">
    <img src="reports/figures/pykanto-logo-grey-04.svg" alt="pykanto logo" title="pykanto" height="80" style="padding-bottom:1em !important;" /> (Use example)
</a> 

<br>
<br>

![version](https://img.shields.io/badge/package_version-0.1.0-orange)
![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)
![license](https://img.shields.io/github/license/mashape/apistatus.svg)
![Open Source Love](https://img.shields.io/badge/open%20source-♡-lightgrey)
![Python 3.8](https://img.shields.io/badge/python->=3.8-blue.svg)

**pykanto** is a python library to manage and analyse bird vocalisations. This is a reproducible example showcasing one of its possible applications. 

[Installation](#installation) •
[Getting started](#getting-started) •
# 

</div>

### Installation

> Note: this is a large repository (~600 MiB) including the data necessary to train a deep learning model and reproduce the results included in the paper. It might take a couple of minutes to download.

Create a new environment, e.g. using conda:

```bash
conda create -n pykanto-example python=3.8
```
Then install pykanto:see [installing pykanto](https://nilomr.github.io/pykanto/contents/1_getting-started.html) for a complete installation guide for the library. (Or just run `pip install pykanto`!)

Finally, install this example:

```bash
pip install .
```

Note: you can also install in developer mode, and install along extra dependencies that you might find useful: `pip install -e ."[extras]"`. See `[project.optional-dependencies]` in the pyproject.toml file to see options for `extras`.



One of the steps to reproduce this example involves training a deep neural network, which requires compatible GPU resources. The repository includes the feature vectors output by the model under version control, so the step can be safely skipped. 

If you do want to train the model yourself, you'll need a few more libraries that are not installed automatically with pykanto. This is just because they are a bit finicky: which installation you need depends on which version of CUDA you have and the like).

I recommend that, if this is the case, you first create a fresh environment with conda:

```bash
conda create -n pykanto-example python=3.8    
```         
And then install torch, pykanto and this example including the extra libraries.

```bash
conda install -c pytorch pytorch torchvision   
pip install pykanto
pip install ."[torch]"
```




### Getting started

[Sample pipeline](https://nilomr.github.io/pykanto/contents/2_basic-workflow.html)

### Use guide

<details>
  <summary><b>Click to expand</b></summary>
  
#### Prepare dataset
First, make sure that you have activated this project's environment (```pykanto-example``` if you followed the instructions above). Next, let's run a simple app that calls the relevant ```pykanto``` methods to ingest, create spectrograms, and segment the dataset.

Next, navigate to the ```/notebooks``` folder.

If you want to see which options are available, run

```bash
python 2_prepare-dataset.py --help
```
This is a reproducible example, so in this case just run the following:

```bash
python 1_prepare-dataset.py -d pykanto-example -f pykanto-example
```
If you want to run this in a HPC you can use `pykanto`'s tool for this, which makes it very easy: file:///home/nilomr/projects/pykanto/docs/html/contents/hpc.html

```bash
pykanto-slaunch --exp BigBird2020 --p short --time 01:00:00 -n 1 --memory 40000 --gpu 0 -c "python 1_prepare-dataset.py -d pykanto-example -f pykanto-example" -env pykanto-example
```

The rest are computationally much lighter scripts, you can run them like so:

```
python 2_interactive-labels.py
python 3_export-training-data.py 
```
Now the dataset is ready to do whatever you please with it! In our case, we're going to 


#### Use the interactive app

Now you can start the interactive app on your browser by simply running `dataset.open_label_app()`. To learn how to use it, see [using the interactive app](./interactive-app.md). #FIXME - link to pykanto docs

Once you are done checking the automatically assigned labels you need to reload the dataset:

```python
dataset = dataset.reload()
```

### Some Code
```js
function logSomething(something) {
 console.log('Something', something);
}
```
</details>


<sub>© Nilo M. Recalde, 2021-present</sub>

