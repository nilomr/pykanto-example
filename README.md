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
# ㅤ

</div>

### Installation

See [installing pykanto](https://nilomr.github.io/pykanto/contents/1_getting-started.html) for a complete installation guide for the library. (Or just run `pip install pykanto`!)

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
python 2_prepare-dataset.py -d pykanto-example -f pykanto-example
```

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

