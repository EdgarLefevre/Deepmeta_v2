# DeepMeta

[![License](https://img.shields.io/github/license/EdgarLefevre/deepmeta?label=license)](https://github.com/EdgarLefevre/deepmeta/blob/main/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<a href="https://gitmoji.dev">
  <img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square" alt="Gitmoji">
</a>


![CBiB Logo](imgs/cbib_logo.png)

Deep learning techniques used to segment lungs and metastasis on mice MRI images.


## Contents
- [Installation](#installation)
- [Performance](#performance)
- [Usage](#usage)
- [Documentation](#Documentation)
- [Demo](#Demo)


## Installation

We recommend you to use conda to create and manage dependencies.

To install required libraries:
```shell script
conda env create -f environment.yml
```

If you do not want to use conda, with python 3.10 and pip:
```shell script
pip install -f requirements.txt
```

## Performance
To measure the performance of each network, we rely on several metrics:
 - IoU (Jaccard index)
 - AUC (AUROC).


[//]: # (## Usage)

[//]: # ()
[//]: # (You can find some example notebooks in `example` folder.)

[//]: # (In these notebooks, we teach you how to train a model, run inference and generate graphs.)

## Documentation

To generate documentation files and read them :

```shell
cd docs
make html
open _build/html/index.html
```

## Demo

To test our network rapidly on your images, we advise you to try our plugin for napari,
[napari-deepmeta](https://github.com/EdgarLefevre/napari-deepmeta). This plugin allows you to try our 2 networks on your
images without writing any line of code.