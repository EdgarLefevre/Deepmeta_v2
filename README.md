# DeepMeta

[![License](https://img.shields.io/github/license/EdgarLefevre/deepmeta?label=license)](https://github.com/EdgarLefevre/deepmeta/blob/main/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<a href="https://gitmoji.dev">
  <img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square" alt="Gitmoji">
</a>


[![CBiB Logo](imgs/cbib_logo.png)](https://www.cbib.u-bordeaux.fr/)

Deep learning techniques used to segment lungs and metastasis on mice MRI images.


## Contents
- [Installation](#installation)
- [Performance](#performance)
- [Usage](#usage)
- [Documentation](#Documentation)
- [Demo](#Demo)


## Installation

We recommend you to use conda to create your virtual env and manage dependencies.

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


## Usage

### Training

```shell
python -m src.train
```

### Prediction

```shell
python -m src.predict
```

### CLI arguments

```
  --epochs EPOCHS, -e EPOCHS
                        number of epochs of training
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        size of the batches
  --size SIZE, -s SIZE  Size of the image, one number
  --drop DROP, -d DROP  Dropout rate
  --filters FILTERS, -f FILTERS
                        Number of filters in first conv block
  -w1 W1                Weight for bg
  -w2 W2                Weight for lungs
  -w3 W3                Weight for metastases
  -lr LR                Learning rate.
  -nworkers NWORKERS    Number of workers in dataloader.
  -classes CLASSES      Number of classes to predict.
  --save                If flag, save predictions.
  --postprocess         If flag, apply postprocess on predictions.
  --restart RESTART     Restart for cosine annealing. Default 5. Set this to 1 to disable.
  --restart_mult RESTART_MULT
                        A factor increases Ti after a restart. Default 1.
  --model {unet,unet_res,unet3p,res_unet3p,urcnn,att_unet,att_unet3p,unet++}
                        Model name. unet or unet_res
  --model_path MODEL_PATH
                        Weights file name.
  --metric {iou,f1}     Metric for stats. iou or f1
  --loss {ce,focal,tanimoto,lovasz,fusion}
                        Loss function.
  --conv {conv,convsep}
                        Conv layer to use (conv or convsep).
  --contrast            If flag, enhance contrast on image.
  --img_path IMG_PATH   Path to the tiff mouse stack.
  --label_path LABEL_PATH
                        Path to the labels folder, if exist. If not, no stats will be processed.
```

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