# Subsampled Bi-Transformer Surrogate Models for Flow Estimation Arround Airfoil Geometries

This repository shows an original model adapted from Transformers [[1]](#1) for solving the RANS equations, based on the LIPS framework [[2]](#2) and the Airfrans Dataset [[3]](#3).

The study provided here is part of the ML4physim challenge hosted by IRT-Systemx (see [Codabench page](https://www.codabench.org/competitions/1534/)).
CFD simulations being very costly, the use of data-driven surrogate models can be useful to optimize the shape of airfoils without paying the cost of expensive simulations.

## Subsampled Bi-Transformer (SsBT) models:
SsBT are modified version of transformer networks, where for each simulations, the query tokens are only attended to a subsampled set of key tokens from the pointcloud of the simulation which we call the skeleton of the mesh. It uses a hybrid approach on decoder-only models where both the input to be regressed and the tokens to attend to are jointly transformed.
The best model's implementation is locateed in the `subsampled_bi_transformers/bi_transformer` folder, and can be ran using the `run.py` file.

This model got us the $4^{\text{th}}$ place in this challenge!

---

## Installation

### Install the LIPS framework

#### Setup an Environment

```commandline
conda create --name ml4science python=3.9
```

##### Create a virtual environment

##### Enter virtual environment
```commandline
conda activate ml4science
```

#### Install from source
Download the LIPS repository in the `src` folder
```commandline
cd src
git clone https://github.com/IRT-SystemX/LIPS.git
```
Then remove the `numpy` and `scipy` requirement from the `setup.py` file to avoid conflicts.

```commandline
cd LIPS
pip install -U .
cd ..
```

### Install pytorch
Checkout https://pytorch.org/get-started/locally/

### Install the Airfrans library and install the datasets

#### Install the library
```sh
pip install airfrans
```

#### Download the dataset
```sh
import os
import airfrans as af

directory_name='Dataset'
if not os.path.isdir(directory_name):
    af.dataset.download(root = ".", file_name = directory_name, unzip = True, OpenFOAM = False)
```


## Running the model

Before running, make sure your repository has the following structure:
```{bash}
+---Dataset/
+---LIPS/
+---src/
    +---models/
        +---bi_transformer/
            +---my_custom_transformer.py
            +---parameters.json
    +---my_ingestion.py
    +---run.py
```
to run the model, simply run `python run.py [model_directory]`.

## The team
- Anthony Kalaydjian, Master student @ ENSTA/EPFL - anthony.kalaydjian@epfl.ch
- Anton Balykov, Master student @ EPFL - anton.balykov@epfl.ch
- Adrien Chan-Hon-Tong, Researcher in ML @ Onera Université Paris Saclay – adrien.chan_hon_tong@onera.fr


## References
<a id="1">[1]</a> 
Attention Is All You Need, A. Vaswani et al. (2017). 

<a id="3">[2]</a> 
LIPS - Learning Industrial Physical Simulation benchmark suite, M. Leyli Abadi et al. (2022).

<a id="4">[3]</a> 
AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions, F. Bonnet et al. (2023).
