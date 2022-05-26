# Interventional Explanations of Neural Networks

Explainability and interpretability play an important role for adopting deep neural networks Through analyzing the effect of path interventions at various nodes on model's performance, we are able to reveal the causal mechanisms within hidden layers and isolate the relevant components from noisy ones.

This repository contains the material used to obtain the results in our [paper](https://openreview.net/pdf?id=1GU5D--W7C) with LeNet trained on the MNIST dataset.
## Prerequisites

Install python 3:

```bash
sudo apt update
sudo apt install python3
sudo apt install python3-pip
```

Install [poetry](https://python-poetry.org/) : 
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

## Installation
1. Clone the repo and go at the root
   ```bash
    git clone https://github.com/nbereux/Interventional-Explanations-of-Neural-Networks.git && cd Interventional-Explanations-of-Neural-Networks
   ```
2. Setup poetry
   ```bash
    poetry install
    # The next part is to setup jupyter with poetry
    poetry run jupyter contrib nbextension install --user
    poetry run jupyter nbextensions_configurator enable --user
    poetry run ipython kernel install --user --name=explainnn
   ```

## Launch example
To launch the main script simply run 
```bash
poetry run python src/explainnn/main.py
```
A step by step demonstration is available in the [jupyter-notebook](demonstration.ipynb) 
## Config file

`config.yaml` contains all the parameters for the main script :

 - `device` : wether to run the script on CPU or GPU (⚠️ : if you want to use the GPU you have to install PyTorch alongside CUDA, see : https://pytorch.org/get-started/locally/)
 - `dataset_name` : the name of the dataset to use
 - `model_name` : the name of the model to use
 - `learn_explainer` : wether to generate the causal graph or not
 - `target_idx` : the index of the targeted class
 - `n_samples` : the number of samples used to generate the causal graph
 - `soft_interventions` : soft or hard interventions
 - `graph_stab` : wether to test graph stability
 - `gen_attr` : wether to generate attributions
 - `save_attr` : wether to save the generated attributions
 - `vis_attr` : wether to plot the attributions
 - `eval_attr` : wether to generate the metrics file
 - `baseline_attr` : wether to plot the baseline methods attributions
 - `layer_name` : the name of the layer used to generate attributions
 - `layer_name_soft` : the name of the layer used to test graph stability


For evaluation metrics and comparison with traditional attributions methods we used the [quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) library

For other methods that don't exist in quantus library refer to [this file](src/explainnn/baseline_att.py)

## TO-DO

Add implementations on other architectures (ResNet18, ResNet50, ConvNext, ...) and datasets (MiniImageNet)

## LICENSE 
Licensed under Apache 2.0 License.

License will be released upon paper review completion