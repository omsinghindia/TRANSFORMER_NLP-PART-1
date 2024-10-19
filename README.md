# PyTorch-Transformer

## Overview

This repository contains the implementation of a Transformer model using PyTorch. The Transformer architecture is widely used for tasks such as machine translation, text generation, and more. This implementation includes training, inference, and visualization scripts to fully utilize the Transformer.

## Features

- **Model Architecture**: Implements a Transformer model from scratch using PyTorch.
- **Training**: Supports both local training (`Local_Train.ipynb`) and Google Colab-based training (`Colab_Train.ipynb`).
- **Inference**: `translate.py` and `Inference.ipynb` allow for generating predictions from a trained model.
- **Beam Search**: Implements beam search decoding (`Beam_Search.ipynb`) for improved inference.
- **Attention Visualization**: `attention_visual.ipynb` provides tools to visualize attention heads and interpret model behavior.
- **Weights and Biases Logging**: Integrated with Weights and Biases for experiment tracking (`train_wb.py`).

## Requirements

Before running the scripts, install the necessary packages using the provided `requirements.txt` or `conda.txt` files:

```bash
pip install -r requirements.txt



Alternatively, if you prefer to use Conda:
conda create --name myenv --file conda.txt
conda activate myenv
File Structure
train.py: Script for training the Transformer model.
translate.py: Performs inference and translation using a trained model.
Beam_Search.ipynb: Notebook implementing beam search for improved decoding.
attention_visual.ipynb: Visualizes attention layers for better model interpretability.
dataset.py: Handles data loading and preprocessing for training.
model.py: Contains the Transformer architecture implemented in PyTorch.
config.py: Stores configurations for training and inference.
Local_Train.ipynb: Jupyter notebook for training the model on local machines.
Colab_Train.ipynb: Jupyter notebook for training the model on Google Colab.
.gitignore: Lists files to be ignored by Git version control.
Usage
Training
To train the model, run:
python train.py --config config.py
