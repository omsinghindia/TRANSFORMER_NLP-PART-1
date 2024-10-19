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




.gitignore: Lists files to be ignored by Git version control.
Usage
Training
To train the model, run:
python train.py --config config.py
