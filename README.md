# Skin-Lesion Classification with MLP and CNN (DermaMNIST)

PyTorch models that classify dermatoscopic images from **DermaMNIST** (MedMNIST) into 7 skin-lesion classes, e.g. melanoma, basal cell carcinoma and melanocytic nevus.

Team project (3 students) for EPFL's *Introduction to Machine Learning* course, Milestone 2 (2025).

## Models

| Model | Architecture | Input |
|---|---|---|
| **MLP** | 4 hidden layers (512 → 256 → 128 → 64), BatchNorm, ReLU, dropout 0.3 | Flattened 28×28×3 image, standardized per feature |
| **CNN** | 3 conv blocks (16 → 32 → 64 channels, two 3×3 convs each), BatchNorm, max-pooling, dropout 0.5, global average pooling, linear classifier | 3×28×28 image, standardized per channel |

Training: cross-entropy loss, Adam optimizer with weight decay 1e-4, mini-batches.
Evaluation: accuracy and macro F1-score on a held-out 20% validation split, or on the official test set with `--test`.

## Project structure

```text
main.py                     # data loading, preprocessing, training and evaluation
src/data.py                 # DermaMNIST loader (via the medmnist package)
src/utils.py                # normalization, one-hot encoding, accuracy and macro-F1
src/methods/deep_network.py # MLP, CNN and the Trainer class
src/methods/dummy_methods.py
```

## Run it

```bash
pip install -r requirements.txt

# MLP, evaluated on a validation split
python main.py --nn_type mlp

# CNN, trained on the full training set and evaluated on the test set
python main.py --nn_type cnn --test --device cuda
```

Options: `--lr` (learning rate), `--max_iters` (epochs), `--nn_batch_size`, `--device cpu|cuda|mps`.

The dataset is downloaded automatically by `medmnist` on first run.

## Tech

Python · PyTorch · NumPy · Matplotlib · MedMNIST
