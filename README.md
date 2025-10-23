# Deep Learning Foundations

## Overview
This repository contains a 4-week structured plan to refresh and strengthen my deep learning fundamentals.  
It combines **theoretical notes**, **mathematical derivations**, and **practical implementations** in both **NumPy (from scratch)** and **PyTorch (modern frameworks)**.

## Objectives
- Re-derive and document the key mathematics behind deep learning.
- Implement core models from scratch in NumPy.
- Reproduce the same models using PyTorch.
- Cover essential architectures: MLPs, CNNs, RNNs/LSTMs.

## Structure
- `src/` → Source code (NumPy and PyTorch implementations).
- `notebooks/` → Jupyter notebooks with experiments and visualizations.
- `docs/` → Theoretical notes (Markdown + LaTeX formulas).
- `data/` → Links or small datasets for experiments.

## Weekly Plan
- **Week 1**: Backpropagation from scratch (NumPy) + theory notes.
- **Week 2**: Optimization algorithms (SGD, Adam) + PyTorch basics.
- **Week 3**: Deep networks and CNNs.
- **Week 4**: RNNs and LSTMs for sequences.

## Requirements
Python 3.10+ and the following packages:
```bash
pip install numpy matplotlib pandas torch torchvision scikit-learn

```

## Usage

Clone the repo and run any notebook:

```bash
git clone git@github.com:Limman-qaidev/deep-learning-foundations.git
cd deep-learning-foundations
jupyter notebook notebookds/week1_backprop.ipynb
```

## Results

Each week will produce:
- A theoretical markdown document (`docs/`).
- A Python implementation (`src/`).
- A Jupyter notebook with experiments (`notebooks/`).

## References

- Andrew Ng – [*Neural Networks and Deep Learning (Coursera)*](https://www.coursera.org/specializations/deep-learning)

- Michael Nielsen – [*Neural Networks and Deep Learning* (free online book) ](http://neuralnetworksanddeeplearning.com/)

- Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola – [*Dive into Deep Learning (D2L)*](https://d2l.ai/)

- Stanford CS230 – [*Deep Learning Course Notes* ](https://cs230.stanford.edu/)

- Ian Goodfellow, Yoshua Bengio, Aaron Courville – [*Deep Learning* (MIT Press)](https://www.deeplearningbook.org/)

- PyTorch Tutorials – [*Deep Learning with PyTorch: A 60 Minute Blitz*](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

- FastAI – [*Practical Deep Learning for Coders* (free course)](https://course.fast.ai/)
 