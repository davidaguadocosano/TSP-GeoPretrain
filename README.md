# :briefcase: Geometric Self-Supervised Pre-training for Neural Combinatorial Optimization

This repository contains the code for the paper "Geometric Self-Supervised Pre-training for Neural Combinatorial Optimization" by David Aguado, Daniel Fuertes, Fernando Jaureguizar, and Carlos R. del-Blanco.

## Overview

- Deep Reinforcement Learning (DRL) agents for routing problems, such as the Traveling Salesman Problem (TSP), provide near-instantaneous inference but struggle with zero-shot extrapolation. When trained from scratch, these models overfit to the specific graph scale seen during training and fail to generalize when exposed to massive, dense topologies.
- To overcome this generalization bottleneck, this paper introduces a novel geometric self-supervised pre-training framework. Based on the premise that optimal TSP sequences remain invariant under isometric transformations, we force the network to learn robust, size-agnostic spatial symmetries prior to the Reinforcement Learning policy optimization.
- By utilizing strictly distance-preserving transformations (such as rotations and axial reflections), our proposed strategy consistently outperforms models trained from scratch. It achieves a significant reduction in the optimality gap for massive zero-shot extrapolation scenarios (e.g., training on 50 nodes and testing on 1,000 nodes), delivering speedups of up to two orders of magnitude over the exact solver Concorde.

## End-to-end Neural Combinatorial Optimization Pipeline

Building upon state-of-the-art neural combinatorial optimization frameworks, we extend the standard experimental pipeline by integrating a novel geometric self-supervised learning stage. This promotes robust zero-shot generalization to massive instances.

1. **Problem Definition:** The combinatorial routing task is mathematically formulated on a 2D Euclidean graph.
2. **Geometric Pre-training (Novel Contribution):** Before any policy optimization, the network undergoes self-supervised contrastive learning. By applying strictly distance-preserving isometric transformations (such as rotations and reflections), the model learns robust, scale-agnostic spatial symmetries.
3. **Graph Embedding:** Topological features and explicit edge distances are processed using an anisotropic GNN. To ensure scalability, this encoder operates on a sparsified *k*-NN graph.
4. **Solution Decoding:** The routing probabilities are assigned to each node conditionally through graph traversal using an Autoregressive Multi-Head Attention decoder.
5. **Solution Search:** During inference, the predicted probabilities are converted into a valid discrete routing sequence through a deterministic greedy decoding strategy.
6. **Policy Learning:** The routing policy is trained end-to-end without labels to minimize the expected tour length via Deep Reinforcement Learning (using the REINFORCE gradient estimator and a Greedy Rollout Baseline).

## Installation
To ensure maximum reproducibility and avoid dependency conflicts, this project is fully containerized using Docker. The environment runs on Python 3.7 and requires a CUDA-enabled GPU.

### Prerequisites
- **Ubuntu** (tested on Ubuntu-based distributions).
- **Docker** installed on your system.
- **NVIDIA Container Toolkit** installed to enable GPU support within Docker.

### Setup Guide

**1. Clone the repository:**
```sh
git clone https://github.com/davidaguadocosano/TSP-GeoPretrain.git
cd TSP-GeoPretrain
```
**2. Build the Docker image:**
The Dockerfile will automatically install all required system dependencies and Python packages listed in your requirements.txt.
```sh
docker build -t learn_tsp .
```
(Optional: You can also build the environment using the provided docker-compose.yml by running docker-compose build).

**3. Run the container:**
We use volumes to map the current local directory to /app inside the container. This allows you to edit the code on your host machine and run it instantly without rebuilding the image. To launch an interactive bash session with full GPU access and your local user permissions, run:
```sh
docker run --user $(id -u):$(id -g) --gpus all -it --rm -v$(pwd):/app learn_tsp bash
```

## Usage

# Pretraining
To pretrain, run the script “pretrain.py”. The pretrained network will be saved with the name specified by “run_name”. If you want to continue pretraining a pretrained model, specify its path with “load_path”. The pretraining method is selected with “pretrain_type”, from the following options ('rotation', 'symmetry', 'translation', 'hybrid'). If you choose 'hybrid', use a variable called “hybrid_transformations” to indicate which transformation you want to use (rot, trans, sym). The minimum and maximum number of nodes for the TSPs can be specified with “min_size” and “max_size”. An example of a command to run hybrid pretraining would be:

```bash
python pretrain.py --encoder gnn --aggregation max --embedding_dim 128 --normalization layer --learn_norm --epoch_size 128000 --n_epochs 50 --run_name "mi_preentrenamiento" --gated --pretrain_type hybrid --hybrid_transformations rot trans sym
```

The models are saved in the “outputs” folder.


# Training
To train, the script “run.py” is executed. Again, the training session is named using the variable “run_name”. Since models can be trained from scratch or pre-trained, there is a variable called “load_path”. If we want to train a pre-trained model, we assign it the relative path of the pre-trained network weights; if we don't want to use a pre-trained model, we don't call on this variable. An example of a command to run a training session would be:

```bash
python run.py --min_size 50 --max_size 50 --encoder gnn --gated --normalization layer --learn_norm --lr_model 0.0001 --epoch_size 128000 --batch_size 256 --n_epochs 50 --load_path "outputs/tsp_20-50/mi_preentrenamiento_20260223T214729/encoder-epoch-46.pt" --run_name "TSP50_pretrained"
```

Again, the models are saved in the “outputs” folder.


# Evaluation
To perform the evaluation, the script “compare_tsp.py” is executed. The models specified by their relative paths in the variables “load_path” and “resume” are compared. “eval_size” indicates the number of TSPs to be evaluated, and “min_size” the size of the TSPs. If you want to compare a model with a solver, you don't use the “resume” variable, but instead add “--solver X”. And if you want to always evaluate on the same scenarios, you can use a specific seed, for example “--seed 1”. An example of a command to compare two models is:

```bash
python compare_tsp.py --load_path outputs/tsp_20-50/mi_preentrenamiento_20260223T214729/encoder-epoch-49.pt --resume outputs/tsp_50-50/TSP_pretrained_normbueno_nofreeze_20260301T235836/epoch-49.pt --embedding_dim 128 --gated --eval_size 10 --normalization layer --min_size 50
```

To compare a model with a solver algorithm, select the model in "load_path" and specify the comparison with the solver using `--solver x`. The following solvers are available: `concorde, lkh, ortools, aco, ga`.

-lkh: LKH-3
-ortools: Google OR-Tools (performs an initial route using NN, and then "unravels" using 3-opt). When using this algorithm, we will indicate with the `--time_limit` variable how much time we want it to spend solving each instance of the TSP.
-aco: Ant Colony Optimization
-ga: Genetic Algorithms

Several calculated values ​​will appear on the screen, and an image will be generated in the folder:
“comparativa_concorde.png” or “comparativa_modelo 2.png” as appropriate.
