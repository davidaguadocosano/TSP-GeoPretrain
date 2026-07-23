# :briefcase: Learning TSP Requires Rethinking Generalization

This repository contains code for the paper [**"Learning TSP Requires Rethinking Generalization"**](https://arxiv.org/abs/2006.07054) by Chaitanya K. Joshi, Quentin Cappart, Louis-Martin Rousseau, Thomas Laurent, accepted to the **27th International Conference on Principles and Practice of Constraint Programming** (CP 2021).

## Overview

- End-to-end training of neural network solvers for combinatorial problems such as the **Travelling Salesman Problem** is intractable and inefficient beyond a few hundreds of nodes. 
While state-of-the-art Machine Learning approaches perform closely to classical solvers for trivially small sizes, they are **unable to generalize** the learnt policy to larger instances of practical scales.
- Towards leveraging transfer learning to **solve large-scale TSPs**, this paper identifies inductive biases, model architectures and learning algorithms that promote generalization to instances larger than those seen in training. 
Our controlled experiments provide the first principled investigation into such **zero-shot generalization**, revealing that extrapolating beyond training data requires rethinking the entire neural combinatorial optimization pipeline, from network layers and learning paradigms to evaluation protocols.

## End-to-end Neural Combinatorial Optimization Pipeline

Towards a controlled study of **neural combinatorial optimization**, we unify several state-of-the-art architectures and learning paradigms into one experimental pipeline and provide the first principled investigation on zero-shot generalization to large instances.

![End-to-end neural combinatorial optimization pipeline](/img/pipeline.png)

1. **Problem Definition:** The combinatorial problem is formulated via a graph.
2. **Graph Embedding:** Embeddings for each graph node areobtained using a Graph Neural Network encoder.
3. **Solution Decoding:** Probabilities are assigned to each node for belonging to the solution set, either independent of one-another (i.e. Non-autoregressive decoding) or conditionally through graph traversal (i.e. Autoregressive decoding).
4. **Solution Search:** The predicted probabilities are converted intodiscrete decisions through classical graph search techniques such as greedy search or beam search.
5. **Policy Learning:** The entire model in trained end-to-end via imitating anoptimal solver (i.e. supervised learning) or through minimizing a cost function (i.e. reinforcement learning).

**We open-source our framework and datasets to encourage the community to go beyond evaluating performance on fixed TSP sizes, develop more expressive and scale-invariant GNNs, as well as study transfer learning for combinatorial problems.**

## Installation
We ran our code on Ubuntu 16.04, using Python 3.6.7, PyTorch 1.2.0 and CUDA 10.0. 
We highly recommend installation via Anaconda.

```sh
# Clone the repository. 
git clone https://github.com/chaitjo/learning-tsp.git
cd learning-tsp

# Set up a new conda environment and activate it.
conda create -n tsp python=3.6.7
source activate tsp

# Install all dependencies and Jupyter Lab (for using notebooks).
conda install pytorch=1.2.0 cudatoolkit=10.0 -c pytorch  
conda install numpy scipy cython tqdm scikit-learn matplotlib seaborn tensorboard pandas
conda install jupyterlab -c conda-forge
pip install tensorboard_logger

# Download datasets and unpack to the /data/tsp directory.
pip install gdown
gdown https://drive.google.com/uc?id=152mpCze-v4d0m9kdsCeVkLdHFkjeDeF5
tar -xvzf tsp-data.tar.gz ./data/tsp/
```


## Usage

For reproducing experiments, we provide a set of scripts for training, finetuning and evaluation in the `/scripts` directory. 
Pre-trained models for some experiments described in the paper can be found in the `/pretrained` directory.

Refer to `options.py` for descriptions of each option. 
High-level commands are as follows:
```sh
# Training
CUDA_VISIBLE_DEVICES=<available-gpu-ids> python run.py 
    --problem <tsp/tspsl> 
    --model <attention/nar> 
    --encoder <gnn/gat/mlp> 
    --baseline <rollout/critic> 
    --min_size <20/50/100> 
    --max_size <50/100/200>
    --batch_size 128 
    --train_dataset data/tsp/tsp<20/50/100/20-50>_train_concorde.txt 
    --val_datasets data/tsp/tsp20_val_concorde.txt data/tsp/tsp50_val_concorde.txt data/tsp/tsp100_val_concorde.txt
    --lr_model 1e-4
    --run_name <custom_run_name>
    
# Evaluation
CUDA_VISIBLE_DEVICES=<available-gpu-ids> python eval.py data/tsp/tsp10-200_concorde.txt
    --model outputs/<custom_run_name>_<datetime>/
    --decode_strategy <greedy/sample/bs> 
    --eval_batch_size <128/1/16>
    --width <1/128/1280>
```

## Citation and Resources
**Citation:**
```
@inproceedings{joshi2021learning,
  title={Learning TSP Requires Rethinking Generalization},
  author={Joshi, Chaitanya K and Cappart, Quentin and Rousseau, Louis-Martin and Laurent, Thomas},
  booktitle={International Conference on Principles and Practice of Constraint Programming},
  year={2021}
}
```

**Resources:**
- [ArXiv paper](https://arxiv.org/abs/2006.07054)
- [Blog post on neural combinatorial optimization](http://chaitjo.github.io/neural-combinatorial-optimization/)
- [TSP datasets generated with Concorde](https://drive.google.com/uc?id=152mpCze-v4d0m9kdsCeVkLdHFkjeDeF5)

**Acknowledgement and Related Work:** Our codebase is a modified clone of [Wouter Kool's excellent repository](https://github.com/wouterkool/attention-learn-to-route) for the paper ["Attention, Learn to Solve Routing Problems!"](https://openreview.net/forum?id=ByxBFsRqYm), and incorporates ideas from the following papers, among others:
- [W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.](https://openreview.net/forum?id=ByxBFsRqYm)
- [M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.](https://link.springer.com/chapter/10.1007/978-3-319-93031-2_12)
- [C. K. Joshi, T. Laurent, and X. Bresson. An efficient graph convolutional network technique for the travelling salesman problem. arXiv preprint arXiv:1906.01227, 2019.](https://arxiv.org/abs/1906.01227)
- [A. Nowak, S. Villar, A. S. Bandeira, and J. Bruna. A note on learning algorithms for quadratic assignment with graph neural networks. arXiv preprint arXiv:1706.07450, 2017.](https://arxiv.org/abs/1706.07450v1)
- [I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio. Neural combinatorial optimization with reinforcement learning. In International Conference on Learning Representations, 2017.](https://arxiv.org/abs/1611.09940)





# Pre-entrenamiento

El proyecto está en dos ordenadores (piranesi_03 y minerva), el procedimiento en cada uno de ellos es ligeramente distinto. Por lo que especifico las instrucciones para cada uno de ellos.
Iniciar el proyecto:
En ambos está en una carpeta llamada “learn-tsp” -> cd learning-tsp/
En piranesi_03 se utiliza un entorno de conda -> conda activate learn_tsp
En minerva se utiliza un Docker: docker run --user $(id -u):$(id -g) --gpus all -it --rm -v $(pwd):/app learn_tsp bash

Ejecución de los scripts:
En esta sección la única diferencia es que el proyecto no funciona cuando trabajan dos GPU en paralelo. Para piranesi_03 esto no supone nada, pero para minerva sí. Por lo que delante de cada comando que se ejecute en minerva se debe añadir -> CUDA_VISIBLE_DEVICES=0


Pre-entrenamiento:
Para pre-entrenar se ejecuta el script “pretrain.py”. Se guardará la red pre-entrenada con el nombre que indiques con “run_name”. En caso de querer continuar un pre-entrenamiento de un modelo pre-entrenado a medias, indicas su path con “load_path”. El pre-entrenamiento se escoge con “pretrain_type”, entre las siguientes opciones ('rotation', 'symmetry', 'translation', 'hybrid'). En caso de escoger 'hybrid', utiliza una variable llamada “hybrid_transformations” para indicar cuales quieres utilizar (rot trans sym). La cantidad de nodos mínima y máxima de los TSP se pueden indicar con “min_size” y “max_size”. Un ejemplo de comando para ejecutar un pre-entrenamiento híbrido sería:

```bash
python pretrain.py --encoder gnn --aggregation max --embedding_dim 128 --normalization layer --learn_norm --epoch_size 128000 --n_epochs 50 --run_name "mi_preentrenamiento" --gated --pretrain_type hybrid --hybrid_transformations rot trans sym
```

Los modelos se guardan en la carpeta “outputs”.


# Entrenamiento
Para entrenar se ejecuta el script “run.py”. Nuevamente al entrenamiento se le nombra con la variable “run_name”. Como se pueden entrenar modelos desde 0 o pre-entrenados, existe una variable llamada “load_path” en la que, si queremos entrenar un modelo pre-entrenado le damos el valor de la ruta relativa de los pesos de la red pre-entrenada, y si no queremos usar un modelo, pre-entrenado, no llamamos a esa variable. Un ejemplo de comando para ejecutar un entrenamiento sería:

```bash
python run.py --min_size 50 --max_size 50 --encoder gnn --gated --normalization layer --learn_norm --lr_model 0.0001 --epoch_size 128000 --batch_size 256 --n_epochs 50 --load_path "outputs/tsp_20-50/mi_preentrenamiento_20260223T214729/encoder-epoch-46.pt" --run_name "TSP50_pretrained"
```

Nuevamente, los modelos se guardan en la carpeta “outputs”.


# Evaluación
Para evaluar se ejecuta el script “compare_tsp.py”. Se comparan los modelos indicados por su ruta relativa en las variables “load_path” y “resume”. “eval_size” indica el número de tsp que se evaluarán, y “min_size” el tamaño de los TSP. Si quieres comparar un modelo con concorde, no utilizas la variable “resume”, y a cambio añades “--solver concorde”. Y si quieres evaluar siempre sobre los mismos escenarios, se puede utilizar una semolla concreta “--seed 1”. Un ejemplo de comando para comparar 2 modelos es:

```bash
python compare_tsp.py --load_path outputs/tsp_20-50/mi_preentrenamiento_20260223T214729/encoder-epoch-49.pt --resume outputs/tsp_50-50/TSP_pretrained_normbueno_nofreeze_20260301T235836/epoch-49.pt --embedding_dim 128 --gated --eval_size 10 --normalization layer --min_size 50
```

Si queremos comparar un modelo con un algoritmo de resolucción, seleccionaremos el modelo en "load_path" e indicaremos la comparativa con en solver con `--solver x`. Están disponibles los siguientes solvers `concorde, lkh, ortools, aco, ga`.

-lkh es LKH-3
-ortools es Google OR-Tools (hace una primera ruta usando NN, y luego "desenmaraña" usando 3-opt). Al utilizar este algoritmo, indicaremos con la variable `--time_limit` cuanto tiempo queremos que dedique a resolver cada instancia del TSP.
-aco es Ant Colony Optimization
-ga es Genetic Algorithms

Por pantalla te saldrán varios valores calculados y en la carpeta se te generará una imagen
“comparativa_concorde.png” o “comparativa_modelo 2.png” según corresponda.

Si quieres ver las gráficas de cómo han ido los entrenamientos/pre-entrenamientos -> tensorboard --logdir logs/
