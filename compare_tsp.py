import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from options import get_options
from utils import load_problem
from problems.tsp.problem_tsp import nearest_neighbor_graph

from nets.attention_model import AttentionModel
from nets.encoders.gnn_encoder import GNNEncoder

def load_model_custom(path, opts, device):
    problem = load_problem('tsp')
    model = AttentionModel(
        problem=problem,
        embedding_dim=opts.embedding_dim,
        encoder_class=GNNEncoder,
        n_encode_layers=opts.n_encode_layers,
        aggregation=opts.aggregation,
        normalization=opts.normalization,
        gated=opts.gated,
        n_heads=opts.n_heads
    ).to(device)

    if path is not None:
        print(f"Cargando {path}...")
        checkpoint = torch.load(path, map_location=device)
        if 'encoder' in checkpoint:
            model.embedder.load_state_dict(checkpoint['encoder'])
            model.init_embed.load_state_dict(checkpoint['init_embed'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
    
    model.eval()
    # SOLUCIÓN AL ERROR DE IMPORTACIÓN: 
    # Seteamos el tipo de decodificación directamente en el atributo del modelo
    model.set_decode_type("greedy") 
    return model

def plot_comparison(nodes, tour1, cost1, tour2, cost2, save_path):
    nodes = nodes.cpu().numpy()[0]
    # Extraer los tours (en este repo suelen venir en el tercer valor del return)
    t1 = tour1[0].cpu().numpy()
    t2 = tour2[0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    tours = [t1, t2]
    costs = [cost1[0].item(), cost2[0].item()]
    titles = ["Modelo 1 (Path 1)", "Modelo 2 (Path 2)"]

    for i, ax in enumerate(axes):
        ax.scatter(nodes[:, 0], nodes[:, 1], c='red', s=40, zorder=2)
        # Cerrar el ciclo para la visualización
        full_tour = np.append(tours[i], tours[i][0])
        path_coords = nodes[full_tour]
        ax.plot(path_coords[:, 0], path_coords[:, 1], c='blue', linewidth=1.5, zorder=1)
        ax.set_title(f"{titles[i]}\nCosto Total: {costs[i]:.4f}")
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[*] Comparación guardada en: {save_path}")

def run_comparison(opts):
    device = torch.device("cuda:0" if opts.use_cuda and torch.cuda.is_available() else "cpu")
    
    path1 = opts.load_path 
    path2 = opts.resume 
    
    model1 = load_model_custom(path1, opts, device)
    model2 = load_model_custom(path2, opts, device)

    # 2. Generar nodos
    #torch.manual_seed(42)
    num_nodes = opts.min_size #50
    nodes = torch.rand(1, num_nodes, 2).to(device)

    # 3. GENERAR EL GRAFO (Esto es lo que faltaba)
    # GNN necesita saber qué nodos están conectados con cuáles
    # Usamos los parámetros que vienen en opts (neighbors y knn_strat)
    print("Generando grafo k-NN para el modelo GNN...")
    graph_data = nearest_neighbor_graph(nodes[0].cpu(), opts.neighbors, opts.knn_strat)
    # Cambiamos ByteTensor por un tensor booleano para evitar avisos de deprecación
    graph = torch.tensor(graph_data).to(device).bool().unsqueeze(0)
    
    # 4. PASAR AMBOS al modelo
    with torch.no_grad():
        # Ahora pasamos (nodes, graph)
        cost1, _, tour1 = model1(nodes, graph, return_pi=True)
        cost2, _, tour2 = model2(nodes, graph, return_pi=True)

    plot_comparison(nodes, tour1, cost1, tour2, cost2, "comparativa_modelos.png")

if __name__ == "__main__":
    opts = get_options()
    run_comparison(opts)