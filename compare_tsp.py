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

def plot_comparison(nodes, tour1, cost1, tour2, cost2, filename, avg1=None, ci1=None, avg2=None, ci2=None, n_eval=1):
    # nodes llega como [1, V, 2], lo convertimos a numpy para graficar
    nodes = nodes[0].cpu().numpy()
    tour1 = tour1.cpu().numpy()
    tour2 = tour2.cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Títulos dinámicos basados en si hay evaluación por lote o no
    title1 = f"Modelo 1 | Coste: {cost1:.4f}"
    title2 = f"Modelo 2 | Coste: {cost2:.4f}"
    
    if n_eval > 1:
        title1 += f"\nMedia ({n_eval} TSPs): {avg1:.4f} ± {ci1:.4f} (95% CI)"
        title2 += f"\nMedia ({n_eval} TSPs): {avg2:.4f} ± {ci2:.4f} (95% CI)"

    for i, (tour, cost, title) in enumerate([(tour1, cost1, title1), (tour2, cost2, title2)]):
        # Dibujar nodos
        ax[i].scatter(nodes[:, 0], nodes[:, 1], c='red', zorder=2)
        
        # Dibujar la ruta (unimos el último nodo con el primero para cerrar el tour)
        tour_nodes = nodes[tour]
        tour_nodes = np.vstack([tour_nodes, tour_nodes[0]])
        ax[i].plot(tour_nodes[:, 0], tour_nodes[:, 1], c='blue', linewidth=1, zorder=1)
        
        ax[i].set_title(title, fontsize=12, fontweight='bold')
        ax[i].set_aspect('equal')
        ax[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"[*] Imagen guardada con estadísticas en: {filename}")
    plt.close()

def run_comparison(opts):
    device = torch.device("cuda:0" if opts.use_cuda and torch.cuda.is_available() else "cpu")
    
    # 1. Cargar modelos
    model1 = load_model_custom(opts.load_path, opts, device)
    model2 = load_model_custom(opts.resume, opts, device)

    # 2. Configuración y Generación de Nodos en Lote (Batch)
    num_nodes = opts.min_size
    eval_size = opts.eval_size  # Obtenido del terminal (por defecto 1)
    
    print(f"Generando {eval_size} instancias de TSP con {num_nodes} nodos...")
    # Generamos un tensor de tamaño [eval_size, num_nodes, 2]
    # Esto crea 'eval_size' problemas diferentes de una sola vez
    nodes = torch.rand(eval_size, num_nodes, 2).to(device)

    # 3. Generar los Grafos k-NN para todo el lote
    print(f"Generando grafos k-NN para las {eval_size} instancias...")
    batch_graphs = []
    for i in range(eval_size):
        # Procesamos cada instancia para crear su matriz de adyacencia
        g_data = nearest_neighbor_graph(nodes[i].cpu(), opts.neighbors, opts.knn_strat)
        batch_graphs.append(torch.tensor(g_data).bool())
    
    # Apilamos los grafos en un solo tensor de [eval_size, num_nodes, num_nodes]
    graphs = torch.stack(batch_graphs).to(device)
    
    # 4. Inferencia: Resolvemos todos los problemas a la vez
    with torch.no_grad():
        # El AttentionModel procesa el batch completo [eval_size, ...] 
        # y devuelve tensores con todos los costes y rutas
        costs1, _, tours1 = model1(nodes, graphs, return_pi=True)
        costs2, _, tours2 = model2(nodes, graphs, return_pi=True)

    # 5. Calcular Medias y preparar Visualización
    n = costs1.size(0) # eval_size, para intervalo confianza
    avg_cost1 = costs1.mean().item()
    avg_cost2 = costs2.mean().item()

    # Calculamos la desviación estándar muestral (unbiased)
    std1 = costs1.std().item()
    std2 = costs2.std().item()

    # Error Estándar de la Media (SEM)
    sem1 = std1 / (n**0.5)
    sem2 = std2 / (n**0.5)
    
    # Intervalo de Confianza del 95% (Z ≈ 1.96 para n suficientemente grande)
    ci1 = 1.96 * sem1
    ci2 = 1.96 * sem2

    # Extraemos solo la PRIMERA instancia para el dibujo
    # Usamos .item() para convertir los tensores en números normales
    current_cost1 = costs1[0].item()
    current_cost2 = costs2[0].item()
    
    # Pasamos solo los datos del primer TSP a la función de dibujo
    # Pero ahora incluimos las medias en el título o etiquetas
    print(f"\nResultados sobre {eval_size} instancias:")
    print(f"Modelo 1 - Media: {avg_cost1:.4f} ± {ci1:.4f} (95% CI) | Actual: {current_cost1:.4f}")
    print(f"Modelo 2 - Media: {avg_cost2:.4f} ± {ci2:.4f} (95% CI) | Actual: {current_cost2:.4f}")

    # Modificamos ligeramente la llamada al plot para pasar las medias si quieres
    # (O simplemente usa los costes actuales para mantener la imagen limpia)
    plot_comparison(
        nodes[0:1], # Solo el primer conjunto de nodos [1, V, 2]
        tours1[0],  # Solo la primera ruta del modelo 1
        current_cost1, 
        tours2[0],  # Solo la primera ruta del modelo 2
        current_cost2, 
        "comparativa_modelos.png",
        avg1=avg_cost1, ci1=ci1,
        avg2=avg_cost2, ci2=ci2,
        n_eval=n
    )

if __name__ == "__main__":
    opts = get_options()
    run_comparison(opts)