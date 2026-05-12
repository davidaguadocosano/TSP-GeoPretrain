import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from options import get_options
from utils import load_problem
from problems.tsp.problem_tsp import nearest_neighbor_graph

from nets.attention_model import AttentionModel
from nets.encoders.gnn_encoder import GNNEncoder

from problems.tsp.problem_tsp import solve_concorde, solve_lkh  # dac
from tqdm import tqdm

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

def plot_comparison(nodes, tour1, cost1, tour2, cost2, filename, avg1=None, ci1=None, avg2=None, ci2=None, n_eval=1, name2="Solucionador"):
    # nodes llega como [1, V, 2], lo convertimos a numpy para graficar
    nodes = nodes[0].cpu().numpy()
    tour1 = tour1.cpu().numpy()
    tour2 = tour2.cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Títulos dinámicos basados en si hay evaluación por lote o no
    title1 = f"Modelo 1 | Coste: {cost1:.4f}"
    title2 = f"Modelo 2 | Coste: {cost2:.4f}"
    
    if n_eval > 1:
        title1 += f"\n Modelo 1 |Media ({n_eval} TSPs): {avg1:.4f} ± {ci1:.4f} (95% CI)"
        title2 += f"\n {name2} | Media ({n_eval} TSPs): {avg2:.4f} ± {ci2:.4f} (95% CI)"

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


"""
def run_comparison(opts):

    if opts.seed is not None and opts.seed != 1234:
        seed = opts.seed
        print(f"[*] Usando semilla fija: {seed}")
    else:
        # Generamos una semilla basada en el tiempo si no se especifica
        import time
        seed = int(time.time() * 1000) & 0xFFFFFFFF
        print(f"[*] Generando escenario aleatorio (Semilla dinámica: {seed})")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
"""

def run_comparison(opts):
    # --- 0. Configuración de Semilla ---
    if opts.seed is not None and opts.seed != 1234:
        seed = opts.seed
        print(f"[*] Usando semilla fija: {seed}")
    else:
        import time
        seed = int(time.time() * 1000) & 0xFFFFFFFF
        print(f"[*] Generando escenario aleatorio (Semilla dinámica: {seed})")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if opts.use_cuda and torch.cuda.is_available() else "cpu")
    
    # --- 1. Cargar Modelo 1 (Siempre necesario) ---
    model1 = load_model_custom(opts.load_path, opts, device)

    # --- 2. Preparar Modelo 2 o Solucionador ---
    # Usaremos variables genéricas para el "segundo competidor"
    model2 = None
    competitor_name = "Modelo 2"
    
    if opts.solver is not None:
        competitor_name = opts.solver.upper()
        print(f"[*] Modo comparación: Modelo 1 vs {competitor_name}")
    else:
        model2 = load_model_custom(opts.resume, opts, device)
        print(f"[*] Modo comparación: Modelo 1 vs Modelo 2")

    # --- 3. Generación de Nodos y Grafos ---
    num_nodes = opts.min_size
    eval_size = opts.eval_size 
    
    print(f"Generando {eval_size} instancias de TSP con {num_nodes} nodos...")
    nodes = torch.rand(eval_size, num_nodes, 2).to(device)

    print(f"Generando grafos k-NN...")
    batch_graphs = []
    for i in range(eval_size):
        g_data = nearest_neighbor_graph(nodes[i].cpu(), opts.neighbors, opts.knn_strat)
        batch_graphs.append(torch.tensor(g_data).bool())
    graphs = torch.stack(batch_graphs).to(device)
    
    # --- 4. Inferencia: Resolver Problemas ---
    
    # Inferencia Modelo 1
    with torch.no_grad():
        costs1, _, tours1 = model1(nodes, graphs, return_pi=True)

    # Inferencia Competidor (Modelo 2 o Solver)
    costs2_list = []
    tours2_list = []

    if opts.solver is not None:
        # IMPORTANTE: Importar aquí para evitar errores si no están instalados
        from problems.tsp.problem_tsp import solve_concorde, solve_lkh
        from tqdm import tqdm

        print(f"Ejecutando {competitor_name}...")
        for i in tqdm(range(eval_size)):
            # El solver recibe numpy en CPU
            curr_nodes = nodes[i].cpu().numpy()
            
            if opts.solver.lower() == 'concorde':
                cost, tour = solve_concorde(curr_nodes)
            elif opts.solver.lower() == 'lkh':
                cost, tour = solve_lkh(curr_nodes)
            else:
                raise ValueError(f"Solver {opts.solver} no reconocido")
            
            costs2_list.append(cost)
            tours2_list.append(torch.tensor(tour))
        
        costs2 = torch.tensor(costs2_list).to(device)
        tours2 = torch.stack(tours2_list).to(device)
    else:
        # Inferencia normal Modelo 2
        with torch.no_grad():
            costs2, _, tours2 = model2(nodes, graphs, return_pi=True)

    # --- 5. Estadísticas y Gap ---
    n = costs1.size(0)
    avg_cost1, avg_cost2 = costs1.mean().item(), costs2.mean().item()
    std1, std2 = costs1.std().item(), costs2.std().item()
    
    sem1, sem2 = std1 / (n**0.5), std2 / (n**0.5)
    ci1, ci2 = 1.96 * sem1, 1.96 * sem2

    # Si comparamos contra un solver, calculamos el GAP porcentual
    if opts.solver is not None:
        gap = ((costs1 - costs2) / costs2).mean().item() * 100
        print(f"\n[!] OPTIMALITY GAP PROMEDIO: {gap:.4f}%")

    current_cost1 = costs1[0].item()
    current_cost2 = costs2[0].item()
    
    print(f"\nResultados finales sobre {eval_size} instancias:")
    print(f"Modelo 1 - Media: {avg_cost1:.4f} ± {ci1:.4f}")
    print(f"{competitor_name} - Media: {avg_cost2:.4f} ± {ci2:.4f}")

    # --- 6. Visualización ---
    plot_comparison(
        nodes[0:1], 
        tours1[0], 
        current_cost1, 
        tours2[0], 
        current_cost2, 
        f"comparativa_{competitor_name.lower()}.png",
        avg1=avg_cost1, ci1=ci1,
        avg2=avg_cost2, ci2=ci2,
        n_eval=n,
        name2=competitor_name # Pasa el nombre del competidor a la función de dibujo
    )


if __name__ == "__main__":
    opts = get_options()
    run_comparison(opts)