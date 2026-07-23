import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from options import get_options
from utils import load_problem
from problems.tsp.problem_tsp import nearest_neighbor_graph

from nets.attention_model import AttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from tqdm import tqdm

def load_model_custom(path, opts, device):
    problem = load_problem('tsp')
    
    checkpoint = None
    is_gated = opts.gated  
    
    # 1. IMPORTAR AMBAS ARQUITECTURAS
    from nets.encoders.gnn_encoder import GNNEncoder
    from nets.encoders.gat_encoder import GraphAttentionEncoder

    if path is not None:
        print(f"    Cargando pesos desde {path}...")
        checkpoint = torch.load(path, map_location=device)
        
        # 2. INFERENCIA BINARIA: Inspeccionamos el diccionario de pesos
        state_dict = checkpoint.get('model', checkpoint.get('encoder', {}))
        is_gated = any('layers.0.U.weight' in k or 'layers.0.A.weight' in k for k in state_dict.keys())
        
    # 3. ASIGNACIÓN DIRECTA
    if is_gated:
        print("      -> Arquitectura detectada: GatedGCN (Pre-entrenada / Isometrías)")
        EncoderClass = GNNEncoder
    else:
        print("      -> Arquitectura detectada: Transformer GAT (Base sin pretrain)")
        EncoderClass = GraphAttentionEncoder

    # 4. CONSTRUCCIÓN DINÁMICA DE ARGUMENTOS
    kwargs = {
        'problem': problem,
        'embedding_dim': opts.embedding_dim,
        'encoder_class': EncoderClass, 
        'n_encode_layers': opts.n_encode_layers,
        'n_heads': opts.n_heads
    }

    # Añadir parámetros específicos solo si es tu GatedGCN
    if is_gated:
        kwargs['aggregation'] = getattr(opts, 'aggregation', 'max')
        kwargs['normalization'] = getattr(opts, 'normalization', 'layer')
        kwargs['gated'] = True

    # 5. INSTANCIACIÓN A PRUEBA DE FALLOS
    try:
        model = AttentionModel(**kwargs).to(device)
    except TypeError:
        # Si es la GAT clásica y protesta por los argumentos de la GNN, los limpiamos
        kwargs.pop('aggregation', None)
        kwargs.pop('normalization', None)
        kwargs.pop('gated', None)
        model = AttentionModel(**kwargs).to(device)

    # 6. CARGA DE PESOS EN LA ARQUITECTURA CONSTRUIDA
    if checkpoint is not None:
        if 'encoder' in checkpoint:
            model.embedder.load_state_dict(checkpoint['encoder'])
            model.init_embed.load_state_dict(checkpoint['init_embed'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
    
    model.eval()
    model.set_decode_type("greedy") 
    return model

# Modificado para aceptar N resultados y dibujar N subplots dinámicamente
def plot_comparison(nodes, results, filename, n_eval=1):
    n_plots = len(results)
    fig, ax = plt.subplots(1, n_plots, figsize=(7 * n_plots, 7))
    if n_plots == 1:
        ax = [ax]

    nodes = nodes[0].cpu().numpy()

    for i, res in enumerate(results):
        tour = res['tour'].cpu().numpy() if torch.is_tensor(res['tour']) else np.array(res['tour'])
        
        title = f"{res['name']} | Coste: {res['current_cost']:.4f}"
        if n_eval > 1:
            title += f"\nMedia: {res['avg_cost']:.4f} ± {res['ci']:.4f} (95% CI)\nTiempo/inst: {res['avg_time']:.6f}s"

        ax[i].scatter(nodes[:, 0], nodes[:, 1], c='red', zorder=2)
        
        tour_nodes = nodes[tour]
        tour_nodes = np.vstack([tour_nodes, tour_nodes[0]])
        ax[i].plot(tour_nodes[:, 0], tour_nodes[:, 1], c='blue', linewidth=1, zorder=1)
        
        ax[i].set_title(title, fontsize=11, fontweight='bold')
        ax[i].set_aspect('equal')
        ax[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n[*] Imagen comparativa guardada en: {filename}")
    plt.close()

def run_comparison(opts):
    # 1. FIJAR SEMILLA Y GENERAR DATOS
    if opts.seed is not None and opts.seed != 1234:
        seed = opts.seed
        print(f"[*] Usando semilla fija: {seed}")
    else:
        seed = int(time.time() * 1000) & 0xFFFFFFFF
        print(f"[*] Generando escenario aleatorio (Semilla dinámica: {seed})")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if opts.use_cuda and torch.cuda.is_available() else "cpu")
    is_cuda = device.type == 'cuda'
    
    # [!] MODO LEGACY: "Quemar" el generador de números aleatorios para recuperar escenarios antiguos
    print("[!] Activando modo de compatibilidad para recuperar escenarios antiguos...")
    _ = load_model_custom(None, opts, device)
    _ = load_model_custom(None, opts, device)

    num_nodes = opts.min_size
    eval_size = opts.eval_size 
    
    print(f"\n[*] Generando {eval_size} instancias de TSP con {num_nodes} nodos...")
    nodes = torch.rand(eval_size, num_nodes, 2).to(device)

    print(f"[*] Generando grafos k-NN...")
    batch_graphs = []
    for i in range(eval_size):
        g_data = nearest_neighbor_graph(nodes[i].cpu(), opts.neighbors, opts.knn_strat)
        batch_graphs.append(torch.tensor(g_data).bool())
    graphs = torch.stack(batch_graphs).to(device)

    # 2. DEFINIR LISTA DE COMPETIDORES
    competitors = []
    
    model_paths = []
    if opts.load_path is not None:
        model_paths.append(opts.load_path)
    if opts.resume is not None:
        model_paths.append(opts.resume)

    for idx, path in enumerate(model_paths):
        competitors.append({
            'name': f'Modelo {idx+1}',
            'type': 'neural',
            'path': path
        })

    if opts.solver is not None:
        competitors.append({
            'name': opts.solver.upper(),
            'type': 'solver',
            'solver_name': opts.solver.lower()
        })

    # 3. EVALUACIÓN SECUENCIAL
    results = []
    
    for comp in competitors:
        print(f"\n================ Evaluando: {comp['name']} ================")
        
        costs_list = []
        tours_list = []
        
        if comp['type'] == 'neural':
            model = load_model_custom(comp['path'], opts, device)
            
            if is_cuda: torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                costs, _, tours = model(nodes, graphs, return_pi=True)
                
            if is_cuda: torch.cuda.synchronize()
            total_time = time.time() - start_time
            avg_time = total_time / eval_size
            
        elif comp['type'] == 'solver':
            from problems.tsp.problem_tsp import solve_concorde, solve_lkh
            
            start_time = time.time()
            for i in tqdm(range(eval_size), desc="Resolviendo"):
                curr_nodes = nodes[i].cpu().numpy()
                if comp['solver_name'] == 'concorde':
                    cost, tour = solve_concorde(curr_nodes)
                elif comp['solver_name'] == 'lkh':
                    cost, tour = solve_lkh(curr_nodes)
                elif comp['solver_name'] == 'ortools':
                    from problems.tsp.problem_tsp import solve_ortools
                    cost, tour = solve_ortools(curr_nodes, time_limit=opts.time_limit)
                elif comp['solver_name'] in ['aco', 'ga']:
                    from problems.tsp.problem_tsp import solve_pycombinatorial
                    cost, tour = solve_pycombinatorial(curr_nodes, algorithm=comp['solver_name'])
                else:
                    raise ValueError(f"Solver {comp['solver_name']} no reconocido")
                
                costs_list.append(cost)
                tours_list.append(torch.tensor(tour))
                
            total_time = time.time() - start_time
            avg_time = total_time / eval_size
            
            costs = torch.tensor(costs_list).to(device)
            tours = torch.stack(tours_list).to(device)

        n = costs.size(0)
        avg_cost = costs.mean().item()
        std = costs.std().item()
        ci = 1.96 * (std / (n**0.5))
        
        print(f"    -> Coste Medio: {avg_cost:.4f} ± {ci:.4f}")
        print(f"    -> Tiempo Medio: {avg_time:.6f}s")

        results.append({
            'name': comp['name'],
            'costs': costs,
            'tour': tours[0],
            'current_cost': costs[0].item(),
            'avg_cost': avg_cost,
            'ci': ci,
            'avg_time': avg_time
        })

    # 4. REPORTE FINAL
    print(f"\n================ RESUMEN FINAL ({eval_size} INSTANCIAS) ================")
    
    reference_idx = -1 if competitors[-1]['type'] == 'solver' else 0
    ref_cost = results[reference_idx]['avg_cost']
    ref_time = results[reference_idx]['avg_time']

    for i, res in enumerate(results):
        gap = ((res['avg_cost'] - ref_cost) / ref_cost) * 100
        speedup = ref_time / res['avg_time'] if res['avg_time'] > 0 else 0
        
        print(f"{res['name']:<12} | Coste: {res['avg_cost']:<8.4f} ± {res['ci']:<6.4f} | Tiempo: {res['avg_time']:.6f}s", end="")
        
        if len(results) > 1 and i != reference_idx and competitors[-1]['type'] == 'solver':
            print(f" | GAP: {gap:>6.2f}% | Speedup: {speedup:>6.2f}x", end="")
        print("")

    plot_comparison(nodes, results, "comparativa_modelos.png", n_eval=eval_size)

if __name__ == "__main__":
    opts = get_options()
    run_comparison(opts)