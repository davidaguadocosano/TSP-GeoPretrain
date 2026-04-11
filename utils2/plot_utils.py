import numpy as np
import matplotlib.pyplot as plt


def plot_tsp(actions: np.ndarray, batch: dict, reward: int = 0) -> None:

    # Initialize plot
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])

    # Data
    nodes = batch['nodes']

    # Plot nodes
    plt.scatter(nodes[..., 0], nodes[..., 1], c='mediumpurple', s=180)
    if 'obstacles' in batch:
        for obs in batch['obstacles']:
            ax.add_patch(plt.Circle(obs[:2], obs[2], color='k'))

    # Plot regions numbers (indexes)
    for i in range(nodes.shape[0]):
        plt.text(nodes[i, 0], nodes[i, 1], str(i))

    # Draw arrows
    d = 0
    for i in range(1, len(actions)):
        
        # Update traveled distance
        d += np.linalg.norm(actions[i, 1:] - actions[i - 1, 1:])
        
        # Plot new position
        plt.plot([actions[i - 1, 1], actions[i, 1]], [actions[i - 1, 2], actions[i, 2]], c='g')
        
        # Check if finished
        dist2obs = np.linalg.norm(actions[i, 1:] - batch['obstacles'][:, :2], axis=-1)
        if np.any(dist2obs < batch['obstacles'][:, 2]):
            plt.scatter(*actions[i, 1:], marker='x', c='r', s=90)
            break
    
    # Title
    title = f"TSP Length: {d:.2f} | Reward: {-reward:.2f}"
    plt.title(title)
    print(title)
    
    # Show plot
    plt.show()

#dac
import os

""" No borrar, util capturas TFM.
def save_transformation_check(nodes, transformed_nodes, save_dir, transform_type="rotation", epoch=0, angle=None):
    #Guarda una imagen comparativa del grafo original y el transformado
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import torch

    # Convertir a numpy si son tensores
    if isinstance(nodes, torch.Tensor):
        nodes = nodes.detach().cpu().numpy()
    if isinstance(transformed_nodes, torch.Tensor):
        transformed_nodes = transformed_nodes.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    for ax in [ax1, ax2]:
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_aspect('equal')
        ax.plot(0.5, 0.5, 'rx', markersize=10, label='Centro') 

        # DIBUJAR EL EJE: Si hay ángulo, dibujamos la línea del espejo
        if angle is not None:
            r = 0.7  # Radio de la línea
            # El eje de simetría axial pasa por el centro (0.5, 0.5)
            x1 = 0.5 + r * np.cos(angle)
            y1 = 0.5 + r * np.sin(angle)
            x2 = 0.5 - r * np.cos(angle)
            y2 = 0.5 - r * np.sin(angle)
            ax.plot([x1, x2], [y1, y2], color='red', linestyle='--', alpha=0.6, label='Eje de simetría')

    # Graficar Original (Azul)
    ax1.scatter(nodes[:, 0], nodes[:, 1], c='blue', edgecolors='black', s=50, alpha=0.7)
    ax1.set_title(f"Original")
    
    # Graficar Transformado (Verde)
    ax2.scatter(transformed_nodes[:, 0], transformed_nodes[:, 1], c='green', edgecolors='black', s=50, alpha=0.7)
    ax2.set_title(f"Transformación: {transform_type}")

    #ax1.legend()
    #ax2.legend()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    output_path = os.path.join(save_dir, f'CHECK_{transform_type}_ep{epoch}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[*] Visualización guardada: {output_path}")
"""
def save_transformation_check(nodes, transformed_nodes, save_dir, transform_type="rotation", epoch=0, angle=None):
    """Guarda una imagen comparativa del grafo original y el transformado con marco de referencia."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os
    import numpy as np
    import torch

    # Convertir a numpy si son tensores
    if isinstance(nodes, torch.Tensor):
        nodes = nodes.detach().cpu().numpy()
    if isinstance(transformed_nodes, torch.Tensor):
        transformed_nodes = transformed_nodes.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Límites ampliados para que quepa la traslación sin recortar
    limit_min, limit_max = -0.2, 1.2

    for ax in [ax1, ax2]:
        ax.set_xlim([limit_min, limit_max])
        ax.set_ylim([limit_min, limit_max])
        ax.set_aspect('equal')
        
        # 1. Dibujar el marco original [0,1] como referencia
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--', alpha=0.5)
        ax.add_patch(rect)
        
        # 2. Dibujar el centro original
        ax.plot(0.5, 0.5, 'rx', markersize=8, alpha=0.5) 

        # 3. Dibujar el eje (Solo para simetría)
        if angle is not None:
            r = 0.8 
            x1, y1 = 0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)
            x2, y2 = 0.5 - r * np.cos(angle), 0.5 - r * np.sin(angle)
            ax.plot([x1, x2], [y1, y2], color='red', linestyle='--', alpha=0.6, label='Eje')

    # Graficar Original (Azul)
    ax1.scatter(nodes[:, 0], nodes[:, 1], c='blue', edgecolors='black', s=50, alpha=0.7)
    ax1.set_title(f"Original")
    
    # Graficar Transformado (Verde)
    ax2.scatter(transformed_nodes[:, 0], transformed_nodes[:, 1], c='green', edgecolors='black', s=50, alpha=0.7)
    ax2.set_title(f"Transformación: {transform_type.upper()}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    output_path = os.path.join(save_dir, f'CHECK_{transform_type}_ep{epoch}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[*] Visualización guardada: {output_path}")


#dac para visualizar los resultados en un png (al estar en ssh no se como ir visualizando la panatalla))
def save_training_results(history, label, save_dir, filename):
    """
    Genera y guarda una gráfica de la evolución del entrenamiento/validación.
    """
    import matplotlib
    matplotlib.use('Agg') # Asegura compatibilidad con SSH
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(history)), history, marker='o', linestyle='-', color='b')
    plt.title(f'Evolución de {label} por Época')
    plt.xlabel('Época')
    plt.ylabel(label)
    plt.grid(True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(save_dir, f'{filename}.png')
    plt.savefig(path)
    plt.close()
    print(f"[*] Gráfica de rendimiento guardada en: {path}")

#dac
def plot_tsp_solution(nodes, tour, save_path, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Convertir nodos a numpy si son tensores
    if torch.is_tensor(nodes):
        nodes = nodes.cpu().numpy()
    
    # Convertir tour a numpy entero de forma segura
    if torch.is_tensor(tour):
        tour = tour.cpu().numpy()
    tour = np.atleast_1d(tour).astype(int).flatten()

    plt.figure(figsize=(8, 8))
    # Dibujamos todos los nodos en rojo
    plt.scatter(nodes[:, 0], nodes[:, 1], c='red', zorder=2)
    
    # Crear ruta cerrada si hay más de un nodo
    if len(tour) > 1:
        # Para el TSP cerramos el ciclo volviendo al primero
        full_tour = np.append(tour, tour[0])
        tour_nodes = nodes[full_tour]
        plt.plot(tour_nodes[:, 0], tour_nodes[:, 1], c='blue', linewidth=2, zorder=1)
    
    # Dibujar el inicio (cuadrado verde)
    if len(tour) > 0:
        start_node = tour[0]
        plt.scatter(nodes[start_node, 0], nodes[start_node, 1], c='green', s=100, marker='s', label='Inicio', zorder=3)

    # USAR TÍTULO PERSONALIZADO O DEFECTO
    if title is None:
        title = f"Ruta TSP (Nodos: {len(np.unique(tour))})"
    
    plt.title(title)
    #plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Imagen generada en: {save_path}")