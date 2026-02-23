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

def save_rotation_check(nodes, rotated_nodes, save_dir):
    """
    Guarda una imagen comparativa del grafo original y el rotado.
    """
    # Usar el backend 'Agg' para que no intente abrir una ventana (necesario para SSH)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Configuración de los ejes
    for ax in [ax1, ax2]:
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_aspect('equal')
        # Dibujar el centro de rotación (0.5, 0.5) para referencia
        ax.plot(0.5, 0.5, 'rx') 

    # Graficar Original
    ax1.scatter(nodes[:, 0], nodes[:, 1], c='blue', label='Original')
    ax1.set_title("Grafo Original")
    
    # Graficar Rotado
    ax2.scatter(rotated_nodes[:, 0], rotated_nodes[:, 1], c='green', label='Rotado')
    ax2.set_title("Grafo Rotado")

    # Guardar
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    output_path = os.path.join(save_dir, 'rotation_test.png')
    plt.savefig(output_path)
    plt.close()
    print(f"\n[*] Visualización de rotación guardada en: {output_path}")

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
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Imagen generada en: {save_path}")