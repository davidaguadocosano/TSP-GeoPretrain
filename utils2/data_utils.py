import torch
import numpy as np
from typing import Tuple, Optional, Union, List


def actions2numpy(actions, nodes, num_dirs=4, time_step=2e-2):
    actions = actions.squeeze(dim=0).detach().cpu().numpy()
    path = []
    for i, action in enumerate(actions):
        if i == 0:
            position = nodes[action[0], :2]
        else:
            angle = action[1] * 2 * np.pi / num_dirs
            position = position + np.array([time_step * np.cos(angle), time_step * np.sin(angle)])
        path.append([action[0], *position])
    return np.array(path)


def batch2numpy(batch):
    for k, v in batch.items():
        batch[k] = v[0].cpu().detach().numpy()
    return batch


def add_flag(vector, flag: int = 0):
    return torch.cat((vector, torch.zeros_like(vector)[..., 0, None] + flag), dim=-1)


def add_prize(vector, reward_type: str = 'const', value: int = 0):
    if reward_type == 'const':
        return torch.cat((vector, torch.zeros_like(vector)[..., 0, None] + value), dim=-1)


def generate_obstacles(min_obs: int = 5, max_obs: int = 10, r_max: float = 0.12, r_min: float = 0.02) -> torch.Tensor:  # r_max: float = 0.2, r_min: float = 0.05
    """
    Generate obstacles.

    Args:
        num_obs (int): Maximum number of obstacles.
        r_max (float): Maximum radius.
        r_min (float): Minimum radius.

    Returns:
        torch.Tensor: Obstacles.
    """
    if max_obs <= 0:
        return None

    # Number of obstacles
    num_obs = torch.randint(low=min_obs, high=max_obs + 1, size=[1])[0]

    # Generate random obstacles (circles)
    radius = torch.rand(num_obs) * (r_max - r_min) + r_min
    center = torch.rand((num_obs, 2))
    obstacles = torch.cat((center, radius[..., None]), dim=-1)

    # Pad with (-1, -1, 0) where num_obstacles < num_obs
    obstacles = torch.nn.functional.pad(
        input=obstacles,
        pad=(0, 0, 0, max_obs - obstacles.shape[0]),
        mode='constant',
        value=-1
    )
    obstacles[..., 2][obstacles[..., 2] == -1] = 0
    return obstacles

#dac
def generate_nodes(
    num_nodes: int,
    num_depots: int = 1,
    obs: Optional[torch.Tensor] = None,
    image_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate nodes.

    Args:
        num_nodes (int): Number of nodes.
        num_depots (int): Number of depots.
        obs (torch.Tensor): Obstacles.
        image_size (int): Binary map shape (assume squared image).

    Returns:
        tuple: nodes, initial depot, and end depot.
    """
    node_size = 0.02
    num_nodes = num_nodes + num_depots

    # Meshgrid
    #num_obs = obs.shape[0]
    x, y = torch.meshgrid(torch.linspace(0, 1, image_size), torch.linspace(0, 1, image_size), indexing="ij")
    #xy = torch.stack((x, y), dim=-1).expand([num_obs, image_size, image_size, 2])
    xy_base = torch.stack((x, y), dim=-1)

    # No obstacles
    if obs is None:
        #points = torch.FloatTensor(num_nodes, 2).uniform_(0, 1)
        #obs_mask = torch.ones(image_size, image_size)
        #dac para que no se desborden al rotarlos
        theta = 2 * np.pi * torch.rand(num_nodes)
        r = 0.5 * torch.sqrt(torch.rand(num_nodes))
        
        # Convertir a cartesianas y desplazar al centro (0.5, 0.5)
        points_x = 0.5 + r * torch.cos(theta)
        points_y = 0.5 + r * torch.sin(theta)
        points = torch.stack((points_x, points_y), dim=-1)

        obs_mask = torch.zeros(image_size, image_size, dtype=torch.bool)
        xy_for_nodes = xy_base.unsqueeze(0)

    # Obstacles
    else:
        num_obs = obs.shape[0]
        # Expandir malla para comparar con cada centro de obstáculo
        xy = xy_base.expand([num_obs, image_size, image_size, 2])
        obs_center = obs[..., :2]
        obs_radius = obs[..., 2]

        # Calculate distance squared from each point of the meshgrid to each obstacle center
        distances = (xy - obs_center[..., None, None, :]).norm(2, dim=-1)
        obs_free_mask = (distances > obs_radius[..., None, None] + node_size).all(dim=0).T

        # 2. NUEVA RESTRICCIÓN: El punto debe estar dentro del círculo inscrito (radio <= 0.5)
        dist_to_center = (xy_base - 0.5).norm(2, dim=-1)
        in_circle_mask = dist_to_center <= 0.5
        
        # Combinamos ambas: zona libre Y dentro del círculo
        valid_mask = obs_free_mask & in_circle_mask

        # Generar puntos de la lista de candidatos válidos
        non_colliding_points = torch.nonzero(valid_mask)
        # Mantenemos el swap de índices original para coherencia con el proyecto
        non_colliding_points = torch.stack((non_colliding_points[..., 1], non_colliding_points[..., 0]), dim=-1)
        
        # Selección aleatoria y normalización
        points = non_colliding_points[torch.randperm(non_colliding_points.shape[0])[:num_nodes]] / float(image_size - 1)
        
        obs_mask = ~obs_free_mask
        
        """
        # Create the masks by comparing distances with squared radius
        obs_mask = (distances > obs_radius[..., None, None] + node_size).all(dim=0).T

        # Generate non-colliding points
        non_colliding_points = torch.nonzero(obs_mask)
        non_colliding_points = torch.stack((non_colliding_points[..., 1], non_colliding_points[..., 0]), dim=-1)
        points = non_colliding_points[torch.randperm(non_colliding_points.shape[0])[:num_nodes]] / float(image_size - 1)
        obs_mask = ~obs_mask
        """
        xy_for_nodes = xy[0, None]
    
    # Separate regions and depots
    depot_ini = points[0] if num_depots > 0 else None
    depot_end = points[1] if num_depots == 2 else points[0]
    nodes = points[num_depots:]
    
    # Generate binary map
    binary_map = torch.zeros(image_size, image_size, 2 + num_depots)
    # Canal 0: Nodos
    binary_map[..., 0] = ((xy_for_nodes - nodes[..., None, None, :]).norm(2, dim=-1) < node_size).sum(dim=0).permute(1, 0)
    # Canal 1: Obstáculos
    binary_map[..., 1] = obs_mask.float()
    
    binary_map[binary_map > 1] = 1
    
    if num_depots > 0:
        binary_map[..., 2] = ((xy_for_nodes - depot_ini[None, None, None, :]).norm(2, dim=-1) < node_size).sum(dim=0)
    if num_depots == 2:
        binary_map[..., 3] = ((xy_for_nodes - depot_end[None, None, None, :]).norm(2, dim=-1) < node_size).sum(dim=0)
    """
    binary_map = torch.zeros(image_size, image_size, 2 + num_depots)
    binary_map[..., 0] = ((xy[0, None] - nodes[..., None, None, :]).norm(2, dim=-1) < node_size).sum(dim=0).permute(1, 0)
    binary_map[..., 1] = obs_mask
    binary_map[binary_map > 1] = 1
    if num_depots > 0:
        binary_map[..., 2] = ((xy[0, None] - depot_ini[None, None, None, :]).norm(2, dim=-1) < node_size).sum(dim=0)
    if num_depots == 2:
        binary_map[..., 3] = ((xy[0, None] - depot_end[None, None, None, :]).norm(2, dim=-1) < node_size).sum(dim=0)
    """
    return nodes, depot_ini, depot_end, binary_map.permute(2, 0, 1)

#dac
def rotate_nodes(nodes: torch.Tensor, angles) -> torch.Tensor:
    """
    Versión híbrida: Acepta un solo grafo [N, 2] con un float,
    o un batch [B, N, 2] con un Tensor de ángulos.
    """
    # 1. Detectar si es un solo grafo o un batch
    is_single = nodes.dim() == 2
    if is_single:
        nodes = nodes.unsqueeze(0)  # Convertimos [N, 2] -> [1, N, 2]

    batch_size = nodes.shape[0]
    device = nodes.device

    # 2. Convertir 'angles' a tensor si es un float
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor([angles], device=device, dtype=nodes.dtype)
    else:
        angles = angles.to(device)

    # 3. Preparar ángulos para el cálculo [B, 1, 1]
    angles = angles.view(batch_size, 1, 1)
    
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    
    x = nodes[..., 0:1] - 0.5
    y = nodes[..., 1:2] - 0.5
    
    new_x = x * cos_a - y * sin_a + 0.5
    new_y = x * sin_a + y * cos_a + 0.5
    
    rotated = torch.cat([new_x, new_y], dim=-1)
    if nodes.shape[-1] > 2:
        rotated = torch.cat((rotated, nodes[..., 2:]), dim=-1)

    # 4. Devolver al formato original
    return rotated.squeeze(0) if is_single else rotated


# dac
def reflect_nodes(nodes: torch.Tensor, angles, shuffle=True) -> torch.Tensor:
    """
    Refleja los nodos y, por defecto, baraja su orden (shuffle) para evitar 'shortcuts'.
    """
    is_single = nodes.dim() == 2
    if is_single:
        nodes = nodes.unsqueeze(0)

    batch_size, n_nodes, dims = nodes.shape
    device = nodes.device

    # Convertir angles a tensor si es float
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor([angles], device=device, dtype=nodes.dtype)
    else:
        angles = angles.to(device)

    angles = angles.view(batch_size, 1, 1)
    
    # Ecuación de reflexión axial
    cos_2a = torch.cos(2 * angles)
    sin_2a = torch.sin(2 * angles)
    
    x = nodes[..., 0:1] - 0.5
    y = nodes[..., 1:2] - 0.5
    
    new_x = x * cos_2a + y * sin_2a + 0.5
    new_y = x * sin_2a - y * cos_2a + 0.5
    
    reflected = torch.cat([new_x, new_y], dim=-1)
    if dims > 2:
        reflected = torch.cat((reflected, nodes[..., 2:]), dim=-1)
        
    # --- LA CLAVE: BARAJADO CONDICIONAL ---
    if not shuffle:
        return reflected.squeeze(0) if is_single else reflected
    
    # Barajado aleatorio de los índices de los nodos
    shuffled = torch.zeros_like(reflected)
    for i in range(batch_size):
        perm = torch.randperm(n_nodes, device=device)
        shuffled[i] = reflected[i, perm]
        
    return shuffled.squeeze(0) if is_single else shuffled

def translate_nodes(nodes, angles, distances):
    """
    Desplaza todos los nodos en una dirección (angle) y distancia (distance).
    nodes: [B, N, 2] | angles: [B] en radianes | distances: [B]
    """
    device = nodes.device
    batch_size = nodes.shape[0]
    
    # Asegurar que sean tensores [B, 1, 1] para broadcasting
    angles = angles.to(device).view(batch_size, 1, 1)
    distances = distances.to(device).view(batch_size, 1, 1)
    
    # Calcular desplazamiento en X e Y
    dx = distances * torch.cos(angles)
    dy = distances * torch.sin(angles)
    
    # Aplicar el desplazamiento
    translated_nodes = nodes.clone()
    translated_nodes[..., 0:1] += dx
    translated_nodes[..., 1:2] += dy
    
    return translated_nodes