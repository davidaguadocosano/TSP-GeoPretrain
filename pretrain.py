import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger

# Importaciones del proyecto original
from options import get_options
from problems.tsp.problem_tsp import TSPPretrainDataset
from nets.encoders.gnn_encoder import GNNEncoder
from nets.encoders.gat_encoder import GraphAttentionEncoder
from nets.encoders.mlp_encoder import MLPEncoder
from utils import move_to, torch_load_cpu
from utils.data_utils import BatchedRandomSampler

from utils2.train_utils import set_dataparallel, setup, cleanup, load_lr_scheduler
from utils2.plot_utils import save_transformation_check
import utils2.data_utils as data_utils

"""
def save_rotation_check(nodes_v1, nodes_v2, epoch, save_dir):
    
    # Guarda una imagen con el grafo original y el rotado lado a lado.
    
    # Tomamos solo la primera muestra del batch y la movemos a CPU
    v1 = nodes_v1[0].cpu().numpy()
    v2 = nodes_v2[0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Título dinámico
    title_v2 = "Rotado (v2)" if pretrain_type == 'rotation' else "Reflejado (v2)"
    
    for i, (coords, title) in enumerate([(v1, "Original (v1)"), (v2, title_v2)]):
        axes[i].scatter(coords[:, 0], coords[:, 1], c='red', s=30)
        axes[i].plot(coords[:, 0], coords[:, 1], c='blue', alpha=0.3)
        axes[i].set_title(title)
        axes[i].set_xlim(-0.1, 1.1)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].set_aspect('equal')

    plt.suptitle(f"Chequeo {pretrain_type.capitalize()} - Época {epoch}")
    
    filename = f"{pretrain_type}_check_epoch_{epoch}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Imagen de control guardada en: {save_path}")
    """
"""
    # Dibujamos los puntos y conectamos en orden (para ver mejor la rotación)
    for i, (coords, title) in enumerate([(v1, "Original (v1)"), (v2, "Rotado (v2)")]):
        axes[i].scatter(coords[:, 0], coords[:, 1], c='red', s=30)
        # Dibujamos líneas siguiendo el índice para que se aprecie el "giro" de la estructura
        axes[i].plot(coords[:, 0], coords[:, 1], c='blue', alpha=0.3)
        axes[i].set_title(title)
        axes[i].set_xlim(-0.1, 1.1) # Un poco de margen para ver bien los bordes
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].set_aspect('equal')

    plt.suptitle(f"Chequeo de Rotación - Época {epoch}")
    
    save_path = os.path.join(save_dir, f"rotation_check_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Imagen de control guardada en: {save_path}")
    """

class ProjectionHead(nn.Module):
    """Cabeza de proyección MLP para aprendizaje contrastivo (estilo SimCLR)"""
    def __init__(self, dim_in, dim_out):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_out)
        )

    def forward(self, x):
        return self.mlp(x)

def info_nce_loss(z1, z2, temp=0.07):
    """Cálculo de la pérdida InfoNCE entre dos vistas del mismo grafo"""
    batch_size = z1.shape[0]
    
    # Normalizar los embeddings (esencial para el coseno)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Similitud entre todas las combinaciones (matriz de batch_size x batch_size)
    logits = torch.mm(z1, z2.t()) / temp
    
    # Las etiquetas positivas están en la diagonal (v1_i coincide con v2_i)
    labels = torch.arange(batch_size).to(z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def train_epoch(init_embed,encoder, projector, optimizer, dataloader, epoch, tb_logger, opts, lr_scheduler):
    init_embed.train() #nuevo
    encoder.train()
    projector.train()
    
    total_loss = 0
    step = epoch * (opts.epoch_size // opts.batch_size)
    # quito esto para que genere un angulo por grafo y no por epoch
    # angle_rad = torch.rand(1).to(opts.device) * np.pi
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_id, batch in enumerate(pbar):
        # Si es simetría, recalculamos v2_nodes antes de mover a GPU. 
        #angle_rad = torch.rand(1) * 2 * np.pi

        #dac: añadir ambos pretrains
        curr_batch_size = batch['nodes_v1'].size(0)
        angles_rot = torch.rand(curr_batch_size) * 2 * np.pi
        angles_sym = torch.rand(curr_batch_size) * np.pi
        angles_trans = torch.rand(curr_batch_size) * 2 * np.pi
        dist_trans = torch.rand(curr_batch_size) * 0.15 

        if opts.pretrain_type == 'rot+sym+trans':
            # 1:R, 2:S, 3:R+S, 4:T, 5:R+T, 6:S+T, 7:R+S+T
            choice = torch.randint(1, 8, (curr_batch_size,))
            v2_nodes = batch['nodes_v1'].clone()
            
            # Generamos máscaras basadas en los "bits" de la elección
            mask_rot = (choice & 1) > 0   # Si el bit 0 está activo
            mask_sym = (choice & 2) > 0   # Si el bit 1 está activo
            mask_trans = (choice & 4) > 0 # Si el bit 2 está activo
            
            # Aplicamos en orden: Rotación -> Simetría -> Traslación
            if mask_rot.any():
                v2_nodes[mask_rot] = data_utils.rotate_nodes(v2_nodes[mask_rot], angles_rot[mask_rot])
            
            if mask_sym.any():
                # Shuffle=True para el entrenamiento real
                v2_nodes[mask_sym] = data_utils.reflect_nodes(v2_nodes[mask_sym], angles_sym[mask_sym], shuffle=True)
            
            if mask_trans.any():
                v2_nodes[mask_trans] = data_utils.translate_nodes(v2_nodes[mask_trans], angles_trans[mask_trans], dist_trans[mask_trans])
            
            batch['nodes_v2'] = v2_nodes
            batch['graph_v2'] = batch['graph_v1']
            batch['nodes_v2'] = v2_nodes
            batch['graph_v2'] = batch['graph_v1']

        # --- Lógica de Traslación ---
        if opts.pretrain_type == 'translation':
            curr_batch_size = batch['nodes_v1'].size(0)
            # Ángulo aleatorio 0-360º
            angles_trans = torch.rand(curr_batch_size) * 2 * np.pi
            # Distancia corta (máximo 0.15) para no "perder" el grafo
            dist_trans = torch.rand(curr_batch_size) * 0.15 
            
            batch['nodes_v2'] = data_utils.translate_nodes(batch['nodes_v1'], angles_trans, dist_trans)
            batch['graph_v2'] = batch['graph_v1']

        # --- BLOQUE DE VISUALIZACIÓN DE DEBUG ---
        if epoch == 0 and batch_id == 0:
            if opts.pretrain_type == 'rot+sym+trans':
                # Definimos los nombres de las 7 combinaciones posibles
                combos = {
                    1: "SOLO_ROT", 2: "SOLO_SYM", 3: "ROT+SYM",
                    4: "SOLO_TRANS", 5: "ROT+TRANS", 6: "SYM+TRANS",
                    7: "ROT+SYM+TRANS"
                }

                for code, name in combos.items():
                    # Buscar el primer índice en el batch que tenga esta combinación exacta
                    idx = (choice == code).nonzero(as_tuple=True)[0]
                    
                    if len(idx) > 0:
                        i = idx[0].item()
                        # Re-calculamos v2 para debug (SIN SHUFFLE) para que el ojo humano lo entienda
                        v2_debug = batch['nodes_v1'][i:i+1].clone()
                        
                        if code & 1: # Aplicar Rotación
                            v2_debug = data_utils.rotate_nodes(v2_debug, angles_rot[i:i+1])
                        if code & 2: # Aplicar Simetría (sin shuffle para la foto)
                            v2_debug = data_utils.reflect_nodes(v2_debug, angles_sym[i:i+1], shuffle=False)
                        if code & 4: # Aplicar Traslación
                            v2_debug = data_utils.translate_nodes(v2_debug, angles_trans[i:i+1], dist_trans[i:i+1])
                        
                        # Guardamos la imagen con el eje si hay simetría implicada
                        eje = angles_sym[i].item() if (code & 2) else None
                        save_transformation_check(
                            batch['nodes_v1'][i], v2_debug[0], 
                            opts.save_dir, name, epoch, angle=eje
                        )
            
            if opts.pretrain_type == 'translation':
                save_transformation_check(
                    batch['nodes_v1'][0], 
                    batch['nodes_v2'][0], 
                    opts.save_dir, "TRANSLATION", epoch
                )
            
            else:
                # Caso cuando NO es rot+sym (es rotation o symmetry puro)
                a = None
                if opts.pretrain_type == 'symmetry':
                    # Para symmetry puro, generamos uno sin shuffle para la foto
                    a = angles_sym[0].item()
                    v2_img = data_utils.reflect_nodes(batch['nodes_v1'][0:1], angles_sym[0], shuffle=False)[0]
                else:
                    v2_img = batch['nodes_v2'][0]
                
                save_transformation_check(
                    batch['nodes_v1'][0], v2_img, 
                    opts.save_dir, opts.pretrain_type, epoch, angle=a
                )


        # Mover datos a GPU
        v1_nodes = move_to(batch['nodes_v1'], opts.device)
        v1_graph = move_to(batch['graph_v1'], opts.device)
        v2_nodes = move_to(batch['nodes_v2'], opts.device)
        v2_graph = move_to(batch['graph_v2'], opts.device)
        #por defecto nodes_v1 y nodes_v2 son el grafo normal y el rotado. ya que la rotación en su momento
        #la hice en problems/tsp/problem_tsp.py. un lio, pero funciona

        optimizer.zero_grad()

        # Proyectar de 2D a 128D antes de la GNN
        v1_embedded = init_embed(v1_nodes)
        v2_embedded = init_embed(v2_nodes)

        # Pasar ambas vistas por el encoder y el proyector
        # Nota: GNNEncoder espera (nodos, grafo)
        # Usamos .mean(1) para obtener un embedding global del grafo a partir de los nodos
        h1 = encoder(v1_embedded, v1_graph).mean(1) 
        h2 = encoder(v2_embedded, v2_graph).mean(1)

        z1 = projector(h1)
        z2 = projector(h2)

        # Calcular pérdida contrastiva
        loss = info_nce_loss(z1, z2, temp=opts.cl_temp)
        
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), opts.max_grad_norm)
        optimizer.step()

        # dac: ACTIVAR EL SCHEDULER EN CADA BATCH
        lr_scheduler.step()
        # dac: ver el LR actual en la barra de progreso
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

        total_loss += loss.item()
        
        # Log en TensorBoard
        if tb_logger is not None and step % opts.log_step == 0:
            tb_logger.log_value('pretrain/loss', loss.item(), step)
        
        step += 1
        pbar.set_postfix(loss=loss.item())

def run(opts):
    # Setup inicial idéntico a run.py
    #torch.manual_seed(opts.seed)
    # Si la semilla es None o menor que 0, usamos el reloj del sistema
    if opts.seed <= 0:
        import time
        # El operador & 0xFFFFFFFF asegura que el número esté en el rango [0, 2**32-1]
        seed = int(time.time() * 1000) & 0xFFFFFFFF
        print(f"[*] Usando semilla dinámica del sistema: {seed}")
    else:
        seed = opts.seed
        print(f"[*] Usando semilla fija: {seed}")

    # Aplicamos la semilla a todas las librerías para consistencia
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ----------------------------------
    
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    # Logger
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, f"pretrain_{opts.run_name}"))

    # 0. Capa de proyección inicial (De 2D a embedding_dim)
    init_embed = nn.Linear(2, opts.embedding_dim).to(opts.device)

    # 1. Instanciar Encoder
    encoder_class = {
        'gnn': GNNEncoder,
        'gat': GraphAttentionEncoder,
        'mlp': MLPEncoder
    }.get(opts.encoder)
    
    encoder = encoder_class(
        n_layers=opts.n_encode_layers,
        hidden_dim=opts.embedding_dim,
        aggregation=opts.aggregation,
        norm=opts.normalization,
        learn_norm=opts.learn_norm,
        track_norm=opts.track_norm,
        gated=opts.gated,
        n_heads=opts.n_heads
    ).to(opts.device)

    # 2. Instanciar Cabeza de Proyección
    projector = ProjectionHead(opts.embedding_dim, opts.cl_projector_dim).to(opts.device)

    #dac: Cargar pesos pre-entrenados para pre-entrenamiento continuo
    if opts.load_path is not None:
        print(f'[*] Cargando pesos pre-entrenados para pre-entrenamiento continuo desde: {opts.load_path}')
        # Cargamos el archivo (mapeando a la CPU/GPU correcta)
        checkpoint = torch.load(opts.load_path, map_location=lambda storage, loc: storage)
        
        # Cargamos los estados en nuestros modelos actuales
        init_embed.load_state_dict(checkpoint['init_embed'])
        encoder.load_state_dict(checkpoint['encoder'])
        
        # Si el checkpoint guardó el proyector, también lo cargamos para no perder ese avance
        if 'projector' in checkpoint:
            projector.load_state_dict(checkpoint['projector'])
            print("[*] Proyector cargado desde el checkpoint.")
        
        print("[*] Pesos del Encoder e Init_Embed cargados")

    # Optimizer
    optimizer = torch.optim.Adam(
        list(init_embed.parameters()) + list(encoder.parameters()) + list(projector.parameters()), 
        lr=opts.lr_model
    )
    # 2. dac: INICIALIZAR EL SCHEDULER AQUÍ
    #lr_scheduler = load_lr_scheduler(optimizer, opts)
    lr_scheduler = load_lr_scheduler(
        optimizer=optimizer,
        lr_decay=opts.lr_decay
    )

    # 3. Cargar Dataset (Usa la nueva clase que añadi)
    dataset = TSPPretrainDataset(
        min_size=opts.min_size,
        max_size=opts.max_size,
        num_samples=opts.epoch_size,
        batch_size=opts.batch_size,
        neighbors=opts.neighbors,
        knn_strat=opts.knn_strat
    )
    # se generan grafos de diversos tamaños. antes al barajarlos, pytorch no los podia apilar
    #al ser tensores de distntos tamaños. asi que se barajean dentro de los lotes de cada tamaño
    sampler = BatchedRandomSampler(dataset, opts.batch_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=opts.batch_size, 
        sampler=sampler, 
        num_workers=opts.num_workers
    )

    # Bucle de entrenamiento
    for epoch in range(opts.n_epochs):
        train_epoch(init_embed, encoder, projector, optimizer, dataloader, epoch, tb_logger, opts, lr_scheduler) #dac: añadi lr_scheduler
        
        # Guardar Checkpoint del Encoder
        checkpoint_path = os.path.join(opts.save_dir, f'encoder-epoch-{epoch}.pt')
        torch.save({
            'init_embed': init_embed.state_dict(), 
            'encoder': encoder.state_dict(),
            'opts': vars(opts)
        }, checkpoint_path)

if __name__ == "__main__":
    opts = get_options()
    opts.pretrain = True # Aseguramos modo pretrain
    run(opts)