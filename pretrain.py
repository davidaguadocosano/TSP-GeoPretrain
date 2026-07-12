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
    init_embed.train() 
    encoder.train()
    projector.train()
    
    total_loss = 0
    step = epoch * (opts.epoch_size // opts.batch_size)

    # --- Configuración dinámica del modo Hybrid ---
    # Asignamos un bit a cada transformación: rot=1, sym=2, trans=4
    bit_map = {'rot': 1, 'sym': 2, 'trans': 4}
    allowed_mask = 0
    for t in opts.hybrid_transformations:
        allowed_mask |= bit_map[t]
    
    # Creamos una lista de códigos (1 al 7) que solo usan los bits permitidos
    # Ej: si el usuario elige ['rot', 'trans'], allowed_codes será [1, 4, 5]
    allowed_codes = [i for i in range(1, 8) if (i & allowed_mask) == i]
    allowed_codes_tensor = torch.tensor(allowed_codes)
    # -----------------------------------------------

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_id, batch in enumerate(pbar):
        curr_batch_size = batch['nodes_v1'].size(0)

        # Generamos parámetros aleatorios para todas las posibles transformaciones
        angles_rot = torch.rand(curr_batch_size) * 2 * np.pi
        angles_sym = torch.rand(curr_batch_size) * np.pi
        angles_trans = torch.rand(curr_batch_size) * 2 * np.pi
        dist_trans = torch.rand(curr_batch_size) * 0.15 

        # --- Lógica Híbrida (Selección de transformaciones por bits) ---
        if opts.pretrain_type == 'hybrid':
            # Elegimos aleatoriamente una de las combinaciones permitidas para cada grafo del batch
            idx_choice = torch.randint(0, len(allowed_codes), (curr_batch_size,))
            choice = allowed_codes_tensor[idx_choice]
            
            v2_nodes = batch['nodes_v1'].clone()
            
            # Aplicamos máscaras basadas en los bits activos en 'choice'
            mask_rot = (choice & 1) > 0   # Bit 0 activo
            mask_sym = (choice & 2) > 0   # Bit 1 activo
            mask_trans = (choice & 4) > 0 # Bit 2 activo
            
            if mask_rot.any():
                v2_nodes[mask_rot] = data_utils.rotate_nodes(v2_nodes[mask_rot], angles_rot[mask_rot])
            
            if mask_sym.any():
                # Shuffle=True para el entrenamiento
                v2_nodes[mask_sym] = data_utils.reflect_nodes(v2_nodes[mask_sym], angles_sym[mask_sym], shuffle=True)
            
            if mask_trans.any():
                v2_nodes[mask_trans] = data_utils.translate_nodes(v2_nodes[mask_trans], angles_trans[mask_trans], dist_trans[mask_trans])
            
            batch['nodes_v2'] = v2_nodes
            batch['graph_v2'] = batch['graph_v1']

        # --- Lógica para tipos individuales (por si quieres usarlos por separado) ---
        elif opts.pretrain_type == 'translation':
            batch['nodes_v2'] = data_utils.translate_nodes(batch['nodes_v1'], angles_trans, dist_trans)
            batch['graph_v2'] = batch['graph_v1']
        
        elif opts.pretrain_type == 'rotation':
            batch['nodes_v2'] = data_utils.rotate_nodes(batch['nodes_v1'], angles_rot)
            batch['graph_v2'] = batch['graph_v1']

        elif opts.pretrain_type == 'symmetry':
            batch['nodes_v2'] = data_utils.reflect_nodes(batch['nodes_v1'], angles_sym, shuffle=True)
            batch['graph_v2'] = batch['graph_v1']

        # --- BLOQUE DE VISUALIZACIÓN DE DEBUG ---
        if epoch == 0 and batch_id == 0:
            if opts.pretrain_type == 'hybrid':
                combos = {
                    1: "SOLO_ROT", 2: "SOLO_SYM", 3: "ROT+SYM",
                    4: "SOLO_TRANS", 5: "ROT+TRANS", 6: "SYM+TRANS",
                    7: "ROT+SYM+TRANS"
                }

                # Solo intentamos guardar fotos de los códigos que están en allowed_codes
                for code in allowed_codes:
                    name = combos[code]
                    idx = (choice == code).nonzero(as_tuple=True)[0]
                    
                    if len(idx) > 0:
                        i = idx[0].item()
                        v2_debug = batch['nodes_v1'][i:i+1].clone()
                        
                        if code & 1: v2_debug = data_utils.rotate_nodes(v2_debug, angles_rot[i:i+1])
                        if code & 2: v2_debug = data_utils.reflect_nodes(v2_debug, angles_sym[i:i+1], shuffle=False)
                        if code & 4: v2_debug = data_utils.translate_nodes(v2_debug, angles_trans[i:i+1], dist_trans[i:i+1])
                        
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