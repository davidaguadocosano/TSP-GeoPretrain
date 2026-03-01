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

from utils2.train_utils import set_dataparallel, setup, cleanup

def save_rotation_check(nodes_v1, nodes_v2, epoch, save_dir):
    """
    Guarda una imagen con el grafo original y el rotado lado a lado.
    """
    # Tomamos solo la primera muestra del batch y la movemos a CPU
    v1 = nodes_v1[0].cpu().numpy()
    v2 = nodes_v2[0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
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

def train_epoch(init_embed,encoder, projector, optimizer, dataloader, epoch, tb_logger, opts):
    init_embed.train() #nuevo
    encoder.train()
    projector.train()
    
    total_loss = 0
    step = epoch * (opts.epoch_size // opts.batch_size)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_id, batch in enumerate(pbar):
        
        if epoch == 0 and batch_id == 0:
            save_rotation_check(batch['nodes_v1'], batch['nodes_v2'], epoch, opts.save_dir)

        # Mover datos a GPU
        v1_nodes = move_to(batch['nodes_v1'], opts.device)
        v1_graph = move_to(batch['graph_v1'], opts.device)
        v2_nodes = move_to(batch['nodes_v2'], opts.device)
        v2_graph = move_to(batch['graph_v2'], opts.device)

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

        total_loss += loss.item()
        
        # Log en TensorBoard
        if tb_logger is not None and step % opts.log_step == 0:
            tb_logger.log_value('pretrain/loss', loss.item(), step)
        
        step += 1
        pbar.set_postfix(loss=loss.item())

def run(opts):
    # Setup inicial idéntico a run.py
    torch.manual_seed(opts.seed)
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

    # Optimizer
    optimizer = torch.optim.Adam(
        list(init_embed.parameters()) + list(encoder.parameters()) + list(projector.parameters()), 
        lr=opts.lr_model
    )

    # 3. Cargar Dataset (Usa la nueva clase que añadimos)
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
        train_epoch(init_embed, encoder, projector, optimizer, dataloader, epoch, tb_logger, opts)
        
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