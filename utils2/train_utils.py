import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import Union, Dict, Any
from torch.utils.data import DataLoader
from typing import Union, Any, Optional

from utils2.model_utils import get_inner_model


def setup(rank, world_size):
    """
    Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    torch.distributed.destroy_process_group()


def move_to(var: Union[dict, torch.Tensor], device: Any) -> Union[torch.Tensor, dict]:
    """
    Move tensor or dictionary of tensors to specified device.

    Args:
        var (torch.Tensor or dict): Tensor or dictionary of tensors to move to the device.
        device (torch.device): Device to move the tensor(s) to.

    Returns:
        torch.Tensor or dict: Moved tensor or dictionary of moved tensors.
    """
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

#dac: modifico funcion para que a parted e bajar como a poco el lr, tb empieze con lr bajo los 1000 primeros pasos)
def load_lr_scheduler(optimizer: torch.optim.Optimizer, lr_decay: int = 1) -> torch.optim.lr_scheduler.LambdaLR:
    """Load the learning rate scheduler for the optimizer."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_decay ** epoch)

#dac: empezar con lr bajo las primeras epocas
def adjust_learning_rate_manual(optimizer, epoch, opts):
    """
    Ajusta el LR manualmente:
    - Épocas 0-4: Warmup (sube desde un valor bajo hasta el del terminal).
    - Épocas 5-50: Valor del terminal (con el decaimiento que elijas).
    """
    warmup_epochs = 5
    
    if epoch < warmup_epochs:
        # RAMP UP: Empezamos en el 20% y llegamos al 100% en la época 5
        # Época 0: 0.2 * LR | Época 4: 1.0 * LR
        factor = 0.2 + 0.8 * (epoch / warmup_epochs)
        lr = opts.lr_model * factor
    else:
        # POST-WARMUP: Usamos el LR del terminal con un decaimiento muy suave
        # para que no se muera en la época 10.
        lr = opts.lr_model #* (opts.lr_decay ** (epoch - warmup_epochs))

    # Aplicamos el nuevo LR al optimizador
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr


def load_optimizer(opts: argparse.Namespace, model: torch.nn.Module, baseline: Any, checkpoint: dict = None) -> torch.optim.Optimizer:
    """
    Load the optimizer.

    Args:
        opts (argparse.Namespace): Options for the training.
        model (torch.nn.Module): The model.
        baseline: Baseline.
        checkpoint: Data to load.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """

    # Load optimizer
    if baseline is None:
        optimizer = torch.optim.Adam(
            [{'params': model.parameters(), 'lr': opts.lr}]
        )
    else:
        optimizer = torch.optim.Adam(
            [{'params': model.parameters(), 'lr': opts.lr}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )

    # Load optimizer state
    if checkpoint is not None:
        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"[*] Optimizer state loaded from {opts.load_path}")
            except:
                print("[*] Optimizer state not loaded")
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)
    return optimizer


def clip_grad_norms(param_groups: list, max_norm: float = np.inf):
    """
    Clip gradients to a maximum norm.

    Args:
        param_groups (list): Parameter groups.
        max_norm (float): Maximum norm for clipping.

    Returns:
        tuple: Grad norms before and after clipping.
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else np.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

#dac
def contrastive_loss(graph_embedding, rotated_embedding, *args, **kwargs):
    # https://github.com/openai/CLIP/blob/main/clip/model.py | https://github.com/openai/CLIP/issues/83
    
    # Average embeddings across nodes/patches to get 2-dimensional embeddings: (batch_size, hidden_dim)
    graph_emb = graph_embedding.mean(dim=1)
    rotated_emb = rotated_embedding.mean(dim=1)
    
    # Normalize embeddings
    graph_emb = torch.nn.functional.normalize(graph_emb, dim=1)
    rotated_emb = torch.nn.functional.normalize(rotated_emb, dim=1)
    
    # Cosine similarity as logits
    logits = graph_emb @ rotated_emb.t()
    #logits_per_graph = graph_embedding @ image_embedding.t()
    #logits_per_image = logits_per_graph.t()
    
    # Define ground truth for CrossEntropyLoss
    labels = torch.arange(logits.shape[0]).to(logits.device)
    #ground_truth = torch.arange(logits_per_graph.shape[0]).to(device=logits_per_graph.device)
    
    # Calculate CrossEntropyLoss
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    #loss_graph = torch.nn.CrossEntropyLoss()(logits_per_graph, ground_truth)
    #loss_image = torch.nn.CrossEntropyLoss()(logits_per_image, ground_truth)
    return loss #(loss_graph + loss_image) / 2


def reinforce_loss(reward, log_prob, reward_bl=0, loss_bl=0, val=False, *args, **kwargs):
    if val:
        return reward.mean()
    return ((reward - reward_bl) * log_prob).mean() + loss_bl, reward


def validate(model: Any, dataloader: DataLoader, loss_func: Any, device: torch.types.Device, text: str = f"Validation") -> float:
    
    # Put model in eval mode
    model = set_train(model=model, train=False, return_actions=False)
    
    # Define progress bar
    pbar = tqdm(dataloader, desc=text)
    
    # Load batches on device
    loss = []
    for batch in pbar:
        batch = move_to(var=batch, device=device)
        
        # Make predictions
        predictions = model(batch)
        
        # Calculate validation loss
        _loss = loss_func(*predictions[:2], val=True)
        if isinstance(_loss, tuple):
            _loss = _loss[0]
        loss.append(_loss.data.cpu().item())
        
        # Update progress bar
        pbar.set_description(f"Validation: loss = {torch.tensor(loss).mean().item():.4f}")
    
    # Put model in train mode
    model = set_train(model=model, train=True, return_actions=False)
    
    # Return validation loss
    return torch.tensor(loss)


def set_train(model, train=False, return_actions=True, *args, **kwargs):
    model.train() if train else model.eval()
    get_inner_model(model).set_train(train=train, return_actions=return_actions)
    return model

def set_rng_state(checkpoint, use_cuda=False):
    """Set rng state (for reproducibility)."""
    if checkpoint is not None:
        torch.set_rng_state(checkpoint['rng_state'])
        if use_cuda:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        
def set_dataparallel(model, device, use_cuda, use_distributed):
    """Parallel training (if possible)"""
    if use_cuda:
        if use_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])  # , find_unused_parameters=True
        elif torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    return model
