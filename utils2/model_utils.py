import os
import json
import torch
from typing import Any, Tuple


def get_inner_model(model: torch.nn.Module | torch.nn.DataParallel | torch.nn.parallel.DistributedDataParallel) -> torch.nn.Module:
    """
    Get the inner model from a DataParallel wrapper.

    Args:
        model (torch.nn.Module or torch.nn.DataParallel or torch.nn.parallel.DistributedDataParallel): The model to extract the inner model from.

    Returns:
        torch.nn.Module: The inner model.
    """
    is_dp = isinstance(model, torch.nn.DataParallel)
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    return model.module if is_dp or is_ddp else model


def get_module(module_name, module_dict, module_type, module_path: int = '', *args, **kwargs):
    
    # Load module from file
    if os.path.exists(module_path):
        module = load_module(
            module_path=module_path,
            module_dict=module_dict,
            module_type=module_type,
            *args, **kwargs
        )
    else:
        module = None
    
    # Initialize new module
    if module is None:
        module = module_dict[module_name](*args, **kwargs)
    return module
    

def load_module(module_path, module_dict, module_type, *args, **kwargs):
    
    # Load module's checkpoint
    checkpoint, module_args, _ = load_checkpoint(module_path, module_name=module_type)
    
    # Initialize module from arguments
    if module_type in module_args:
        if module_args[module_type] in module_dict:
            module = module_dict[module_args[module_type]](*args, **(module_args | kwargs))
            
            # Get checkpoint data the module
            module_checkpoint = checkpoint.get(module_type, {})
            
            # Overwrite module state dict and return module
            module.load_state_dict(module.state_dict() | module_checkpoint)  # Avoids loading error
            # module.load_state_dict(module_checkpoint)                      # May cause error if different weights are loaded
            return module
    return None


def save_checkpoint(
        model: torch.nn.Module | torch.nn.DataParallel,
        optimizer: torch.optim.Optimizer,
        baseline: Any,
        save_dir: str,
        epoch: int) -> None:
    """
    Saves the model, optimizer state, and other information to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        baseline: The baseline.
        save_dir (str): The directory to save the model to.
        epoch (int): The current epoch.

    Returns:
        None
    """
    
    # Get inner module (for data parallel)
    model = get_inner_model(model)
    
    # Save mandatory info
    if baseline is None:
        save_info = {
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
        }
    else:
        save_info = {
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'baseline': baseline.state_dict(),
        }
    
    # Save graph encoder if exists
    if 'graph_encoder' in model._modules['encoder']._modules:
        save_info['graph_encoder'] = model._modules['encoder']._modules['graph_encoder'].state_dict()
        
    # Save image encoder if exists
    if 'image_encoder' in model._modules['encoder']._modules:
        save_info['image_encoder'] = model._modules['encoder']._modules['image_encoder'].state_dict()
    
    # Save decoder if exists
    if 'decoder' in model._modules:
        save_info['decoder'] = model._modules['decoder']._modules['decoder'].state_dict()
    
    # Save info
    path = os.path.join(save_dir, f"epoch-{str(epoch).zfill(3)}.pt")
    torch.save(save_info, path)
    print(f"[*] Model and state saved in {path}")
    

def load_checkpoint(load_path, opts=None, module_name='model'):
    
    if os.path.exists(load_path):
    
        # Identify checkpoint (if any)
        cp_filename, cp_dirname, first_epoch = get_checkpoint(path=load_path)
        
        # Load checkpoint data
        checkpoint = load_checkpoint_data(filename=cp_filename)
        print(f"[*] Checkpoint data loaded for {module_name} from: {cp_filename}")
        
        # Load args (either from checkpoint or from opts)
        args = load_args(dirname=cp_dirname, opts=opts)
        
        # Return checkpoint data, args, and first epoch
        return checkpoint, args, first_epoch
    return None, vars(opts), 0


def load_checkpoint_data(filename: str) -> Any:
    """
    Loads data from a file while ensuring it is loaded onto the CPU.

    Args:
        load_path (str): The path to the file containing the data.

    Returns:
        Any: The loaded data.
    """
    return torch.load(filename, map_location=lambda storage, loc: storage)

"""
def get_checkpoint(path: str, epoch: int | None = None) -> Tuple[str, str, int]:
    #esto se comentaba con 3 comas
    Determines whether the provided path is a file or directory and returns the appropriate model filename. If it is a
    directory, it returns the latest saved model filename when there are more than one in the directory.

    Args:
        path (str): The path to the file or directory containing the saved model(s).
        epoch (int, optional): The epoch to load from if `path` is a directory. Defaults to None.

    Returns:
        tuple: A tuple containing the model filename, its directory, and the next epoch.
    #esto se comentaba con 3 comas

    # Path indicates the saved checkpoint
    if os.path.isfile(path):
        checkpoint_filename = path

    # Path indicates the directory where checkpoints are saved
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        checkpoint_filename = os.path.join(path, f"epoch-{str(epoch).zfill(3)}.pt")
    
    # Path is not valid
    else:
        return "", "", 0
    
    # Get checkpoint dirname from checkpoint filename
    checkpoint_dirname = os.path.dirname(checkpoint_filename)
    
    # Return checkpoint filename, checkpoint dirname, and next epoch
    return checkpoint_filename, checkpoint_dirname, epoch + 1
"""
def get_checkpoint(path: str, epoch: int | None = None) -> Tuple[str, str, int]:
    """
    Determina si la ruta es un archivo o directorio y extrae el nombre del modelo y la siguiente época.
    """
    # CASO 1: La ruta es un archivo directo
    if os.path.isfile(path):
        checkpoint_filename = path
        # Intentamos extraer el número de época del nombre del archivo (ej: epoch-000.pt)
        if epoch is None:
            try:
                # Extrae '000' de 'epoch-000.pt' y lo convierte a entero
                epoch = int(os.path.basename(path).split("-")[1].split(".")[0])
            except (IndexError, ValueError):
                epoch = 0 # Valor por defecto si el nombre no sigue el formato

    # CASO 2: La ruta es un directorio
    elif os.path.isdir(path):
        if epoch is None:
            # Busca el archivo con el número de época más alto
            files = [f for f in os.listdir(path) if f.endswith('.pt') and '-' in f]
            if not files:
                return "", "", 0
            epoch = max(int(os.path.splitext(f)[0].split("-")[1]) for f in files)
        
        checkpoint_filename = os.path.join(path, f"epoch-{str(epoch).zfill(3)}.pt")
    
    else:
        return "", "", 0
    
    checkpoint_dirname = os.path.dirname(checkpoint_filename)
    
    # Ahora epoch ya no es None, por lo que la suma funcionará
    return checkpoint_filename, checkpoint_dirname, epoch + 1


def load_args(dirname, opts=None) -> dict:
    
    # Get JSON filename from dirname
    filename = os.path.join(dirname, 'args.json')
    
    # Load arguments
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            args = json.load(f)
        
    # Arguments not found
    else:
        args = {}
        
    # Return combination of args and opts
    return args if opts is None else args | vars(opts)
