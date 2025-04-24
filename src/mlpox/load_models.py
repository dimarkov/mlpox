"""
Functions for loading pre-trained PyTorch models into Equinox models.
"""
import os
import json
import re
from typing import Dict, Optional, Tuple, Any, Set

import numpy as np
import torch
import progressbar
from urllib.request import urlretrieve

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.tree_util as jtu
from jax.tree_util import GetAttrKey, SequenceKey

import equinox as eqx

from mlpox.networks import DeepMlp

# Default model names and their corresponding checkpoints
default_checkpoints = {
    'B_12-Wi_1024_res_64_in21k': 'B-12_Wi-1024_res_64_imagenet21_epochs_800',
    'B_12-Wi_512_res_64_in21k': 'B-12_Wi-512_res_64_imagenet21_epochs_600',
    'B_6-Wi_1024_res_64_in21k': 'B-6_Wi-1024_res_64_imagenet21_epochs_800',
    'B_6-Wi_512_res_64_in21k': 'B-6_Wi-512_res_64_imagenet21_epochs_800',
    'B_12-Wi_1024_res_64_in21k_cifar10': 'B-12_Wi-1024_res_64_cifar10_epochs_20',
    'B_12-Wi_1024_res_64_in21k_cifar100': 'B-12_Wi-1024_res_64_cifar100_epochs_40',
    'B_12-Wi_1024_res_64_in21k_imagenet': 'B-12_Wi-1024_res_64_imagenet_epochs_50',
    'B_12-Wi_512_res_64_in21k_cifar10': 'B-12_Wi-512_res_64_cifar10_epochs_20',
    'B_12-Wi_512_res_64_in21k_cifar100': 'B-12_Wi-512_res_64_cifar100_epochs_20',
    'B_12-Wi_512_res_64_in21k_imagenet': 'B-12_Wi-512_res_64_imagenet_epochs_20',
    'B_6-Wi_512_res_64_in21k_cifar10': 'B-6_Wi-512_res_64_cifar10_epochs_20',
    'B_6-Wi_512_res_64_in21k_cifar100': 'B-6_Wi-512_res_64_cifar100_epochs_20',
    'B_6-Wi_512_res_64_in21k_imagenet': 'B-6_Wi-512_res_64_imagenet_epochs_20',
    'B_6-Wi_1024_res_64_in21k_cifar10': 'B-6_Wi-1024_res_64_cifar10_epochs_20',
    'B_6-Wi_1024_res_64_in21k_cifar100': 'B-6_Wi-1024_res_64_cifar100_epochs_20',
    'B_6-Wi_1024_res_64_in21k_imagenet': 'B-6_Wi-1024_res_64_imagenet_epochs_20'
}

# Weight URLs for each checkpoint
weight_urls = {
    'B-12_Wi-1024_res_64_imagenet21_epochs_800':
        'https://drive.usercontent.google.com/download?id=1rcV8RXij_kW9X2zSLNyNOTO_bKUjE0cJ&export=download&authuser=0&confirm=t&uuid=72ba7ef7-5c0e-43a8-8538-c78a7b6ae34c&at=APZUnTVYImDEDtOncjUjlRW2Fa-v%3A1718049334362',
    'B-12_Wi-512_res_64_imagenet21_epochs_600':
        'https://drive.usercontent.google.com/download?id=1sL9j_4FFeBTWTzuRFbHLfwrmPVLXhUtW&export=download&authuser=0&confirm=t&uuid=91299093-c2dc-4538-93fa-d34be798cedc&at=APZUnTUEplQUhKAe6zbjUuFWUGiV%3A1718049319992',
    'B-6_Wi-1024_res_64_imagenet21_epochs_800':
        'https://drive.usercontent.google.com/download?id=1cmO3QSz_hfHtyzkUOZnPmXPxIE2YzEbf&export=download&authuser=0&confirm=t&uuid=c102f0ec-18b9-496a-b615-819513501d65&at=APZUnTX6iqbWKmcVQzv4nf04efor%3A1718049304706',
    'B-6_Wi-512_res_64_imagenet21_epochs_800':
        'https://drive.usercontent.google.com/download?id=1QV3a99UT8llfh9zDDuNKWDH_5c6S_YT5&export=download&authuser=0&confirm=t&uuid=fa3e3e51-9eae-4f4c-9c88-f9882258160c&at=APZUnTWSzjI5fY70cc3I1t_E3nv1%3A1718049288621',
    'B-12_Wi-1024_res_64_cifar10_epochs_20':
        'https://drive.usercontent.google.com/download?id=1GyxuewoOzMRhzEOyUrIBLQzquc-QEYNV&export=download&authuser=0&confirm=t&uuid=02337a36-362b-41bc-8b66-c1e8737c6729&at=APZUnTV0pOpn9aeIkKng_OtiRw0l%3A1718049274446',
    'B-12_Wi-1024_res_64_cifar100_epochs_40':
        'https://drive.usercontent.google.com/download?id=1LNqC58cSwtuDr-C4bk1O3GA_vAWls-UH&export=download&authuser=0&confirm=t&uuid=37ee7032-ec34-4414-ac3b-fcb8a3f5e17d&at=APZUnTXsPCgnt2__IQ7fScHpXmcX%3A1718049262372',
    'B-12_Wi-1024_res_64_imagenet_epochs_50':
        'https://drive.usercontent.google.com/download?id=1MVebvnSGL02k_ql1gUCjh4quGqM9RM4F&export=download&authuser=0&confirm=t&uuid=4118f07e-ffdd-4b74-9b74-2508ffcc2b02&at=APZUnTWAAiNwrzrTzDm3Sl3MtzMF%3A1718049247748',
    'B-12_Wi-512_res_64_cifar10_epochs_20':   
        'https://drive.usercontent.google.com/download?id=1F1NvoOsYCgsn1GOZcwsoToOtsw-9Aw1v&export=download&authuser=0&confirm=t&uuid=899cb74b-2bce-4b51-81ab-2df63af3dcbe&at=APZUnTWAC-eENGH6rRWchnMHsSBm%3A1718049232656',
    'B-12_Wi-512_res_64_cifar100_epochs_20':   
        'https://drive.usercontent.google.com/download?id=1KIULehrqOyxIkZj0HiqowNmBy4Ye1EQ2&export=download&authuser=0&confirm=t&uuid=bf208699-bbf3-4ad3-9bb6-d61eec237265&at=APZUnTXwvPbxLngt1wCVCVNriiXA%3A1718049215282',
    'B-12_Wi-512_res_64_imagenet_epochs_20':   
        'https://drive.usercontent.google.com/download?id=1f0ZYzB_XujX8hDcEn_J6iWvw1meJ4Cbg&export=download&authuser=0&confirm=t&uuid=90d9b1d0-fd5f-468e-b637-53afa56d3f22&at=APZUnTXFslB7n2WqcFxsmxZzNqoB%3A1718049045796',
    'B-6_Wi-1024_res_64_cifar10_epochs_20':   
        'https://drive.usercontent.google.com/download?id=1Tyd5CkROPCMQybnrZ_o1wiAW7VoK6AfJ&export=download&authuser=0&confirm=t&uuid=7534d430-40ab-4475-a862-9413499b0f79&at=APZUnTXX2ioldX_5JCXDt0nP3pCu%3A1718049195583',
    'B-6_Wi-1024_res_64_cifar100_epochs_20':   
        'https://drive.usercontent.google.com/download?id=1FrRb78bjun6QGbbH-pCWDaaE_8LWW785&export=download&authuser=0&confirm=t&uuid=b2c5459c-ede5-4ba4-97dc-e7a247cfba6a&at=APZUnTWa7Uha96h-6FxJosR1b2F0%3A1718048945010',
    'B-6_Wi-1024_res_64_imagenet_epochs_20':   
        'https://drive.google.com/uc?id=115Lks211vx1at2dWn3JtQ57EZ7eNAVP4E&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar10_epochs_20':
        'https://drive.usercontent.google.com/download?id=1VjHgjheSm_w7xPtheEmY5kV_KE4-38zQ&export=download&authuser=0&confirm=t&uuid=71fbd376-79f4-43db-bd4a-381255571319&at=APZUnTUJN8LEutgI-L0oVjbG3df3%3A1718048695392',
    'B-6_Wi-512_res_64_cifar100_epochs_20':
        'https://drive.usercontent.google.com/download?id=1iK3t20-GS_Vs-_Q3ZexSiCfjGJ3IaPC2&export=download&authuser=0&confirm=t&uuid=0196232e-f83d-4c9d-921e-857db8848725&at=APZUnTV3aw0EOkJS4SEIo4XToVT4%3A1718048904050',
    'B-6_Wi-512_res_64_imagenet_epochs_20':
        'https://drive.usercontent.google.com/download?id=1iK3t20-GS_Vs-_Q3ZexSiCfjGJ3IaPC2&export=download&authuser=0&confirm=t&uuid=d8f548aa-ccd6-4f0a-be49-356e9ee2e243&at=APZUnTXb7Ss81nGKgrixYS0binTs%3A1718048598551'
}

# Config URLs for each checkpoint
config_urls = {
    'B-12_Wi-1024_res_64_imagenet21_epochs_800':
        'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_imagenet21_epochs_600':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_imagenet21_epochs_800':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_imagenet21_epochs_800':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_cifar100_epochs_20':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_cifar100_epochs_20':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar100_epochs_20':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
}

# Global progress bar for download tracking
pbar = None

def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    """
    Display download progress using progressbar.
    
    Args:
        block_num: Current block number being downloaded
        block_size: Size of each block in bytes
        total_size: Total file size in bytes
    """
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def download(name: str, checkpoint_path: str) -> Tuple[str, str]:
    """
    Download model weights and configuration if they don't exist locally.
    
    Args:
        name: Name of the checkpoint to download
        checkpoint_path: Directory path to store the downloaded files
        
    Returns:
        Tuple of (weight_path, config_path) with paths to the downloaded files
    """
    weight_url = weight_urls[name]
    config_url = config_urls[name]

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_path, exist_ok=True)
    
    weight_path = os.path.join(checkpoint_path, f"{name}_weights")
    config_path = os.path.join(checkpoint_path, f"{name}_config")
    
    weight_exists = os.path.isfile(weight_path)
    config_exists = os.path.isfile(config_path)

    if not weight_exists:
        print(f'Downloading weights for {name}...')
        urlretrieve(weight_url, weight_path, show_progress)
    else:
        print(f'Weights for {name} already downloaded')
        
    if not config_exists:
        print(f'Downloading config for {name}...')
        urlretrieve(config_url, config_path)
    else:
        print(f'Config for {name} already downloaded')

    return weight_path, config_path


def stringify_name(path: Tuple) -> str:
    """
    Convert a path tuple from JAX tree_flatten_with_path to a string.
    
    Args:
        path: Path tuple from JAX tree_flatten_with_path
        
    Returns:
        String representation of the path
    """
    stringified = []
    for p in path:
        if isinstance(p, GetAttrKey):
            stringified.append(p.name)
        if isinstance(p, SequenceKey):
            stringified.append(str(p.idx))
    return ".".join(stringified)


def convert_params_from_torch(
    jax_model: eqx.Module,
    torch_params: Dict[str, torch.Tensor],
    replace_cfg: Dict[str, str] = None,
    expand_cfg: Dict[str, list] = None,
    squeeze_cfg: Dict[str, Optional[int]] = None,
    whitelist: Set[str] = None,
) -> Dict:
    """
    Convert PyTorch model parameters to Equinox module parameters.
    
    Args:
        jax_model: Equinox module to load parameters into
        torch_params: Dictionary of PyTorch parameters
        replace_cfg: Dictionary mapping PyTorch parameter names to Equinox parameter names
        expand_cfg: Dictionary specifying which parameters need to be expanded
        squeeze_cfg: Dictionary specifying which parameters need to be squeezed
        whitelist: Set of parameters to exclude from format conversion
        
    Returns:
        Dictionary of converted parameters compatible with the Equinox module
    """
    # Use empty dictionaries as defaults if not provided
    replace_cfg = replace_cfg or {}
    expand_cfg = expand_cfg or {}
    squeeze_cfg = squeeze_cfg or {}
    whitelist = whitelist or set()
    
    # Extract the parameters from the defined Jax model
    jax_params = eqx.filter(jax_model, eqx.is_array)
    jax_params_flat, jax_param_pytree = jax.tree_util.tree_flatten_with_path(jax_params)
    
    # Default replacement configuration
    default_replace_cfg = {
        "layers": "blocks",
        "linear1": "block.0",
        'linear2': "block.2",
        'linear_embed': 'linear_in',
        'fc': 'linear_out'
    }
    
    # Merge with provided replace_cfg
    replace_cfg = {**default_replace_cfg, **(replace_cfg or {})}

    # Convert parameters one by one
    torch_params_flat = []
    remaining_params = dict(torch_params)  # Copy to track unused params
    
    for path, param in jax_params_flat:
        # Convert the path to a string
        param_path = stringify_name(path)
        
        # Map JAX parameter names to PyTorch parameter names
        param_path = re.sub(r"\.scale|.kernel", ".weight", param_path)
        
        # Apply replacements
        for old, new in replace_cfg.items():
            param_path = param_path.replace(old, new)
        
        # Special handling for normalization layers
        if 'norm.' in param_path:
            pp = param_path.split('.')
            param_path = f'layernorms.{pp[1]}.{pp[-1]}'
            
        # Skip if not in torch_params
        if param_path not in remaining_params:
            print(f"Warning: Parameter '{param_path}' not found in PyTorch model")
            # Use original parameter value from JAX model
            torch_params_flat.append(param)
            continue
        
        # Get the PyTorch parameter
        torch_param = remaining_params[param_path]
        
        # Print shape information for debugging
        # print(f"Converting {param_path}: PyTorch shape {torch_param.shape}, JAX shape {param.shape}")
        
        # Convert to JAX array
        torch_param_array = jnp.asarray(torch_param.numpy())
        
        # Add to the list of converted parameters
        torch_params_flat.append(torch_param_array)
        
        # Remove from remaining parameters
        remaining_params.pop(param_path)
    
    # Log any remaining parameters that weren't used
    if remaining_params:
        print(f"Warning: {len(remaining_params)} parameters from PyTorch model were not used:")
        for name in remaining_params.keys():
            print(f"  - {name}")
    
    # Reconstruct the JAX parameter tree with the converted parameters
    loaded_params = jax.tree_util.tree_unflatten(jax_param_pytree, torch_params_flat)
    
    return loaded_params


def load_model(
    model_name: str,
    checkpoint_path: str = '.checkpoints/',
    ) -> eqx.Module:
    """
    Load a pre-trained PyTorch model into an Equinox model.
    
    Args:
        model_name: Name of the model to load (key in default_checkpoints)
        checkpoint_path: Directory path to store/load checkpoint files
        
    Returns:
        Loaded Equinox model with converted parameters
    """
    if model_name not in default_checkpoints:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(default_checkpoints.keys())}")
    
    checkpoint_name = default_checkpoints[model_name]
    
    # Download weights and config
    weight_path, config_path = download(checkpoint_name, checkpoint_path)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract model parameters from the config
    # Parse model name to get key parameters
    parts = model_name.split('_')
    
    # Extract model configuration from name:
    # Format: B_{num_blocks}-Wi_{embed_dim}_res_{img_size}_{dataset}
    num_blocks = int(parts[1].split('-')[0])
    embed_dim = int(parts[2])
    img_size = int(parts[4])
    
    # Number of classes depends on the dataset
    if 'cifar100' in model_name:
        num_classes = 100
    elif 'cifar10' in model_name:
        num_classes = 10
    elif 'imagenet' in model_name:
        num_classes = 1000
    else:  # Default to 'in21k' which is ImageNet-21K
        num_classes = 11230
    
    # Load PyTorch weights
    torch_params = {
        k: v.cpu()
        for k, v in torch.load(weight_path, map_location=torch.device('cpu')).items()
    }
    
    # Create Equinox model
    eqx_model = DeepMlp(
        img_size=img_size,
        in_chans=3,  # RGB images
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_classes=num_classes,
        key=PRNGKey(0)
    )
    
    # Define parameter name replacement mapping
    replace_cfg = {
        "layers": "blocks",
        "linear1": "block.0",
        'linear2': "block.2",
        'linear_embed': 'linear_in',
        'fc': 'linear_out'
    }
    
    # Separate dynamic (trainable) and static parts of the model
    dynamic_model, static_model = eqx.partition(eqx_model, eqx.is_array)
    
    # Convert PyTorch parameters to JAX format
    converted_dynamic = convert_params_from_torch(
        dynamic_model,
        torch_params,
        replace_cfg=replace_cfg,
        expand_cfg=None,
        squeeze_cfg=None
    )
    
    # Combine converted parameters with static model structure
    converted_model = eqx.combine(converted_dynamic, static_model)
    
    return converted_model
