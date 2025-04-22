"""
Tests for the model loading functionality.
"""
import os
import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock, mock_open

import jax
import jax.numpy as jnp
import numpy as np
import torch

import equinox as eqx

from mlpox.networks import DeepMlp
from mlpox.load_models import (
    load_model, 
    convert_params_from_torch, 
    download, 
    stringify_name
)


def test_stringify_name():
    """Test the stringify_name function."""
    from jax.tree_util import GetAttrKey, SequenceKey
    
    # Create a simple path
    path = (GetAttrKey("layers"), SequenceKey(0), GetAttrKey("weight"))
    
    # Convert to string
    result = stringify_name(path)
    
    # Check result
    assert result == "layers.0.weight"


@patch("mlpox.load_models.urlretrieve")
def test_download(mock_urlretrieve):
    """Test the download function."""
    # Set up mock
    mock_urlretrieve.return_value = None
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_name = "B-6_Wi-512_res_64_imagenet21_epochs_800"
        weight_path, config_path = download(checkpoint_name, temp_dir)
        
        # Check paths
        assert weight_path == os.path.join(temp_dir, f"{checkpoint_name}_weights")
        assert config_path == os.path.join(temp_dir, f"{checkpoint_name}_config")
        
        # Check that urlretrieve was called twice (once for weights, once for config)
        assert mock_urlretrieve.call_count == 2


@patch("builtins.open", new_callable=mock_open, read_data="{}")
@patch("mlpox.load_models.torch.load")
def test_convert_params_from_torch(mock_torch_load, mock_file):
    """Test the convert_params_from_torch function."""
    # Create a small JAX model
    test_model = MagicMock()
    test_params = {
        "param1": jnp.zeros((5, 5)),
        "param2": jnp.ones((3, 3))
    }
    
    # Mock jax.tree_util.tree_flatten_with_path
    with patch("jax.tree_util.tree_flatten_with_path") as mock_flatten:
        from jax.tree_util import GetAttrKey
        # Mock the flatten operation
        mock_flatten.return_value = (
            [((GetAttrKey("param1"),), test_params["param1"]), 
             ((GetAttrKey("param2"),), test_params["param2"])],
            "pytree_def"  # This is a placeholder for the actual PyTree definition
        )
        
        # Mock tree_unflatten
        with patch("jax.tree_util.tree_unflatten") as mock_unflatten:
            mock_unflatten.return_value = test_params
            
            # Create mock torch params
            torch_params = {
                "param1": torch.zeros(5, 5),
                "param2": torch.ones(3, 3)
            }
            
            # Convert
            result = convert_params_from_torch(test_model, torch_params)
            
            # Check that the result is what we expect
            assert mock_unflatten.called
            assert isinstance(result, dict)


@pytest.mark.parametrize(
    "model_name,expected_blocks,expected_dim,expected_classes",
    [
        ("B_6-Wi_512_res_64_in21k", 6, 512, 11230),
        ("B_12-Wi_1024_res_64_in21k_cifar10", 12, 1024, 10),
        ("B_6-Wi_512_res_64_in21k_cifar100", 6, 512, 100),
        ("B_12-Wi_1024_res_64_in21k_imagenet", 12, 1024, 1000),
    ]
)
def test_model_name_parsing(model_name, expected_blocks, expected_dim, expected_classes):
    """Test that the model name parsing extracts the correct parameters."""
    # Parse model name
    parts = model_name.split('_')
    num_blocks = int(parts[1].split('-')[0])
    embed_dim = int(parts[2])
    img_size = int(parts[4])
    
    # Determine num_classes
    if 'cifar100' in model_name:
        num_classes = 100
    elif 'cifar10' in model_name:
        num_classes = 10
    elif 'imagenet' in model_name:
        num_classes = 1000
    else:  # Default to 'in21k' which is ImageNet-21K
        num_classes = 11230
    
    # Check parsed values
    assert num_blocks == expected_blocks
    assert embed_dim == expected_dim
    assert num_classes == expected_classes
    assert img_size == 64  # This is fixed in all the provided examples


@patch("mlpox.load_models.download")
@patch("mlpox.load_models.torch.load")
@patch("builtins.open", new_callable=MagicMock)
@patch("mlpox.load_models.convert_params_from_torch")
@patch("mlpox.load_models.eqx.partition")
@patch("mlpox.load_models.eqx.combine")
def test_load_model(mock_combine, mock_partition, mock_convert_params, 
                    mock_open, mock_torch_load, mock_download):
    """Test that load_model calls the expected functions with the right parameters."""
    # Set up mocks
    mock_download.return_value = ("weight_path", "config_path")
    
    # Mock file open and JSON load
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    mock_file.read.return_value = '{}'
    
    # Mock PyTorch load
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value = mock_tensor
    mock_torch_params = {"param1": mock_tensor, "param2": mock_tensor}
    mock_torch_load.return_value = mock_torch_params
    
    # Mock partition and combine
    mock_partition.return_value = ("dynamic", "static")
    mock_combine.return_value = "converted_model"
    
    # Mock convert_params_from_torch
    mock_convert_params.return_value = "converted_params"
    
    # Call the function
    model = load_model("B_6-Wi_512_res_64_in21k")
    
    # Verify the function calls
    mock_download.assert_called_once()
    mock_open.assert_called_once()
    mock_torch_load.assert_called_once()
    mock_partition.assert_called_once()
    mock_convert_params.assert_called_once()
    mock_combine.assert_called_once_with("converted_params", "static")
    
    # Check the result
    assert model == "converted_model"


@patch("mlpox.load_models.download")
def test_load_model_invalid_name(mock_download):
    """Test that load_model raises a ValueError for an invalid model name."""
    with pytest.raises(ValueError):
        load_model("invalid_model_name")
    
    # Verify download was not called
    mock_download.assert_not_called()


@pytest.mark.integration
def test_load_model_integration():
    """
    Integration test for load_model.
    
    This test actually downloads a model and loads it, so it's marked as an integration test
    and should be skipped during regular testing.
    """
    # Skip this test if not running integration tests
    pytest.importorskip("torch")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use a small model to minimize download time
        model = load_model("B_6-Wi_512_res_64_in21k_cifar10", checkpoint_path=temp_dir)
        
        # Check that we got a DeepMlp model
        assert isinstance(model, DeepMlp)
        
        # Check basic model structure
        assert len(model.layers) == 6
        assert model.linear_embed is not None
        assert model.fc is not None
        
        # Check that the model can process an input
        test_input = jnp.zeros((64, 64, 3))
        output = model(test_input)
        
        # Output should be a 10-dimensional vector for CIFAR-10
        assert output.shape == (10,)
