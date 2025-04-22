import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import jax.tree_util as jtu
from jax import grad, vmap
import pytest
import numpy as np

from mlpox.layers import (
    MlpBlock,
    MixerBlock,
    StandardMlpBlock,
    BottleneckMlpBlock
)

@pytest.fixture
def key():
    return jr.PRNGKey(0)

def test_mlp_block(key):
    block = MlpBlock(10, 20, 30, jnn.relu, key=key)
    x = jnp.ones(10)
    y = block(x)
    assert y.shape == (30,)

def test_mixer_block(key):
    block = MixerBlock(
        tokens_dim=8,
        embed_dim=16,
        tokens_hidden_dim=32,
        embed_hidden_dim=64,
        activation=jnn.gelu,
        key=key
    )
    x = jnp.ones((8, 16))  # (tokens, embed_dim)
    y = block(x)
    assert y.shape == (8, 16)

def test_standard_mlp_block(key):
    block = StandardMlpBlock(10, 20, jnn.relu, key=key)
    x = jnp.ones(10)
    y = block(x)
    assert y.shape == (20,)

def test_bottleneck_mlp_block(key):
    block = BottleneckMlpBlock(10, jnn.relu, ratio=4, key=key)
    x = jnp.ones(10)
    y = block(x)
    assert y.shape == (10,)  # Preserves input dimension due to residual connection

def test_mlp_block_numerical_accuracy(key):
    block = MlpBlock(2, 4, 1, jnn.relu, key=key)
    x = jnp.array([1.0, 2.0])
    y = block(x)
    # Test if output is deterministic
    y2 = block(x)
    np.testing.assert_allclose(y, y2)

def test_mixer_block_edge_cases(key):
    block = MixerBlock(4, 8, 16, 32, jnn.gelu, key=key)
    # Test with zero input
    x_zeros = jnp.zeros((4, 8))
    y_zeros = block(x_zeros)
    assert y_zeros.shape == (4, 8)
    # Test with very large values
    x_large = jnp.ones((4, 8)) * 1e6
    y_large = block(x_large)
    assert not jnp.any(jnp.isnan(y_large))
    # Test with very small values
    x_small = jnp.ones((4, 8)) * 1e-6
    y_small = block(x_small)
    assert not jnp.any(jnp.isnan(y_small))

def test_standard_mlp_block_gradients(key):
    block = StandardMlpBlock(3, 2, jnn.relu, key=key)
    
    def loss_fn(model, x):
        return jnp.mean(model(x) ** 2)
    
    x = jnp.array([1.0, 2.0, 3.0])
    grads = eqx.filter_grad(loss_fn)(block, x)
    
    # Check if gradients exist for all parameters
    assert all(g is not None for g in jtu.tree_leaves(grads))

def test_bottleneck_mlp_block_parameter_shapes(key):
    in_features = 10
    ratio = 4
    block = BottleneckMlpBlock(in_features, jnn.relu, ratio=ratio, key=key)
    
    # Test internal block parameters
    assert block.linear1.weight.shape == (ratio * in_features, in_features)
    assert block.linear1.bias.shape == (ratio * in_features,)
    
    # Test projection parameters
    assert block.linear2.weight.shape == (in_features, ratio * in_features)
    assert block.linear2.bias.shape == (in_features,)

def test_mixer_block_batch_processing(key):
    block = MixerBlock(4, 8, 16, 32, jnn.gelu, key=key)
    batch_size = 3
    x_batch = jnp.ones((batch_size, 4, 8))
    y_batch = vmap(block)(x_batch)
    assert y_batch.shape == (batch_size, 4, 8)
