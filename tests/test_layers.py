import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import pytest

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
