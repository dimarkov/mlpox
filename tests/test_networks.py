import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import pytest

from mlpox.networks import (
    MLP,
    MlpMixer,
    DeepMlp
)

@pytest.fixture
def key():
    return jr.PRNGKey(0)

def test_mlp(key):
    mlp = MLP(
        in_size=784,
        out_size=10,
        width_size=256,
        depth=2,
        dropout_rate=0.1,
        key=key
    )
    x = jnp.ones(784)
    y = mlp(x, key=key)
    assert y.shape == (10,)

def test_mlp_mixer(key):
    mixer = MlpMixer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=256,
        tokens_hidden_dim=128,
        num_classes=10,
        num_blocks=2,
        key=key
    )
    x = jnp.ones((32, 32, 3))
    y = mixer(x, key=key)
    assert y.shape == (10,)

def test_deep_mlp_bottleneck(key):
    mlp = DeepMlp(
        img_size=32,
        in_channels=3,
        embed_dim=256,
        num_classes=10,
        num_blocks=2,
        type="bottleneck",
        key=key
    )
    x = jnp.ones((32, 32, 3))
    y = mlp(x, key=key)
    assert y.shape == (10,)

def test_deep_mlp_standard(key):
    mlp = DeepMlp(
        img_size=32,
        in_channels=3,
        embed_dim=256,
        num_classes=10,
        num_blocks=2,
        type="standard",
        key=key
    )
    x = jnp.ones((32, 32, 3))
    y = mlp(x, key=key)
    assert y.shape == (10,)

def test_deep_mlp_invalid_type(key):
    with pytest.raises(NotImplementedError):
        DeepMlp(
            img_size=32,
            in_channels=3,
            embed_dim=256,
            num_classes=10,
            num_blocks=2,
            type="invalid",
            key=key
        )
