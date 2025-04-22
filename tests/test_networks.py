import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jax import vmap, tree_util as jtu
import pytest

from mlpox.networks import (
    MLP,
    MlpMixer,
    DeepMlp,
    MlpType
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
        in_chans=3,
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
        in_chans=3,
        embed_dim=256,
        num_classes=10,
        num_blocks=2,
        mlp_type="bottleneck",
        key=key
    )
    x = jnp.ones((32, 32, 3))
    y = mlp(x, key=key)
    assert y.shape == (10,)

def test_deep_mlp_standard(key):
    mlp = DeepMlp(
        img_size=32,
        in_chans=3,
        embed_dim=256,
        num_classes=10,
        num_blocks=2,
        mlp_type="standard",
        key=key
    )
    x = jnp.ones((32, 32, 3))
    y = mlp(x, key=key)
    assert y.shape == (10,)

def test_mlp_type_enum():
    # Test enum values
    assert MlpType.BOTTLENECK.value == "bottleneck"
    assert MlpType.STANDARD.value == "standard"
    
    # Test string conversion
    assert MlpType("bottleneck") == MlpType.BOTTLENECK
    assert MlpType("standard") == MlpType.STANDARD
    
    # Test invalid value
    with pytest.raises(ValueError):
        MlpType("invalid")

def test_deep_mlp_with_enum_type(key):
    # Test with enum value directly
    mlp = DeepMlp(
        img_size=32,
        in_chans=3,
        embed_dim=256,
        num_classes=10,
        num_blocks=2,
        mlp_type=MlpType.BOTTLENECK,
        key=key
    )
    x = jnp.ones((32, 32, 3))
    y = mlp(x, key=key)
    assert y.shape == (10,)

def test_deep_mlp_invalid_type(key):
    with pytest.raises(ValueError):
        DeepMlp(
            img_size=32,
            in_chans=3,
            embed_dim=256,
            num_classes=10,
            num_blocks=2,
            mlp_type="invalid",
            key=key
        )

def test_mlp_numerical_stability(key):
    mlp = MLP(10, 2, 32, 3, dropout_rate=0.1, key=key)
    x = jnp.linspace(-100, 100, 10)  # Test with large range
    y = mlp(x, key=key)
    assert not jnp.any(jnp.isnan(y))
    assert not jnp.any(jnp.isinf(y))

def test_mlp_mixer_gradients(key):
    mixer = MlpMixer(img_size=16, patch_size=4, in_chans=1, num_blocks=2, key=key)
    
    def loss_fn(model, x):
        return jnp.mean(model(x) ** 2)
    
    x = jnp.ones((16, 16, 1))
    grads = eqx.filter_grad(loss_fn)(mixer, x)
    
    # Check if gradients exist and are finite
    assert all(jnp.all(jnp.isfinite(g)) for g in jtu.tree_leaves(grads))

def test_deep_mlp_gradients(key):
    dmlp = DeepMlp(img_size=16, in_chans=1, num_blocks=2, mlp_type="standard", key=key)
    
    def loss_fn(model, x):
        return jnp.mean(model(x) ** 2)
    
    x = jnp.ones((16, 16, 1))
    grads = eqx.filter_grad(loss_fn)(dmlp, x)
    
    # Check if gradients exist and are finite
    assert all(jnp.all(jnp.isfinite(g)) for g in jtu.tree_leaves(grads))

    dmlp = DeepMlp(img_size=16, in_chans=1, num_blocks=2, mlp_type="bottleneck", key=key)
    grads = eqx.filter_grad(loss_fn)(dmlp, x)
    
    # Check if gradients exist and are finite
    assert all(jnp.all(jnp.isfinite(g)) for g in jtu.tree_leaves(grads))

def test_deep_mlp_parameter_count(key):
    mlp = DeepMlp(
        img_size=16,
        in_chans=1,
        embed_dim=32,
        num_blocks=2,
        mlp_type="bottleneck",
        key=key
    )
    
    # Count parameters
    params, static = eqx.partition(mlp, eqx.is_array)
    param_count = sum(p.size for p in jtu.tree_leaves(params))
    
    # Calculate expected parameter count
    expected_count = (
        16 * 16 * 32 + 32 + # linear_embed
        2 * (32 * 129 + 128 * 33) +  # bottleneck blocks
        2 * (2 * 32) +  # norm layer
        33 * 10  # final classifier
    )
    assert param_count == expected_count

def test_networks_batch_processing(key):
    # Test MLP with batch
    batch_size = 5
    key = jr.PRNGKey(0)
    keys = jr.split(key, batch_size)
    mlp = MLP(784, 10, 256, 2, dropout_rate=0.1, key=key)
    x_batch = jnp.ones((batch_size, 784))
    y_batch = vmap(mlp)(x_batch, key=keys)
    assert y_batch.shape == (5, 10)
    
    # Test MlpMixer with batch
    mixer = MlpMixer(img_size=32, patch_size=4, in_chans=3, num_blocks=2, key=key)
    x_batch = jnp.ones((5, 32, 32, 3))
    y_batch = vmap(mixer)(x_batch, key=keys)
    assert y_batch.shape == (5, 10)

    # Test Standard DeepMlp with batch
    dmlp = DeepMlp(img_size=32, in_chans=3, num_blocks=2, mlp_type="standard", key=key)
    y_batch = vmap(dmlp)(x_batch, key=keys)
    assert y_batch.shape == (5, 10)

    # Test Bottleneck DeepMlp with batch
    dmlp = DeepMlp(img_size=32, in_chans=3, num_blocks=2, mlp_type="bottleneck", key=key)
    y_batch = vmap(dmlp)(x_batch, key=keys)
    assert y_batch.shape == (5, 10)
