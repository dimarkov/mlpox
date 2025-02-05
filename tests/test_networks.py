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

def test_mlp_numerical_stability(key):
    mlp = MLP(10, 2, 32, 3, dropout_rate=0.1, key=key)
    x = jnp.linspace(-100, 100, 10)  # Test with large range
    y = mlp(x, key=key)
    assert not jnp.any(jnp.isnan(y))
    assert not jnp.any(jnp.isinf(y))

def test_mlp_mixer_gradients(key):
    mixer = MlpMixer(img_size=16, patch_size=4, in_channels=1, num_blocks=2, key=key)
    
    def loss_fn(params, x):
        mixer_copy = mixer.replace(**params)
        return jnp.mean(mixer_copy(x) ** 2)
    
    x = jnp.ones((16, 16, 1))
    grads = grad(loss_fn)(mixer.filter(lambda p: True), x)
    
    # Check if gradients exist and are finite
    assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads))

def test_deep_mlp_parameter_count(key):
    mlp = DeepMlp(
        img_size=16,
        in_channels=1,
        embed_dim=32,
        num_blocks=2,
        type="bottleneck",
        key=key
    )
    
    # Count parameters
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(mlp.filter(lambda p: True)))
    
    # Calculate expected parameter count
    expected_count = (
        16 * 16 * 32 +  # linear_embed
        2 * (32 * 128 + 128 * 32) +  # bottleneck blocks
        32 * 10  # final classifier
    )
    assert param_count == expected_count

def test_networks_batch_processing(key):
    # Test MLP with batch
    mlp = MLP(784, 10, 256, 2, dropout_rate=0.1, key=key)
    x_batch = jnp.ones((5, 784))
    y_batch = vmap(mlp, in_axes=(0, None))(x_batch, key)
    assert y_batch.shape == (5, 10)
    
    # Test MlpMixer with batch
    mixer = MlpMixer(img_size=32, patch_size=4, in_channels=3, num_blocks=2, key=key)
    x_batch = jnp.ones((5, 32, 32, 3))
    y_batch = vmap(mixer, in_axes=(0, None))(x_batch, key)
    assert y_batch.shape == (5, 10)
