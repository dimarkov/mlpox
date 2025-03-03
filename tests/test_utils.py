import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from mlpox.utils import (
    Patch,
    PatchConvEmbed,
    PatchLinearEmbed
)

@pytest.fixture
def key():
    return jr.PRNGKey(0)

def test_patch():
    patch = Patch(img_size=32, patch_size=4, in_chans=3, embed_dim=128)
    assert patch.grid_size == (8, 8)
    assert patch.num_patches == 64
    assert patch.in_chans == 3
    assert patch.embed_dim == 128

def test_patch_conv_embed(key):
    embed = PatchConvEmbed(
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=256,
        key=key
    )
    x = jnp.ones((3, 32, 32))
    y = embed(x)
    assert y.shape == (64, 256)  # (num_patches, embed_dim)

def test_patch_linear_embed(key):
    embed = PatchLinearEmbed(
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=256,
        key=key
    )
    x = jnp.ones((3, 32, 32))
    y = embed(x)
    assert y.shape == (64, 256)  # (num_patches, embed_dim)

def test_patch_size_validation():
    # Test base Patch class validation
    with pytest.raises(ValueError):
        Patch(
            img_size=32,
            patch_size=5,  # Not divisible into image size
            in_chans=3,
            embed_dim=256
        )
    
    # Test PatchConvEmbed validation (inherited from Patch)
    with pytest.raises(ValueError):
        PatchConvEmbed(
            img_size=32,
            patch_size=5,  # Not divisible into image size
            in_chans=3,
            embed_dim=256,
            key=jr.PRNGKey(0)
        )
    
    # Test PatchLinearEmbed validation (inherited from Patch)
    with pytest.raises(ValueError):
        PatchLinearEmbed(
            img_size=32,
            patch_size=5,  # Not divisible into image size
            in_chans=3,
            embed_dim=256,
            key=jr.PRNGKey(0)
        )

def test_patch_numerical_accuracy(key):
    embed = PatchConvEmbed(img_size=8, patch_size=2, in_chans=1, embed_dim=4, key=key)
    x = jnp.arange(64, dtype=jnp.float32).reshape(1, 8, 8)
    y = embed(x)
    # Test if output is deterministic
    y2 = embed(x)
    np.testing.assert_allclose(y, y2)

def test_patch_edge_cases(key):
    embed = PatchLinearEmbed(img_size=8, patch_size=2, in_chans=1, embed_dim=4, key=key)
    # Test with zero input
    x_zeros = jnp.zeros((1, 8, 8))
    y_zeros = embed(x_zeros)
    assert not jnp.any(jnp.isnan(y_zeros))
    # Test with very large values
    x_large = jnp.ones((1, 8, 8)) * 1e6
    y_large = embed(x_large)
    assert not jnp.any(jnp.isnan(y_large))

def test_patch_gradients(key):
    embed = PatchConvEmbed(img_size=8, patch_size=2, in_chans=1, embed_dim=4, key=key)
    
    def loss_fn(model, x):
        return jnp.mean(model(x) ** 2)
    
    x = jnp.ones((1, 8, 8))
    grads = eqx.filter_grad(loss_fn)(embed, x)
    
    # Check if gradients exist and are finite
    assert all(jnp.all(jnp.isfinite(g)) for g in jtu.tree_flatten(grads)[0])

def test_patch_parameter_shapes(key):
    in_chans = 3
    patch_size = 4
    embed_dim = 8
    
    conv_embed = PatchConvEmbed(
        img_size=16,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        key=key
    )
    
    # Test conv parameters
    assert conv_embed.proj.weight.shape == (embed_dim, in_chans, patch_size, patch_size)
    assert conv_embed.proj.bias.shape == (embed_dim, 1, 1)
    
    linear_embed = PatchLinearEmbed(
        img_size=16,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        key=key
    )
    
    # Test linear parameters
    in_features = patch_size * patch_size * in_chans
    assert linear_embed.linear.weight.shape == (embed_dim, in_features)
    assert linear_embed.linear.bias.shape == (embed_dim,)
