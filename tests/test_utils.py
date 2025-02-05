import jax
import jax.numpy as jnp
import jax.random as jr
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
    patch = Patch(img_size=32, patch_size=4)
    assert patch.grid_size == (8, 8)
    assert patch.num_patches == 64

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
    with pytest.raises(ValueError):
        PatchConvEmbed(
            img_size=32,
            patch_size=5,  # Not divisible into image size
            in_chans=3,
            embed_dim=256,
            key=jr.PRNGKey(0)
        )
