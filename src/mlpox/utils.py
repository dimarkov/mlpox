from typing import Tuple, Union, Optional, Callable
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap
from jaxtyping import Array, PRNGKeyArray
from einops import rearrange, repeat
import equinox as eqx
from equinox import Module, nn


class Patch(Module):
    """Patch Embedding settings"""

    img_size: Tuple[int, int]
    patch_size: Tuple[int, int]
    grid_size: Tuple[int, int]
    num_patches: int
    flatten: bool
    in_chans: int
    embed_dim: int
    
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
    ):
        """
        **Arguments:**

        - `img_size`: The size of the input image. Defaults to `(224, 224)`
        - `patch_size`: Size of the patch to construct from the input image. Defaults to `(16, 16)`
        - `in_chans`: Number of input channels. Defaults to `3`
        - `embed_dim`: The dimension of the resulting embedding of the patch. Defaults to `768`
        - `flatten`: If enabled, the `2d` patches are flattened to `1d`
        """
        super().__init__()
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        
        # Validate that patch size divides image size evenly
        if self.img_size[0] % self.patch_size[0] != 0 or self.img_size[1] % self.patch_size[1] != 0:
            raise ValueError(
                f"Image size {self.img_size} must be divisible by patch size {self.patch_size}"
            )
            
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim


class PatchConvEmbed(Patch):
    """2D Image to Patch Embedding using Convolution"""

    proj: nn.Conv2d

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """
        **Arguments:**

        - `img_size`: The size of the input image. Defaults to `(224, 224)`
        - `patch_size`: Size of the patch to construct from the input image. Defaults to `(16, 16)`
        - `in_chans`: Number of input channels. Defaults to `3`
        - `embed_dim`: The dimension of the resulting embedding of the patch. Defaults to `768`
        - `flatten`: If enabled, the `2d` patches are flattened to `1d`
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(img_size, patch_size, in_chans, embed_dim, flatten)

        _, key = jrandom.split(key)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, key=key
        )

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape`(in_chans, img_size[0], img_size[1])`.
        - `key`: Ignored
        """

        # Apply convolution
        x = self.proj(x)  # (embed_dim, H', W')
        
        if self.flatten:
            # Rearrange from (embed_dim, H', W') to (H'*W', embed_dim)
            x = rearrange(x, 'e h w -> (h w) e')

        return x


class PatchLinearEmbed(Patch):
    """2D Image to Patch Embedding using Linear projection"""

    linear: nn.Linear

    def __init__(
            self, 
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            *,
            key: PRNGKeyArray
        ):
        """
        **Arguments:**

        - `img_size`: The size of the input image. Defaults to `(224, 224)`
        - `patch_size`: Size of the patch to construct from the input image. Defaults to `(16, 16)`
        - `in_chans`: Number of input channels. Defaults to `3`
        - `embed_dim`: The dimension of the resulting embedding of the patch. Defaults to `768`
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        
        Note: This implementation always uses flatten=True.
        """
        super().__init__(img_size, patch_size, in_chans, embed_dim, True)
        _, key = jrandom.split(key)
        
        in_features = in_chans * patch_size * patch_size
        self.linear = nn.Linear(in_features, embed_dim, key=key)
    
    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape`(in_chans, img_size[0], img_size[1])`.
        - `key`: Ignored

        **Returns:**
        
        A JAX array of shape `(num_patches, embed_dim)`.
        """
        # Rearrange input into patches
        x = rearrange(
            x, 
            'c (h p1) (w p2) -> (h w) (p1 p2 c)', 
            p1=self.patch_size[0], 
            p2=self.patch_size[1]
        )
        
        # Apply linear projection to each patch
        return vmap(self.linear)(x)
