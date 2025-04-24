from math import prod
from enum import Enum
from typing import Optional, Callable, Sequence, Union, Tuple

import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
from jax import vmap
from jaxtyping import Array, PRNGKeyArray
from einops import rearrange

from equinox import nn, Module

from .layers import MixerBlock, StandardMlpBlock, BottleneckMlpBlock
from .utils import PatchConvEmbed

gelu = lambda x: jnn.gelu(x, approximate=False)


class MlpType(Enum):
    """Types of MLP architectures available in DeepMlp."""
    BOTTLENECK = "bottleneck"
    STANDARD = "standard"


class MLP(nn.MLP):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network."""

    dropout: Callable

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = nn.Identity(),
        use_bias: bool = True,
        use_final_bias: bool = True,
        *,
        dropout_rate: float = 0.,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        super().__init__(in_size, out_size, width_size, depth, activation, final_activation, use_bias, use_final_bias, key=key)
        self.dropout = nn.Dropout(p=dropout_rate)

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)` after ravel is applied.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`.
        """
        x = jnp.ravel(x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x, key=key)

        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x


class MlpMixer(Module):
    """MLP Mixer architecture ported from https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py."""
    mixers: Sequence[MixerBlock]
    norm: nn.LayerNorm
    patch_embed: Module
    fc: nn.Linear

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 32,
            patch_size: Union[int, Tuple[int, int]] = 4,
            in_chans: int = 1,
            embed_dim: int = 256,
            tokens_hidden_dim: int = 128,
            hidden_dim_ratio: int = 4,
            num_classes: int = 10,
            num_blocks: int = 8,
            activation: Callable[[Array], Array] = gelu,
            patch_embed: Callable = PatchConvEmbed,
            *,
            key: PRNGKeyArray
        ):
        """**Arguments:**

        - `img_size`: The size of the input image. If an integer is provided, a square image is assumed.
        - `patch_size`: Size of the patch to construct from the input image.
        - `in_chans`: Number of input channels. Defaults to `1`.
        - `embed_dim`: The dimension of the embedding. Defaults to `256`.
        - `tokens_hidden_dim`: Hidden dimension for token mixing MLP. Defaults to `128`.
        - `hidden_dim_ratio`: Ratio for channel mixing hidden dimension. Defaults to `4`.
        - `num_classes`: Number of output classes. Defaults to `10`.
        - `num_blocks`: Number of mixer blocks to use. Defaults to `8`.
        - `activation`: Activation function to use. Defaults to `gelu`.
        - `patch_embed`: Patch embedding function to use. Defaults to `PatchConvEmbed`.
        - `key`: A `jax.random.PRNGKey` used for parameter initialization.
        """
        super().__init__()
        embed_hidden_dim = hidden_dim_ratio * tokens_hidden_dim
        keys = jr.split(key, num_blocks + 2)

        self.patch_embed = patch_embed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            key=keys[-2]
        )

        tokens_dim = self.patch_embed.num_patches
        self.mixers = [
            MixerBlock(
                tokens_dim, 
                embed_dim, 
                tokens_hidden_dim, 
                embed_hidden_dim, 
                activation, 
                key=keys[i]
            ) 
            for i in range(num_blocks)
        ]

        self.norm = nn.LayerNorm(embed_dim, eps=1e-5, use_bias=False)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes, key=keys[-1])

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(height, width, channels)`.
        - `key`: A `jax.random.PRNGKey` not used; present for syntax consistency. (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(num_classes,)`.
        """

        x = rearrange(x, 'h w c -> c h w')
        
        x = self.patch_embed(x)
        # x shape is (num_patches, embed_dim)
        for mixer in self.mixers:
            x = mixer(x)

        x = vmap(self.norm)(x)
        x = jnp.mean(x, axis=-2)
        return self.fc(x)


class DeepMlp(Module):
    """Deep Standard and Inverted Bottleneck MLP ported from https://github.com/gregorbachmann/scaling_mlps."""
    layers: Union[Sequence[BottleneckMlpBlock],Sequence[StandardMlpBlock]]
    linear_embed: nn.Linear
    fc: nn.Linear

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 32,
            in_chans: int = 1,
            embed_dim: int = 256,
            hidden_dim_ratio: int = 4,
            num_classes: int = 10,
            num_blocks: int = 8,
            activation: Callable[[Array], Array] = gelu,
            mlp_type: Union[MlpType, str] = MlpType.BOTTLENECK,
            *,
            key: PRNGKeyArray
        ):
        """**Arguments:**

        - `img_size`: The size of the input image. If an integer is provided, a square image is assumed.
        - `in_chans`: Number of input channels. Defaults to `1`.
        - `embed_dim`: The dimension of the embedding. Defaults to `256`.
        - `hidden_dim_ratio`: Ratio for hidden dimension expansion. Defaults to `4`.
        - `num_classes`: Number of output classes. Defaults to `10`.
        - `num_blocks`: Number of MLP blocks to use. Defaults to `8`.
        - `activation`: Activation function to use. Defaults to `gelu`.
        - `mlp_type`: Type of MLP architecture to use. Either MlpType.BOTTLENECK or MlpType.STANDARD.
        - `key`: A `jax.random.PRNGKey` used for parameter initialization.
        """
        super().__init__()
        keys = jr.split(key, num_blocks + 2)
        
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        in_features = prod(img_size) * in_chans
        self.linear_embed = nn.Linear(in_features=in_features, out_features=embed_dim, key=keys[-1])
        
        # Convert string to enum if needed
        if isinstance(mlp_type, str):
            try:
                mlp_type = MlpType(mlp_type)
            except ValueError:
                raise ValueError(f"Invalid MLP type: {mlp_type}. Must be one of {[t.value for t in MlpType]}")
        
        if mlp_type == MlpType.BOTTLENECK:
            self.layers = [
                BottleneckMlpBlock(
                    embed_dim,
                    activation,
                    ratio=hidden_dim_ratio,
                    key=keys[i]
                ) 
                for i in range(num_blocks)
            ]

            # Classifier head
            self.fc = nn.Linear(embed_dim, num_classes, key=keys[-2])

        elif mlp_type == MlpType.STANDARD:
            self.layers = [
                StandardMlpBlock(
                    embed_dim, 
                    embed_dim,
                    activation, 
                    key=keys[i]
                ) 
                for i in range(num_blocks)
            ]

            # Classifier head
            self.fc = nn.Sequential(
                [   
                    nn.LayerNorm(embed_dim, eps=1e-5),
                    nn.Lambda(activation),
                    nn.Linear(embed_dim, num_classes, key=keys[-2]),
                ]
            )

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `img_size + (num_channels,)`.
        - `key`: A `jax.random.PRNGKey` not used; present for syntax consistency. (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(num_classes,)`.
        """
        x = rearrange(x, 'h w c -> (c h w)')
        x = self.linear_embed(x)
        # x shape is embed_dim
        for layer in self.layers:
            x = layer(x)

        return self.fc(x)
