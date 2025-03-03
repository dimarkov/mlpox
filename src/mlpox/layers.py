from typing import Callable, Sequence, Tuple

import jax.random as jr
from jax import vmap
from jaxtyping import Array, PRNGKeyArray
from einops import rearrange

from equinox import Module, nn

class MlpBlock(Module):
    """Two-layer MLP block with activation in between."""
    activation: Callable
    layers: Tuple[nn.Linear, nn.Linear]

    def __init__(
        self,
        in_features: int,
        hidden_dim: int, 
        out_features: int,
        activation: Callable,
        *,
        key: PRNGKeyArray
    ):
        """
        **Arguments:**

        - `in_features`: Number of input features.
        - `hidden_dim`: Number of hidden dimensions.
        - `out_features`: Number of output features.
        - `activation`: Activation function to apply after the first layer.
        - `key`: A `jax.random.PRNGKey` used for parameter initialization.
        """
        super().__init__()
        keys = jr.split(key, 2)
        self.activation = activation
        
        self.layers = (
            nn.Linear(in_features=in_features, out_features=hidden_dim, key=keys[0]),
            nn.Linear(in_features=hidden_dim, out_features=out_features, key=keys[1])
        )

    def __call__(self, x: Array) -> Array:
        """
        **Arguments:**

        - `x`: Input array with shape `(in_features,)`.

        **Returns:**

        Output array with shape `(out_features,)`.
        """
        y = self.layers[0](x)
        y = self.activation(y)
        return self.layers[1](y)


class MixerBlock(Module):
    """Mixer block layer for MLP-Mixer architecture."""
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    blocks: Sequence[MlpBlock]

    def __init__(
        self,
        tokens_dim: int,
        embed_dim: int,
        tokens_hidden_dim: int,
        embed_hidden_dim: int,
        activation: Callable,
        *,
        key: PRNGKeyArray
    ):
        """
        **Arguments:**

        - `tokens_dim`: Number of tokens (patches).
        - `embed_dim`: Embedding dimension for each token.
        - `tokens_hidden_dim`: Hidden dimension for token-mixing MLP.
        - `embed_hidden_dim`: Hidden dimension for channel-mixing MLP.
        - `activation`: Activation function to use in MLPs.
        - `key`: A `jax.random.PRNGKey` used for parameter initialization.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6, use_weight=False, use_bias=False)

        keys = jr.split(key, 2)
        self.blocks = [
            MlpBlock(tokens_dim, tokens_hidden_dim, tokens_dim, activation, key=keys[0]),  # token_mixing
            MlpBlock(embed_dim, embed_hidden_dim, embed_dim, activation, key=keys[1])  # channel_mixing
        ]
    
    def __call__(self, x: Array) -> Array:
        """
        **Arguments:**

        - `x`: Input array with shape `(tokens_dim, embed_dim)`.

        **Returns:**

        Output array with shape `(tokens_dim, embed_dim)`.
        """
        # x: (tokens embed_dim)
        y = vmap(self.norm1)(x)
        # Token mixing: transpose for per-channel mixing
        y = rearrange(y, 't e -> e t')  # (embed_dim tokens)
        y = vmap(self.blocks[0])(y)     # Apply token mixing to each channel
        y = rearrange(y, 'e t -> t e')  # (tokens embed_dim)
        x = x + y
        # Channel mixing: apply directly to each token
        y = vmap(self.norm2)(x)
        return x + vmap(self.blocks[1])(y)


class StandardMlpBlock(Module):
    """Standard MLP block with layer normalization."""
    norm: nn.LayerNorm
    activation: Callable
    linear: nn.Linear

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable[[Array], Array],
        *,
        key: PRNGKeyArray
    ):
        """
        **Arguments:**

        - `in_features`: Number of input features.
        - `out_features`: Number of output features.
        - `activation`: Activation function to apply after the linear layer.
        - `key`: A `jax.random.PRNGKey` used for parameter initialization.
        """
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, key=key)
        self.norm = nn.LayerNorm(in_features, use_weight=False, use_bias=False)

    def __call__(self, x: Array) -> Array:
        """
        **Arguments:**

        - `x`: Input array with shape `(in_features,)`.

        **Returns:**

        Output array with shape `(out_features,)`.
        """
        x = self.norm(x)
        x = self.linear(x)
        return self.activation(x)


class BottleneckMlpBlock(Module):
    """Bottleneck MLP block with residual connection."""
    block: StandardMlpBlock
    linear: nn.Linear

    def __init__(
        self,
        in_features: int,
        activation: Callable[[Array], Array],
        ratio: int = 4,
        *,
        key: PRNGKeyArray
    ):
        """
        **Arguments:**

        - `in_features`: Number of input features.
        - `activation`: Activation function to use in the StandardMlpBlock.
        - `ratio`: Expansion ratio for hidden dimension. Defaults to `4`.
        - `key`: A `jax.random.PRNGKey` used for parameter initialization.
        """
        super().__init__()
        keys = jr.split(key, 2)
        self.block = StandardMlpBlock(in_features, ratio * in_features, activation, key=keys[0])
        self.linear = nn.Linear(in_features=ratio * in_features, out_features=in_features, key=keys[1])

    def __call__(self, x: Array) -> Array:
        """
        **Arguments:**

        - `x`: Input array with shape `(in_features,)`.

        **Returns:**

        Output array with shape `(in_features,)` (same as input due to residual connection).
        """
        y = self.block(x)
        return x + self.linear(y)
