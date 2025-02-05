from typing import Callable, Sequence

import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jaxtyping import PRNGKeyArray
from einops import rearrange

from equinox import Module, nn

class MlpBlock(Module):
    activation: Callable
    layers: Sequence[nn.Linear]

    def __init__(
        self,
        in_features,
        hidden_dim, 
        out_features,
        activation,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()
        keys = jr.split(key, 2)
        self.activation = activation
        
        self.layers = (
            nn.Linear(in_features=in_features, out_features=hidden_dim, key=keys[0]),
            nn.Linear(in_features=hidden_dim, out_features=out_features, key=keys[1])
        )

    def __call__(self, x):
        y = self.layers[0](x)
        y = self.activation(y)
        return self.layers[1](y)


class MixerBlock(Module):
    """Mixer block layer."""
    norm1: Callable
    norm2: Callable
    blocks: Sequence[MlpBlock]

    def __init__(
        self,
        tokens_dim,
        embed_dim,
        tokens_hidden_dim,
        embed_hidden_dim,
        activation,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6, use_weight=False, use_bias=False)

        keys = jr.split(key, 2)
        self.blocks = [
            MlpBlock(tokens_dim, tokens_hidden_dim, tokens_dim, activation, key=keys[0]),  # token_mixing
            MlpBlock(embed_dim, embed_hidden_dim, embed_dim, activation, key=keys[1])  # channel_mixing
        ]
    
    def __call__(self, x):
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
    norm: Callable
    activation: Callable
    linear: nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        activation,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, key=key)
        self.norm = nn.LayerNorm(in_features, use_weight=False, use_bias=False)

    def __call__(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return self.activation(x)


class BottleneckMlpBlock(Module):
    block: StandardMlpBlock
    linear: nn.Linear

    def __init__(
        self,
        in_features,
        activation,
        ratio=4,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()
        keys = jr.split(key, 2)
        self.block = StandardMlpBlock(in_features, ratio * in_features, activation, key=keys[0])
        self.linear = nn.Linear(in_features=ratio * in_features, out_features=in_features, key=keys[1])

    def __call__(self, x):
        y = self.block(x)
        return x + self.linear(y)
