import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Optional, Callable, Tuple
from functools import wraps, partial
from einops import rearrange, repeat
import inspect


LRELU_SLOPE = 0.02


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn



class PreNorm(nnx.Module):
    def __init__(self, dim: int, fn: nnx.Module, context_dim: Optional[int] = None, *, rngs: nnx.Rngs):
        self.fn = fn
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_context = nnx.LayerNorm(context_dim, rngs=rngs) if exists(context_dim) else None

        # Cache function argument names to detect if 'context' is allowed
        self.fn_accepts_context = "context" in inspect.signature(fn.__call__).parameters

    def __call__(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            if "context" not in kwargs:
                raise ValueError("`context` must be passed when `context_dim` is set")
            context = kwargs["context"]
            normed_context = self.norm_context(context)

            if self.fn_accepts_context:
                return self.fn(x, context=normed_context)
            # Skip passing context if not supported
            return self.fn(x)

        return self.fn(x)


class GEGLU(nnx.Module):
    def __call__(self, x):
        x, gates = jnp.split(x, 2, axis=-1)
        return x * jax.nn.gelu(gates)


class FeedForward(nnx.Module):
    def __init__(self, dim: int, mult: int = 4, *, rngs: nnx.Rngs):
        self.net = [
            nnx.Linear(dim, dim * mult * 2, rngs=rngs),
            GEGLU(),
            nnx.Linear(dim * mult, dim, rngs=rngs),
        ]

    def __call__(self, x):
        for layer in self.net:
            x = layer(x)
        return x
    

class Attention(nnx.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nnx.Linear(query_dim, inner_dim, use_bias=False, rngs=rngs)
        self.to_kv = nnx.Linear(context_dim, inner_dim * 2, use_bias=False, rngs=rngs)
        self.to_out = nnx.Linear(inner_dim, query_dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.dropout_p = dropout

    def __call__(self, x, context=None, mask=None, training=False):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k, v = jnp.split(self.to_kv(context), 2, axis=-1)

        q = rearrange(q, "b n (h d) -> (b h) n d", h=h)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=h)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=h)

        sim = jnp.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -jnp.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim = jnp.where(mask, sim, max_neg_value)

        attn = jax.nn.softmax(sim, axis=-1)
        attn = self.dropout(attn, deterministic=not training)
        out = jnp.einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self.to_out(out)
        return out


# Activation layer factory
def act_layer(act: str):
    if act == "relu":
        return jax.nn.relu
    elif act == "lrelu":
        return lambda x: jax.nn.leaky_relu(x, negative_slope=LRELU_SLOPE)
    elif act == "elu":
        return jax.nn.elu
    elif act == "tanh":
        return jnp.tanh
    elif act == "prelu":
        raise ValueError("PReLU is parameterized — use a module-based approach like nnx.PReLU()")
    else:
        raise ValueError("%s not recognized." % act)


def norm_layer2d(norm: str, channels: int, *, rngs: nnx.Rngs):
    if norm == "batch":
        return nnx.BatchNorm(num_features=channels, rngs=rngs)
    elif norm == "instance":
        return nnx.GroupNorm(num_features=channels, num_groups=channels, rngs=rngs)
    elif norm == "layer":
        return nnx.GroupNorm(num_features=channels, num_groups=1, rngs=rngs)
    elif norm == "group":
        return nnx.GroupNorm(num_features=channels, num_groups=4, rngs=rngs)
    else:
        raise ValueError(f"{norm} not recognized.")


def norm_layer1d(norm: str, num_channels: int, *, rngs: nnx.Rngs):
    if norm == "batch":
        return nnx.BatchNorm(num_features=num_channels, rngs=rngs)
    elif norm == "instance":
        return nnx.GroupNorm(num_features=num_channels, num_groups=num_channels, rngs=rngs)
    elif norm == "layer":
        return nnx.GroupNorm(num_features=num_channels, num_groups=1, rngs=rngs)
    elif norm == "group":
        return nnx.GroupNorm(num_features=num_channels, num_groups=4, rngs=rngs)
    else:
        raise ValueError(f"{norm} not recognized.")


def calculate_gain(nonlinearity: str, param: float = 0.0):
    if nonlinearity == "linear":
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3.0
    elif nonlinearity == "relu":
        return jnp.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        return jnp.sqrt(2.0 / (1 + param**2))
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")


class Conv2DBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: int = 3,
        strides: int = 1,
        norm: str = None,
        activation: str = None,
        padding_mode: str = "replicate",  # note: not yet supported in JAX
        padding: int = None,
        *,
        rngs: nnx.Rngs,
    ):
        print(f"[Conv2DBlock] conv in_channels={in_channels}, out_channels={out_channels}")

        self.out_channels = out_channels
        padding = kernel_sizes // 2 if padding is None else padding

        # Flax/JAX Conv2D has no native support for `padding_mode`, so replicate manually outside if needed
        self.conv2d = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_sizes, kernel_sizes),
            strides=(strides, strides),
            padding=[(padding, padding), (padding, padding)],
            kernel_init=self._init_conv_kernel(activation),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

        self.norm = norm_layer2d(norm, out_channels, rngs=rngs) if norm is not None else None
        self.activation = act_layer(activation) if activation is not None else None

    def _init_conv_kernel(self, activation):
        if activation is None:
            return nnx.initializers.xavier_uniform()
        elif activation == "tanh":
            return nnx.initializers.xavier_uniform()
        elif activation == "lrelu":
            return nnx.initializers.kaiming_uniform()
        elif activation == "relu":
            return nnx.initializers.kaiming_uniform()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def __call__(self, x):
        print(f"[Conv2DBlock __call__] Expected in_channels={self.conv2d.in_features}, got x.shape={x.shape}")
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW


def bilinear_upsample(x: jax.Array, size: Tuple[int, int]) -> jax.Array:
    return jax.image.resize(x, (*x.shape[:2], *size), method="linear")


def bilinear_upsample_by_factor(x: jax.Array, factor: int) -> jax.Array:
    h, w = x.shape[2], x.shape[3]
    return bilinear_upsample(x, (h * factor, w * factor))


class Conv2DUpsampleBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strides: int,
        kernel_sizes: int = 3,
        norm: Optional[str] = None,
        activation: Optional[str] = None,
        out_size: Optional[Tuple[int, int]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        print(f"[Conv2DUpsampleBlock] in_channels={in_channels}, out_channels={out_channels}")


        # First conv block — defined before modifying kernel_sizes
        self.block1 = Conv2DBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            strides=1,
            norm=norm,
            activation=activation,
            rngs=rngs,
        )

        # Modify kernel size only after first conv
        if out_size is not None and kernel_sizes % 2 == 0:
            kernel_sizes += 1

        # Second conv block
        self.block2 = Conv2DBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            strides=1,
            norm=norm,
            activation=activation,
            rngs=rngs,
        )

        self.strides = strides
        self.out_size = out_size

    def __call__(self, x: jax.Array) -> jax.Array:
        print(f"[Conv2DUpsampleBlock __call__] x.shape = {x.shape}")
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        x = self.block1(x)

        if self.strides > 1:
            if self.out_size is None:
                x = bilinear_upsample_by_factor(x, self.strides)
            else:
                x = bilinear_upsample(x, self.out_size)

        x = self.block2(x)
        return jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW


class DenseBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: Optional[str] = None,
        activation: Optional[str] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear = nnx.Linear(
            in_features,
            out_features,
            kernel_init=self._init_linear_kernel(activation),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

        self.norm = norm_layer1d(norm, out_features, rngs=rngs) if norm is not None else None
        self.activation = act_layer(activation) if activation is not None else None

    def _init_linear_kernel(self, activation: Optional[str]):
        if activation is None or activation == "tanh":
            return nnx.initializers.xavier_uniform()
        elif activation == "lrelu" or activation == "relu":
            return nnx.initializers.kaiming_uniform()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def __call__(self, x):
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class FixedPositionalEncoding(nnx.Module):
    def __init__(self, feat_per_dim: int, feat_scale_factor: float):
        self.feat_per_dim = feat_per_dim
        self.feat_scale_factor = feat_scale_factor
        self.div_term = jnp.exp(
            jnp.arange(0, feat_per_dim, 2) * (-jnp.log(10000.0) / feat_per_dim)
        )[None, :]  # shape [1, feat_per_dim // 2]

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [batch_size, input_dim]
        assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"

        batch_size, input_dim = x.shape
        x = x.reshape(-1, 1)  # [B * D, 1]
        angles = self.feat_scale_factor * x * self.div_term  # [B * D, feat_per_dim // 2]

        out = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # [B * D, feat_per_dim]
        return out.reshape(batch_size, -1)  # [batch_size, input_dim * feat_per_dim]


def test_modules():
    B, N, D = 2, 16, 64
    context_D = 32
    mult = 4
    heads = 4
    dim_head = 16
    H, W = 32, 32
    C_in, C_out = 32, 64
    rngs = nnx.Rngs(jax.random.key(0))

    def check(name, actual, expected):
        assert actual == expected, f"{name} output shape {actual} != expected {expected}"
        print(f"{name} output shape: {actual} (expected {expected})")

    # Inputs
    x = jax.random.normal(rngs.key(), (B, N, D))
    context = jax.random.normal(rngs.key(), (B, N, context_D))

    # GEGLU
    geglu = GEGLU()
    geglu_out = geglu(jax.random.normal(rngs.key(), (B, N, D * 2)))
    check("GEGLU", geglu_out.shape, (B, N, D))

    # FeedForward
    ff = FeedForward(D, mult=mult, rngs=rngs)
    ff_out = ff(x)
    check("FeedForward", ff_out.shape, (B, N, D))

    # PreNorm
    prenorm = PreNorm(D, ff, context_dim=context_D, rngs=rngs)
    prenorm_out = prenorm(x, context=context)
    check("PreNorm(FeedForward)", prenorm_out.shape, (B, N, D))

    # Attention
    attn = Attention(
        query_dim=D,
        context_dim=context_D,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,
        rngs=rngs,
    )
    attn_out = attn(x, context=context, training=True)
    check("Attention", attn_out.shape, (B, N, D))

    # Conv2DBlock
    x2d = jax.random.normal(rngs.key(), (B, C_in, H, W))
    conv_block = Conv2DBlock(
        in_channels=C_in,
        out_channels=C_out,
        kernel_sizes=3,
        strides=1,
        norm="batch",
        activation="relu",
        rngs=rngs,
    )
    conv_out = conv_block(x2d)
    check("Conv2DBlock", conv_out.shape, (B, C_out, H, W))

    # Conv2DUpsampleBlock
    upsample_block = Conv2DUpsampleBlock(
        in_channels=C_in,
        out_channels=C_out,
        strides=2,
        kernel_sizes=3,
        norm="batch",
        activation="relu",
        out_size=(H * 2, W * 2),
        rngs=rngs,
    )
    upsample_out = upsample_block(x2d)
    check("Conv2DUpsampleBlock", upsample_out.shape, (B, C_out, H * 2, W * 2))

    # DenseBlock
    x_dense = jax.random.normal(rngs.key(), (B, D))
    dense = DenseBlock(D, D, norm="layer", activation="relu", rngs=rngs)
    dense_out = dense(x_dense)
    check("DenseBlock", dense_out.shape, (B, D))

    # FixedPositionalEncoding
    pos_enc = FixedPositionalEncoding(feat_per_dim=D, feat_scale_factor=1.0)
    pos_out = pos_enc(jax.random.normal(rngs.key(), (B, 1)))
    check("FixedPositionalEncoding", pos_out.shape, (B, D))


if __name__ == "__main__":
    # Run the test
    test_modules()