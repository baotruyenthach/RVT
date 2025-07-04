import torch
import jax
import jax.numpy as jnp
import numpy as np

from attn import FeedForward as TorchFF
from attn_jax import FeedForward as FlaxFF
import flax.nnx as nnx


def constant_ff_torch(D, mult):
    ff = TorchFF(D, mult=mult)

    with torch.no_grad():
        for layer in ff.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(1.0)
                layer.bias.fill_(0.0)
    return ff

def constant_ff_flax(D, mult, rngs):
    ff = FlaxFF(D, mult=mult, rngs=rngs)
    for layer in ff.net:
        if isinstance(layer, nnx.Linear):
            layer.kernel.value = jnp.ones_like(layer.kernel.value)
            layer.bias.value = jnp.zeros_like(layer.bias.value)
    return ff


def test_feedforward_equivalence():
    B, N, D = 2, 4, 8
    mult = 2

    input_np = np.random.randn(B, N, D).astype(np.float32)
    input_torch = torch.tensor(input_np, dtype=torch.float32)
    input_jax = jnp.array(input_np)

    # Create models with constant weights
    torch_ff = constant_ff_torch(D, mult)
    flax_ff = constant_ff_flax(D, mult, rngs=nnx.Rngs(jax.random.key(0)))

    # Run both
    torch_output = torch_ff(input_torch).detach().numpy()
    flax_output = flax_ff(input_jax)

    assert np.allclose(torch_output, np.array(flax_output), atol=1e-5, rtol=1e-4), \
        f"Mismatch:\nPyTorch: {torch_output}\nFlax: {np.array(flax_output)}"

    print("✅ FeedForward outputs match!")

if __name__ == "__main__":
    test_feedforward_equivalence()
