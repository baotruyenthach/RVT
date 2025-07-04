import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.nn import relu, softmax


class ConvexUpSample(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        up_ratio: int,
        up_kernel: int = 3,
        mask_scale: float = 0.1,
        with_bn: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        assert up_kernel % 2 == 1

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.up_ratio = up_ratio
        self.up_kernel = up_kernel
        self.mask_scale = mask_scale
        self.with_bn = with_bn

        self.net_out_conv1 = nnx.Conv(in_dim, 2 * in_dim, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.net_out_conv2 = nnx.Conv(2 * in_dim, 2 * in_dim, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.net_out_conv3 = nnx.Conv(2 * in_dim, out_dim, kernel_size=(3, 3), padding=1, rngs=rngs)

        if with_bn:
            self.net_out_bn1 = nnx.BatchNorm(2 * in_dim, rngs=rngs)
            self.net_out_bn2 = nnx.BatchNorm(2 * in_dim, rngs=rngs)

        mask_dim = (up_ratio ** 2) * (up_kernel ** 2)
        self.mask_conv1 = nnx.Conv(in_dim, 2 * in_dim, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.mask_conv2 = nnx.Conv(2 * in_dim, mask_dim, kernel_size=(1, 1), padding=0, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Convert NCHW → NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        B, H, W, C = x.shape
        assert C == self.in_dim, f"Expected input dim {self.in_dim}, got {C}"

        out = self.net_out_conv1(x)
        if self.with_bn:
            out = self.net_out_bn1(out)
        out = relu(out)

        out = self.net_out_conv2(out)
        if self.with_bn:
            out = self.net_out_bn2(out)
        out = relu(out)

        out = self.net_out_conv3(out)

        mask = relu(self.mask_conv1(x))
        mask = self.mask_scale * self.mask_conv2(mask)
        mask = mask.reshape(B, 1, self.up_kernel ** 2, self.up_ratio, self.up_ratio, H, W)
        mask = softmax(mask, axis=2)

        patches = jax.lax.conv_general_dilated_patches(
            out,
            filter_shape=(self.up_kernel, self.up_kernel),
            window_strides=(1, 1),
            padding="SAME",
        )
        patches = patches.reshape(B, H, W, self.out_dim, self.up_kernel ** 2)
        patches = patches.transpose(0, 3, 4, 1, 2)  # B, out_dim, K², H, W
        patches = patches[:, :, :, None, None, :, :]  # B, out_dim, K², 1, 1, H, W

        out = jnp.sum(patches * mask, axis=2)
        out = out.transpose(0, 1, 4, 2, 5, 3).reshape(B, self.out_dim, H * self.up_ratio, W * self.up_ratio)

        return out  # Still in NCHW


def test_convex_upsample():
    rng = nnx.Rngs(jax.random.key(0))
    x = jnp.ones((2, 32, 8, 8))  # NCHW

    # Test without BN
    model = ConvexUpSample(in_dim=32, out_dim=1, up_ratio=2, with_bn=False, rngs=rng)
    out = model(x)
    assert out.shape == (2, 1, 16, 16)
    print("✅ ConvexUpSample (no BN) passed.")

    # Test with BN
    model_bn = ConvexUpSample(in_dim=32, out_dim=1, up_ratio=2, with_bn=True, rngs=rng)
    out_bn = model_bn(x)
    assert out_bn.shape == (2, 1, 16, 16)
    print("✅ ConvexUpSample (with BN) passed.")


if __name__ == "__main__":
    test_convex_upsample()
