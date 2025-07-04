import flax.nnx as nnx
import jax
import jax.numpy as jnp
from rvt.mvt import renderer
from rvt.mvt.attn_jax import (
    Conv2DBlock, Conv2DUpsampleBlock, DenseBlock,
    Attention, FeedForward, PreNorm,
    FixedPositionalEncoding, cache_fn
)
from rvt.mvt.raft_utils_jax import ConvexUpSample
from jax.nn import softmax
from jax.lax import conv_general_dilated_patches

class MVT(nnx.Module):
    def __init__(
        self,
        depth,
        img_size,
        add_proprio,
        proprio_dim,
        add_lang,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        rend_three_views,
        use_point_renderer,
        pe_fix,
        feat_ver,
        wpt_img_aug,
        inp_pre_pro,
        inp_pre_con,
        cvx_up,
        xops,
        rot_ver,
        num_rot,
        renderer_device="cuda:0",
        renderer=None,
        no_feat=False,
        *,
        rngs: nnx.Rngs,
    ):
        """MultiView Transfomer

        :param depth: depth of the attention network
        :param img_size: number of pixels per side for rendering
        :param renderer_device: device for placing the renderer
        :param add_proprio:
        :param proprio_dim:
        :param add_lang:
        :param lang_dim:
        :param lang_len:
        :param img_feat_dim:
        :param feat_dim:
        :param im_channels: intermediate channel size
        :param attn_dim:
        :param attn_heads:
        :param attn_dim_head:
        :param activation:
        :param weight_tie_layers:
        :param attn_dropout:
        :param decoder_dropout:
        :param img_patch_size: intial patch size
        :param final_dim: final dimensions of features
        :param self_cross_ver:
        :param add_corr:
        :param norm_corr: wether or not to normalize the correspondece values.
            this matters when pc is outide -1, 1 like for the two stage mvt
        :param add_pixel_loc:
        :param add_depth:
        :param rend_three_views: True/False. Render only three views,
            i.e. top, right and front. Overwrites other configurations.
        :param use_point_renderer: whether to use the point renderer or not
        :param pe_fix: matter only when add_lang is True
            Either:
                True: use position embedding only for image tokens
                False: use position embedding for lang and image token
        :param feat_ver: whether to max pool final features or use soft max
            values using the heamtmap
        :param wpt_img_aug: how much noise is added to the wpt_img while
            training, expressed as a percentage of the image size
        :param inp_pre_pro: whether or not we have the intial input
            preprocess layer. this was used in peract but not not having it has
            cost advantages. if false, we replace the 1D convolution in the
            orginal design with identity
        :param inp_pre_con: whether or not the output of the inital
            preprocessing layer is concatenated with the ouput of the
            upsampling layer for the "final" layer
        :param cvx_up: whether to use learned convex upsampling
        :param xops: whether to use xops or not
        :param rot_ver: version of the rotation prediction network
            Either:
                0: same as peract, independent discrete xyz predictions
                1: xyz prediction dependent on one another
        :param num_rot: number of discrete rotations per axis, used only when
            rot_ver is 1
        :param no_feat: whether to return features or not
        """
        # ===============================
        # Store configuration parameters
        # ===============================
        self.training = True

        self.depth = depth
        self.img_size = img_size
        self.img_feat_dim = img_feat_dim
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        self.lang_dim = lang_dim
        self.lang_len = lang_len
        self.im_channels = im_channels
        self.attn_dim = attn_dim
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        self.activation = activation
        self.weight_tie_layers = weight_tie_layers
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.rend_three_views = rend_three_views
        self.use_point_renderer = use_point_renderer
        self.pe_fix = pe_fix
        self.feat_ver = feat_ver
        self.wpt_img_aug = wpt_img_aug
        self.inp_pre_pro = inp_pre_pro
        self.inp_pre_con = inp_pre_con
        self.cvx_up = cvx_up
        self.xops = xops
        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.no_feat = no_feat

        assert renderer is not None
        self.renderer = renderer
        self.num_img = self.renderer.num_img

        # ===============================
        # Derived sizes
        # ===============================
        spatial_size = img_size // img_patch_size

        if add_proprio:
            self.input_dim_before_seq = im_channels * 2
        else:
            self.input_dim_before_seq = im_channels

        # ===============================
        # Positional Encoding
        # ===============================
        lang_emb_dim = lang_dim if add_lang else 0
        lang_max_seq_len = lang_len if add_lang else 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        if pe_fix:
            num_pe_token = spatial_size ** 2 * self.num_img
        else:
            num_pe_token = lang_max_seq_len + spatial_size ** 2 * self.num_img

        self.pos_encoding = nnx.Param(
            jax.random.normal(rngs['params'](), (1, num_pe_token, self.input_dim_before_seq)) * 0.02
        )

        # ===============================
        # Input preprocessing
        # ===============================
        inp_img_feat_dim = img_feat_dim
        if add_corr:
            inp_img_feat_dim += 3
        if add_pixel_loc:
            inp_img_feat_dim += 3
            pixel_loc = jnp.zeros((3, self.num_img, self.img_size, self.img_size))

            pixel_loc = pixel_loc.at[0].set(jnp.linspace(-1, 1, self.num_img)[:, None, None])
            pixel_loc = pixel_loc.at[1].set(jnp.linspace(-1, 1, self.img_size)[None, :, None])
            pixel_loc = pixel_loc.at[2].set(jnp.linspace(-1, 1, self.img_size)[None, None, :])

            self.pixel_loc = pixel_loc

        if add_depth:
            inp_img_feat_dim += 1

        if inp_pre_pro:
            self.input_preprocess = Conv2DBlock(
                inp_img_feat_dim, im_channels,
                kernel_sizes=1, strides=1,
                norm=None, activation=activation,
                rngs=rngs,
            )
            inp_pre_out_dim = im_channels
        else:
            self.input_preprocess = lambda x: x
            inp_pre_out_dim = inp_img_feat_dim

        if add_proprio:
            self.proprio_preprocess = DenseBlock(
                proprio_dim, im_channels,
                norm="group", activation=activation,
                rngs=rngs,
            )

        self.patchify = Conv2DBlock(
            inp_pre_out_dim, im_channels,
            kernel_sizes=img_patch_size,
            strides=img_patch_size,
            norm="group", activation=activation,
            padding=0, rngs=rngs,
        )

        if add_lang:
            self.lang_preprocess = DenseBlock(
                lang_emb_dim, im_channels * 2,
                norm="group", activation=activation,
                rngs=rngs,
            )

        # ===============================
        # Attention blocks
        # ===============================
        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq, attn_dim,
            norm=None, activation=None, rngs=rngs,
        )
        self.fc_aft_attn = DenseBlock(
            attn_dim, self.input_dim_before_seq,
            norm=None, activation=None, rngs=rngs,
        )

        get_attn_attn = lambda: PreNorm(
            attn_dim,
            Attention(attn_dim, heads=attn_heads, dim_head=attn_dim_head,
                      dropout=attn_dropout, rngs=rngs),
            rngs=rngs
        )
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim, rngs=rngs), rngs=rngs)
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))

        self.layers = []
        for _ in range(depth):
            self.layers.append([
                get_attn_attn(_cache=weight_tie_layers),
                get_attn_ff(_cache=weight_tie_layers)
            ])

        # ===============================
        # Upsampling and decoder
        # ===============================
        if cvx_up:
            self.up0 = ConvexUpSample(
                in_dim=self.input_dim_before_seq,
                out_dim=1,
                up_ratio=img_patch_size,
            )
        else:
            self.up0 = Conv2DUpsampleBlock(
                self.input_dim_before_seq, im_channels,
                kernel_sizes=img_patch_size, strides=img_patch_size,
                norm=None, activation=activation, out_size=(img_size, img_size),
                rngs=rngs,
            )

            final_inp_dim = im_channels + inp_pre_out_dim if inp_pre_con else im_channels

            self.final = Conv2DBlock(
                final_inp_dim, im_channels,
                kernel_sizes=3, strides=1,
                norm=None, activation=activation,
                rngs=rngs,
            )

            self.trans_decoder = Conv2DBlock(
                final_dim, 1,
                kernel_sizes=3, strides=1,
                norm=None, activation=None,
                rngs=rngs,
            )

        # ===============================
        # Feature prediction heads
        # ===============================
        if not self.no_feat:
            feat_fc_dim = self.input_dim_before_seq
            feat_fc_dim += self.input_dim_before_seq if cvx_up else final_dim

            def get_feat_fc(inp_dim, out_dim, hid=feat_fc_dim):
                return [
                    nnx.Linear(inp_dim, hid, rngs=rngs), jax.nn.relu,
                    nnx.Linear(hid, hid // 2, rngs=rngs), jax.nn.relu,
                    nnx.Linear(hid // 2, out_dim, rngs=rngs)
                ]

            feat_out_size = feat_dim

            if rot_ver == 0:
                self.feat_fc = get_feat_fc(self.num_img * feat_fc_dim, feat_out_size)
            elif rot_ver == 1:
                assert self.num_rot * 3 <= feat_out_size
                feat_out_size_ex_rot = feat_out_size - self.num_rot * 3

                if feat_out_size_ex_rot > 0:
                    self.feat_fc_ex_rot = get_feat_fc(self.num_img * feat_fc_dim, feat_out_size_ex_rot)

                self.feat_fc_init_bn = nnx.BatchNorm(num_features=self.num_img * feat_fc_dim, rngs=rngs)
                self.feat_fc_pe = FixedPositionalEncoding(
                    feat_per_dim=self.num_img * feat_fc_dim,
                    feat_scale_factor=1,
                )
                self.feat_fc_x = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_y = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_z = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
            else:
                raise ValueError("Invalid rot_ver. Must be 0 or 1.")

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(
        self, img, proprio, lang_emb, wpt_local, rot_x_y
    ):

        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_img, f"num_img {num_img} does not match model's num_img {self.num_img}"
        assert h == w == self.img_size, f"Image size {h} does not match model's img_size {self.img_size}"

        img = jnp.reshape(img, (bs * num_img, img_feat_dim, h, w))
        d0 = self.input_preprocess(img)

        ins = self.patchify(d0)
        ins = jnp.reshape(ins, (bs, num_img, self.im_channels, num_pat_img, num_pat_img))
        ins = jnp.transpose(ins, (0, 2, 1, 3, 4))  # (bs, im_channels, num_img, h', w')

        if self.add_proprio:
            p = self.proprio_preprocess(proprio)
            p = p[:, :, None, None, None]
            p = jnp.broadcast_to(p, (bs, self.im_channels, num_img, num_pat_img, num_pat_img))
            ins = jnp.concatenate([ins, p], axis=1)

        ins = jnp.transpose(ins, (0, 2, 3, 4, 1))
        ins_orig_shape = ins.shape
        ins = jnp.reshape(ins, (bs, -1, ins.shape[-1]))

        if self.pe_fix:
            ins += self.pos_encoding

        num_lang_tok = 0
        if self.add_lang:
            l = self.lang_preprocess(jnp.reshape(lang_emb, (bs * self.lang_max_seq_len, self.lang_emb_dim)))
            l = jnp.reshape(l, (bs, self.lang_max_seq_len, -1))
            num_lang_tok = l.shape[1]
            ins = jnp.concatenate((l, ins), axis=1)

        if not self.pe_fix:
            ins += self.pos_encoding

        x = self.fc_bef_attn(ins)
        if self.self_cross_ver == 0:
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x
        elif self.self_cross_ver == 1:
            lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]
            imgx = jnp.reshape(imgx, (bs * num_img, num_pat_img * num_pat_img, -1))
            for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
                imgx = self_attn(imgx) + imgx
                imgx = self_ff(imgx) + imgx
            imgx = jnp.reshape(imgx, (bs, num_img * num_pat_img * num_pat_img, -1))
            x = jnp.concatenate((lx, imgx), axis=1)
            for self_attn, self_ff in self.layers[len(self.layers) // 2:]:
                x = self_attn(x) + x
                x = self_ff(x) + x
        else:
            raise ValueError

        if self.add_lang:
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)

        x = jnp.reshape(x, (bs, *ins_orig_shape[1:-1], x.shape[-1]))
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        feat = []
        _feat = jnp.max(jnp.max(x, axis=-1), axis=-1)
        _feat = jnp.reshape(_feat, (bs, -1))
        feat.append(_feat)

        x = jnp.transpose(x, (0, 2, 1, 3, 4))
        x = jnp.reshape(x, (bs * self.num_img, self.input_dim_before_seq, num_pat_img, num_pat_img))

        if self.cvx_up:
            trans = self.up0(x)
            trans = jnp.reshape(trans, (bs, self.num_img, h, w))
        else:
            u0 = self.up0(x)
            if self.inp_pre_con:
                u0 = jnp.concatenate([u0, d0], axis=1)
            u = self.final(u0)
            
            trans = self.trans_decoder(u)
            trans = jnp.reshape(trans, (bs, self.num_img, h, w))

        out = {"trans": trans}

        if not self.no_feat and self.feat_ver == 0:
            hm = jnp.reshape(trans, (bs, self.num_img, h * w))
            hm = softmax(hm, axis=-1)
            hm = jnp.reshape(hm, (bs * self.num_img, 1, h, w))

            if self.cvx_up:
                _hm = conv_general_dilated_patches(
                    hm, (self.img_patch_size, self.img_patch_size),
                    window_strides=(self.img_patch_size, self.img_patch_size),
                    padding="VALID"
                )
                _hm = jnp.mean(_hm, axis=-1)
                _hm = jnp.reshape(_hm, (bs * self.num_img, 1, num_pat_img, num_pat_img))
                _u = x
            else:
                _hm = hm
                _u = u

            _feat = jnp.sum(_hm * _u, axis=(2, 3))
            _feat = jnp.reshape(_feat, (bs, -1))
            feat.append(_feat)

            feat = jnp.concatenate(feat, axis=-1)

            if self.rot_ver == 0:
                for layer in self.feat_fc:
                    feat = layer(feat)
                out["feat"] = feat

        return out



def test_mvt_forward():
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)

    # === Configuration ===
    bs = 2
    num_img = 5
    img_size = 128
    img_patch_size = 8
    img_feat_dim = 16
    proprio_dim = 4
    lang_dim = 32
    lang_len = 10
    feat_dim = 20
    im_channels = 64
    final_dim = 64
    attn_dim = 128
    attn_heads = 4
    attn_dim_head = 32

    from renderer import BoxRenderer
    renderer = BoxRenderer(
        device="cuda:0",
        img_size=(img_size, img_size),
        three_views=False,
        with_depth=True,
    )

    model = MVT(
        depth=2,
        img_size=img_size,
        add_proprio=True,
        proprio_dim=proprio_dim,
        add_lang=True,
        lang_dim=lang_dim,
        lang_len=lang_len,
        img_feat_dim=img_feat_dim,
        feat_dim=feat_dim,
        im_channels=im_channels,
        attn_dim=attn_dim,
        attn_heads=attn_heads,
        attn_dim_head=attn_dim_head,
        activation="relu",
        weight_tie_layers=False,
        attn_dropout=0.0,
        decoder_dropout=0.0,
        img_patch_size=img_patch_size,
        final_dim=final_dim,
        self_cross_ver=0,
        add_corr=False,
        norm_corr=False,
        add_pixel_loc=False,
        add_depth=False,
        rend_three_views=False,
        use_point_renderer=False,
        pe_fix=True,
        feat_ver=0,
        wpt_img_aug=False,
        inp_pre_pro=True,
        inp_pre_con=True,
        cvx_up=False,
        xops=False,
        rot_ver=0,
        num_rot=1,
        renderer_device="cuda:0",
        renderer=renderer,
        no_feat=False,
        rngs=rngs,
    )

    # === Inputs ===
    key, img_key, proprio_key, lang_key = jax.random.split(key, 4)
    img = jax.random.normal(img_key, (bs, num_img, img_feat_dim, img_size, img_size))
    proprio = jax.random.normal(proprio_key, (bs, proprio_dim))
    lang_emb = jax.random.normal(lang_key, (bs, lang_len, lang_dim))
    wpt_local = None
    rot_x_y = None

    out = model(img, proprio, lang_emb, wpt_local, rot_x_y)

    # === Assertions ===
    assert isinstance(out, dict), "Output must be a dict"
    assert "trans" in out, "Missing 'trans' in output"
    assert out["trans"].shape == (bs, num_img, img_size, img_size), f"Wrong 'trans' shape: {out['trans'].shape}"

    if "feat" in out:
        assert out["feat"].shape == (bs, feat_dim), f"Wrong 'feat' shape: {out['feat'].shape}"

    print("✅ MVT forward pass successful.")


if __name__ == "__main__":
    test_mvt_forward()

