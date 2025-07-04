# mvt_jax.py

import jax
import jax.numpy as jnp
import copy
import numpy as np
import flax.nnx as nnx

from rvt.mvt.mvt_single_jax import MVT as MVTSingle
from rvt.mvt import utils as mvt_utils
from rvt.mvt.renderer import BoxRenderer  # still PyTorch-based

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
        stage_two,
        st_sca,
        st_wpt_loc_aug,
        st_wpt_loc_inp_no_noise,
        img_aug_2,
        renderer_device="cuda:0",
        rngs=None,
    ):
        self.training = True

        args = copy.deepcopy(locals())
        del args["self"]
        # del args["__class__"]
        del args["stage_two"]
        del args["st_sca"]
        del args["st_wpt_loc_aug"]
        del args["st_wpt_loc_inp_no_noise"]
        del args["img_aug_2"]
        del args["rngs"]

        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.stage_two = stage_two
        self.st_sca = st_sca
        self.st_wpt_loc_aug = st_wpt_loc_aug
        self.st_wpt_loc_inp_no_noise = st_wpt_loc_inp_no_noise
        self.img_aug_2 = img_aug_2

        self.feat_ver = feat_ver
        self.img_feat_dim = img_feat_dim
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang

        if add_lang:
            self.lang_emb_dim = lang_dim
            self.lang_max_seq_len = lang_len
        else:
            self.lang_emb_dim = 0
            self.lang_max_seq_len = 0

        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
        )

        self.num_img = self.renderer.num_img
        self.img_size = img_size

        self.mvt1 = MVTSingle(
            **args,
            renderer=self.renderer,
            no_feat=self.stage_two,
            rngs=rngs,
        )

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def render(self, pc, img_feat, img_aug, mvt1_or_mvt2, dyn_cam_info):
        import torch

        mvt = self.mvt1 if mvt1_or_mvt2 else self.mvt2
        dyn_cam_info_itr = (None,) * len(pc) if dyn_cam_info is None else dyn_cam_info

        img = []
        for _pc, _img_feat, _dyn_cam_info in zip(pc, img_feat, dyn_cam_info_itr):
            if not isinstance(_pc, torch.Tensor):
                _pc = torch.tensor(np.asarray(_pc), dtype=torch.float32).to(self.renderer.device)
            if not isinstance(_img_feat, torch.Tensor):
                _img_feat = torch.tensor(np.asarray(_img_feat), dtype=torch.float32).to(self.renderer.device)

            torch_img = self.renderer(
                _pc,
                _img_feat,
                fix_cam=True,
                dyn_cam_info=(_dyn_cam_info,) if _dyn_cam_info is not None else None,
            )
            img.append(torch_img.unsqueeze(0))

        img = torch.cat(img, dim=0)
        img = img.permute(0, 1, 4, 2, 3).contiguous()
        img = jnp.array(img.cpu().numpy())  # Convert torch -> jax

        if mvt.add_pixel_loc:
            # bs = img.shape[0]
            # pixel_loc = mvt.pixel_loc
            # img = jnp.concatenate(
            #     [img, jnp.broadcast_to(pixel_loc[None], (bs, *pixel_loc.shape))],
            #     axis=2,
            # )
            bs = img.shape[0]

            # Reorder pixel_loc to (num_img, 3, H, W)
            pixel_loc = jnp.transpose(mvt.pixel_loc, (1, 0, 2, 3))

            # Broadcast to (bs, num_img, 3, H, W)
            pixel_loc = jnp.broadcast_to(pixel_loc[None], (bs, *pixel_loc.shape))

            print(f"[render] img.shape: {img.shape}")
            print(f"[render] pixel_loc.shape after transpose+broadcast: {pixel_loc.shape}")

            img = jnp.concatenate([img, pixel_loc], axis=2)


        if img_aug != 0:
            stdv = img_aug * jax.random.uniform(jax.random.PRNGKey(0), ())
            noise = stdv * (2 * jax.random.uniform(jax.random.PRNGKey(1), shape=img.shape) - 1)
            img = jnp.clip(img + noise, -1.0, 1.0)

        return img

    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        img_aug=0,
        wpt_local=None,
        rot_x_y=None,
        dyn_cam_info=None,
    ):
        bs = len(pc)
        if self.training and self.img_aug_2 != 0:
            for i in range(bs):
                stdv = self.img_aug_2 * np.random.rand()
                noise = stdv * (2 * np.random.rand(*img_feat[i].shape) - 1)
                img_feat[i] = img_feat[i] + noise
        
        print(">>> img_feat.shape:", [jf.shape for jf in img_feat])
        img = self.render(pc, img_feat, img_aug, True, dyn_cam_info)
        print(">>> img.shape:", img.shape)

        wpt_local_stage_one = (
            jnp.array(wpt_local) if (self.training and wpt_local is not None) else wpt_local
        )

        out = self.mvt1(
            img=img,
            proprio=proprio,
            lang_emb=lang_emb,
            wpt_local=wpt_local_stage_one,
            rot_x_y=rot_x_y,
        )
        return out

    def get_pt_loc_on_img(self, pt, mvt1_or_mvt2, dyn_cam_info, out=None):
        if mvt1_or_mvt2:
            return self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)
        else:
            assert self.stage_two
            assert out is not None and "wpt_local1" in out
            pt, _ = mvt_utils.trans_pc(pt, loc=out["wpt_local1"], sca=self.st_sca)
            return self.mvt2.get_pt_loc_on_img(pt, dyn_cam_info)

    def get_wpt(self, out, mvt1_or_mvt2, dyn_cam_info, y_q=None):
        if mvt1_or_mvt2:
            return self.mvt1.get_wpt(out, dyn_cam_info, y_q)
        else:
            wpt = self.mvt2.get_wpt(out["mvt2"], dyn_cam_info, y_q)
            return out["rev_trans"](wpt)

    def free_mem(self):
        if not self.use_point_renderer:
            print("Freeing up some memory")
            self.renderer.free_mem()


def test_mvt_forward():
    # Inline config values
    depth = 8
    img_size = 220
    add_proprio = True
    proprio_dim = 4
    add_lang = True
    lang_dim = 512
    lang_len = 77
    img_feat_dim = 3
    feat_dim = (72 * 3) + 2 + 2
    im_channels = 64
    attn_dim = 512
    attn_heads = 8
    attn_dim_head = 64
    activation = "lrelu"
    weight_tie_layers = False
    attn_dropout = 0.1
    decoder_dropout = 0.0
    img_patch_size = 11
    final_dim = 64
    self_cross_ver = 1
    add_corr = True
    norm_corr = False
    add_pixel_loc = True
    add_depth = True
    rend_three_views = False
    use_point_renderer = False
    pe_fix = True
    feat_ver = 0
    wpt_img_aug = 0.01
    inp_pre_pro = True
    inp_pre_con = True
    cvx_up = False
    xops = False
    rot_ver = 0
    num_rot = 72
    stage_two = False
    st_sca = 4
    st_wpt_loc_aug = 0.05
    st_wpt_loc_inp_no_noise = False
    img_aug_2 = 0.0

    rngs = nnx.Rngs(0)

    # Instantiate model
    model = MVT(
        depth=depth,
        img_size=img_size,
        add_proprio=add_proprio,
        proprio_dim=proprio_dim,
        add_lang=add_lang,
        lang_dim=lang_dim,
        lang_len=lang_len,
        img_feat_dim=img_feat_dim,
        feat_dim=feat_dim,
        im_channels=im_channels,
        attn_dim=attn_dim,
        attn_heads=attn_heads,
        attn_dim_head=attn_dim_head,
        activation=activation,
        weight_tie_layers=weight_tie_layers,
        attn_dropout=attn_dropout,
        decoder_dropout=decoder_dropout,
        img_patch_size=img_patch_size,
        final_dim=final_dim,
        self_cross_ver=self_cross_ver,
        add_corr=add_corr,
        norm_corr=norm_corr,
        add_pixel_loc=add_pixel_loc,
        add_depth=add_depth,
        rend_three_views=rend_three_views,
        use_point_renderer=use_point_renderer,
        pe_fix=pe_fix,
        feat_ver=feat_ver,
        wpt_img_aug=wpt_img_aug,
        inp_pre_pro=inp_pre_pro,
        inp_pre_con=inp_pre_con,
        cvx_up=cvx_up,
        xops=xops,
        rot_ver=rot_ver,
        num_rot=num_rot,
        stage_two=stage_two,
        st_sca=st_sca,
        st_wpt_loc_aug=st_wpt_loc_aug,
        st_wpt_loc_inp_no_noise=st_wpt_loc_inp_no_noise,
        img_aug_2=img_aug_2,
        rngs=rngs,
    )

    # Dummy input (batch size = 2, num_points = 10)
    bs, num_pts = 2, 10
    pc = [jnp.zeros((num_pts, 3)) for _ in range(bs)]
    img_feat = [jnp.zeros((num_pts, img_feat_dim)) for _ in range(bs)]
    proprio = jnp.zeros((bs, proprio_dim)) if add_proprio else None
    lang_emb = jnp.zeros((bs, lang_len, lang_dim)) if add_lang else None
    wpt_local = jnp.zeros((bs, 3))

    # Run forward pass
    out = model.forward(
        pc=pc,
        img_feat=img_feat,
        proprio=proprio,
        lang_emb=lang_emb,
        img_aug=0.01,
        wpt_local=wpt_local,
        rot_x_y=None,
        dyn_cam_info=None,
    )

    # Print outputs
    print(">> Forward pass successful")
    if isinstance(out, dict):
        for k, v in out.items():
            if isinstance(v, jnp.ndarray):
                print(f">> out['{k}']: shape = {v.shape}")
            else:
                print(f">> out['{k}']: type = {type(v)}")
    else:
        print(f">> out: type = {type(out)}")

if __name__ == "__main__":
    test_mvt_forward()