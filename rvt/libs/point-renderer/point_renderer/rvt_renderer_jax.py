import point_renderer.ops as ops
import jax.numpy as jnp

def get_pt_loc_on_img_jax(self, pt, fix_cam=False, dyn_cam_info=None):
        """
        returns the location of a point on the image of the cameras
        :param pt: torch.Tensor of shape (bs, np, 3)
        :returns: the location of the pt on the image. this is different from the
            camera screen coordinate system in pytorch3d. the difference is that
            pytorch3d camera screen projects the point to [0, 0] to [H, W]; while the
            index on the img is from [0, 0] to [H-1, W-1]. We verified that
            the to transform from pytorch3d camera screen point to img we have to
            subtract (1/H, 1/W) from the pytorch3d camera screen coordinate.
        :return type: torch.Tensor of shape (bs, np, self.num_img, 2)
        """
        assert len(pt.shape) == 3
        assert pt.shape[-1] == 3
        assert fix_cam, "Not supported with point renderer"
        assert dyn_cam_info is None, "Not supported with point renderer"

        bs, np, _ = pt.shape

        self._check_device(pt, "pt")

        # TODO(Valts): Ask Ankit what what is the bs dimension here, and treat it correctly here

        pcs_px = []
        for i in range(bs):
            pc_px, pc_cam = ops.project_points_3d_to_pixels_jax(
                pt[i], self.cameras.inv_poses, self.cameras.intrinsics, self.cameras.is_orthographic())
            pcs_px.append(pc_px)
        pcs_px = jnp.stack(pcs_px, axis=0)
        pcs_px = jnp.transpose(pcs_px, (0, 2, 1, 3))

        return pcs_px