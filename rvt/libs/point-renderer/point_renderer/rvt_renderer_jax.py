from point_renderer.cameras_jax import OrthographicCameras, PerspectiveCameras

import point_renderer.ops_jax as ops
import jax.numpy as jnp
import jax


class RVTBoxRendererJAX():
    """
    Wrapper around PointRenderer that fixes the cameras to be orthographic cameras
    on the faces of a 2x2x2 cube placed at the origin
    """

    def __init__(
        self,
        img_size,
        radius=0.012,
        default_color=0.0,
        default_depth=-1.0,
        antialiasing_factor=1,
        pers=False,
        normalize_output=True,
        with_depth=True,
        device="cuda",
        perf_timer=False,
        strict_input_device=True,
        no_down=True,
        no_top=False,
        three_views=False,
        two_views=False,
        one_view=False,
        add_3p=False,
        **kwargs):

        self.img_size = img_size
        self.splat_radius = radius
        self.default_color = default_color
        self.default_depth = default_depth
        self.aa_factor = antialiasing_factor
        self.normalize_output = normalize_output
        self.with_depth = with_depth

        self.strict_input_device = strict_input_device

        # Pre-compute fixed cameras ahead of time
        self.cameras = self._get_cube_cameras(
            img_size=self.img_size,
            orthographic=not pers,
            no_down=no_down,
            no_top=no_top,
            three_views=three_views,
            two_views=two_views,
            one_view=one_view,
            add_3p=add_3p,
        )
        # self.cameras = self.cameras.to(device)

        # TODO(Valts): add support for dynamic cameras

        # Cache
        self._fix_pts_cam = None
        self._fix_pts_cam_wei = None
        self._pts = None

        # RVT API (that we might want to refactor)
        self.num_img = len(self.cameras)
        self.only_dyn_cam = False

    def _get_cube_cameras(
        self,
        img_size,
        orthographic,
        no_down,
        no_top,
        three_views,
        two_views,
        one_view,
        add_3p,
    ):
        cam_dict = {
            "top": {"eye": [0, 0, 1], "at": [0, 0, 0], "up": [0, 1, 0]},
            "front": {"eye": [1, 0, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
            "down": {"eye": [0, 0, -1], "at": [0, 0, 0], "up": [0, 1, 0]},
            "back": {"eye": [-1, 0, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
            "left": {"eye": [0, -1, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
            "right": {"eye": [0, 0.5, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
        }

        assert not (two_views and three_views)
        assert not (one_view and three_views)
        assert not (one_view and two_views)
        assert not add_3p, "Not supported with point renderer yet,"
        if two_views or three_views or one_view:
            if no_down or no_top or add_3p:
                print(
                    f"WARNING: when three_views={three_views} or two_views={two_views} -- "
                    f"no_down={no_down} no_top={no_top} add_3p={add_3p} does not matter."
                )

        if three_views:
            cam_names = ["top", "front", "right"]
        elif two_views:
            cam_names = ["top", "front"]
        elif one_view:
            cam_names = ["front"]
        else:
            cam_names = ["top", "front", "down", "back", "left", "right"]
            if no_down:
                # select index of "down" camera and remove it from the list
                del cam_names[cam_names.index("down")]
            if no_top:
                del cam_names[cam_names.index("top")]


        cam_list = [cam_dict[n] for n in cam_names]
        eyes = [c["eye"] for c in cam_list]
        ats = [c["at"] for c in cam_list]
        ups = [c["up"] for c in cam_list]

        if orthographic:
            # img_sizes_w specifies height and width dimensions of the image in world coordinates
            # [2, 2] means it will image coordinates from -1 to 1 in the camera frame
            cameras = OrthographicCameras.from_lookat(eyes, ats, ups, img_sizes_w=[2, 2], img_size_px=img_size)
        else:
            cameras = PerspectiveCameras.from_lookat(eyes, ats, ups, hfov=70, img_size=img_size)
        return cameras

    def get_pt_loc_on_img_jax(self, pt: jnp.ndarray, fix_cam=False, dyn_cam_info=None):
        """
        returns the location of a point on the image of the cameras
        :param pt: jax Tensor of shape (bs, np, 3)
        :returns: the location of the pt on the image.
        :return type: jax Tensor of shape (bs, np, self.num_img, 2)
        """
        assert len(pt.shape) == 3
        assert pt.shape[-1] == 3
        assert fix_cam, "Not supported with point renderer"
        assert dyn_cam_info is None, "Not supported with point renderer"

        bs, np, _ = pt.shape

        # self._check_device(pt, "pt")

        pcs_px = []
        for i in range(bs):
            pc_px, pc_cam = ops.project_points_3d_to_pixels_jax(
                pt[i], self.cameras.inv_poses, self.cameras.intrinsics, self.cameras.is_orthographic())
            pcs_px.append(pc_px)
        pcs_px = jnp.stack(pcs_px, axis=0)
        pcs_px = jnp.transpose(pcs_px, (0, 2, 1, 3))

        return pcs_px

