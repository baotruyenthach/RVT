from point_renderer import ops_jax as ops
import jax
import jax.numpy as jnp

def linalg_inv(poses):
    return jnp.linalg.inv(poses)

class Cameras:
    def __init__(self, poses, intrinsics, img_size, inv_poses=None):
        self.poses = poses
        self.img_size = img_size
        if inv_poses is None:
            self.inv_poses = linalg_inv(poses)
        else:
            self.inv_poses = inv_poses
        self.intrinsics = intrinsics

    def __len__(self):
        return len(self.poses)

    def scale(self, constant):
        self.intrinsics = self.intrinsics.clone()
        self.intrinsics[:, :2, :3] *= constant

    def is_orthographic(self):
        raise ValueError("is_orthographic should be called on child classes only")

    def is_perspective(self):
        raise ValueError("is_perspective should be called on child classes only")


class PerspectiveCameras(Cameras):
    def __init__(self, poses, intrinsics, img_size, inv_poses=None):
        super().__init__(poses, intrinsics, img_size, inv_poses)

    @classmethod
    def from_lookat(cls, eyes, ats, ups, hfov, img_size, device="cpu"):
        cam_poses = []
        for eye, at, up in zip(eyes, ats, ups):
            T = ops.lookat_to_cam_pose_jax(eye, at, up, device=device)
            cam_poses.append(T)
        cam_poses = jnp.stack(cam_poses, dim=0)
        intrinsics = ops.fov_and_size_to_intrinsics_jax(hfov, img_size, device=device)
        intrinsics = intrinsics[None, :, :].repeat((cam_poses.shape[0], 1, 1)).contiguous()
        return PerspectiveCameras(cam_poses, intrinsics, img_size)

    def is_orthographic(self):
        return False

    def is_perspective(self):
        return True

class OrthographicCameras(Cameras):
    def __init__(self, poses, intrinsics, img_size, inv_poses=None):
        super().__init__(poses, intrinsics, img_size, inv_poses)

    @classmethod
    def from_lookat(cls, eyes, ats, ups, img_sizes_w, img_size_px, device="cpu"):
        """
        Args:
            eyes: Nx3 tensor of camera coordinates
            ats: Nx3 tensor of look-at directions
            ups: Nx3 tensor of up-vectors
            scale: Nx2 tensor defining image sizes in world coordinates
            img_size: 2-dim tuple defining image size in pixels
        Returns:
            OrthographicCamera
        """
        if isinstance(img_sizes_w, list):
            img_sizes_w = jnp.array(img_sizes_w)[None, :].repeat((len(eyes), 1))

        cam_poses = []
        for eye, at, up in zip(eyes, ats, ups):
            T = ops.lookat_to_cam_pose_jax(eye, at, up, device=device)
            cam_poses.append(T)
        cam_poses = jnp.stack(cam_poses, dim=0)
        intrinsics = ops.orthographic_intrinsics_from_scales_jax(img_sizes_w, img_size_px, device=device)
        return OrthographicCameras(cam_poses, intrinsics, img_size_px)

    def is_orthographic(self):
        return True

    def is_perspective(self):
        return False
