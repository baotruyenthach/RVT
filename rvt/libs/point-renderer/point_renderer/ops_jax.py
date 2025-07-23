# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple
import math


def transform_points_batch_jax(pc: jnp.ndarray, inv_cam_poses: jnp.ndarray) -> jnp.ndarray:
    pc_h = jnp.concatenate([pc, jnp.ones_like(pc[:, 0:1])], axis=1)
    pc_cam_h = jnp.einsum("bxy,ny->bnx", inv_cam_poses, pc_h)
    pc_cam = pc_cam_h[:, :, :3]
    return pc_cam

def orthographic_camera_projection_batch_jax(pc_cam: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    # For orthographic camera projection, treat all points as if they are at depth 1
    ones = jnp.ones_like(pc_cam[:, :, 2:3])
    homog_coords = jnp.concatenate([pc_cam[:, :, :2], ones], axis=2)
    uvZ = jnp.einsum("bxy,bny->bnx", K, homog_coords)
    return uvZ[:, :, :2]

def perspective_camera_projection_batch_jax(pc_cam: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    uvZ = jnp.einsum("bxy,bny->bnx", K, pc_cam)
    uv = jnp.stack([
        uvZ[:, :, 0] / uvZ[:, :, 2],
        uvZ[:, :, 1] / uvZ[:, :, 2]
    ], axis=2)
    return uv

def project_points_3d_to_pixels_jax(pc : jnp.ndarray, inv_cam_poses : jnp.ndarray, 
                                    intrinsics : jnp.ndarray, orthographic : bool):
    """
    This combines the projection from 3D coordinates to camera coordinates using extrinsics,
    followed by projection from camera coordinates to pixel coordinates using the intrinsics.
    """
    # Project points from world to camera frame
    pc_cam = transform_points_batch_jax(pc, inv_cam_poses)
    # Project points from camera frame to pixel space
    if orthographic:
        pc_px = orthographic_camera_projection_batch_jax(pc_cam, intrinsics)
    else:
        pc_px = perspective_camera_projection_batch_jax(pc_cam, intrinsics)
    return pc_px

def lookat_to_cam_pose_jax(eye, at, up=[0, 0, 1]):
    """
    Compute camera-to-world pose matrix from eye, at, and up vectors using JAX.

    Args:
        eye: (3,) array-like — camera position
        at: (3,) array-like — look-at target
        up: (3,) array-like — up direction (default: [0, 0, 1])

    Returns:
        (4, 4) jnp.ndarray camera-to-world transformation matrix
    """
    eye = jnp.asarray(eye, dtype=jnp.float32)
    at = jnp.asarray(at, dtype=jnp.float32)
    up = jnp.asarray(up, dtype=jnp.float32)

    # camera forward vector
    camera_view = at - eye
    camera_view = camera_view / jnp.linalg.norm(camera_view)

    # camera right vector
    camera_right = jnp.cross(camera_view, up)
    camera_right = camera_right / jnp.linalg.norm(camera_right)

    # true up vector
    camera_up = jnp.cross(camera_right, camera_view)
    camera_up = camera_up / jnp.linalg.norm(camera_up)

    # Rotation matrix (OpenCV convention: right, -up, view)
    R = jnp.stack([camera_right, -camera_up, camera_view], axis=1)  # shape (3, 3)

    # Compose 4x4 transform
    T = jnp.eye(4, dtype=jnp.float32)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(eye)

    return T

def fov_and_size_to_intrinsics_jax(fov: float, img_size: Tuple[int, int]) -> jnp.ndarray:
    """
    Compute a 3x3 camera intrinsic matrix from a field of view and image size using JAX.

    Args:
        fov: float, field of view in degrees (assumed to be vertical FOV)
        img_size: (height, width) tuple

    Returns:
        3x3 jnp.ndarray camera intrinsic matrix
    """
    img_h, img_w = img_size
    f = img_h / (2.0 * math.tan(math.radians(fov) / 2))  # vertical FOV

    fx = f
    fy = f
    cx = img_w / 2.0
    cy = img_h / 2.0

    intrinsics = jnp.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=jnp.float32)

    return intrinsics

def orthographic_intrinsics_from_scales_jax(
    img_sizes_w: jnp.ndarray,  # shape (B, 2)
    img_size_px: Tuple[int, int]
) -> jnp.ndarray:
    """
    Create a batch of orthographic camera intrinsics from world-space size and image resolution.

    Args:
        img_sizes_w: (B, 2) array — real-world width and height of the image (in meters, etc.)
        img_size_px: (H, W) — image resolution in pixels

    Returns:
        (B, 3, 3) batch of intrinsics matrices
    """
    img_h, img_w = img_size_px
    fx = img_h / img_sizes_w[:, 0]  # fx = H / real-world height
    fy = img_w / img_sizes_w[:, 1]  # fy = W / real-world width

    B = img_sizes_w.shape[0]
    intrinsics = jnp.zeros((B, 3, 3), dtype=jnp.float32)
    intrinsics = intrinsics.at[:, 0, 0].set(fx)
    intrinsics = intrinsics.at[:, 1, 1].set(fy)
    intrinsics = intrinsics.at[:, 0, 2].set(img_h / 2)
    intrinsics = intrinsics.at[:, 1, 2].set(img_w / 2)
    # Optional: intrinsics = intrinsics.at[:, 2, 2].set(1.0)

    return intrinsics
