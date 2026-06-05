"""Oracle 3D patch correspondences for LIBERO OpenVLA KV studies.

The OpenVLA evaluation path rotates raw LIBERO camera images by 180 degrees,
resizes them to 224x224, and may apply the same center crop used at eval time.
This module keeps that image-space transform explicit: correspondences are
reported in OpenVLA visual patch coordinates, while depth and camera matrices
remain in the raw simulator camera coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_camera_transform_matrix,
    get_real_depth_map,
)


@dataclass
class CameraFrameGeometry:
    """Geometry needed to project between raw camera pixels and world points."""

    camera_name: str
    raw_height: int
    raw_width: int
    model_height: int
    model_width: int
    world_to_pixel: np.ndarray
    pixel_to_world: np.ndarray
    real_depth: np.ndarray
    center_crop: bool
    crop_scale: float


@dataclass
class OracleCorrespondence:
    """Current OpenVLA patch ids and their previous-frame source patch ids."""

    target_patches: List[int]
    source_patches: List[int]
    confidence: List[float]
    depth_error: List[float]
    num_candidates_before_filter: int

    def source_for_target(self) -> Dict[int, int]:
        return dict(zip(self.target_patches, self.source_patches))


def capture_camera_geometry(
    sim,
    camera_name: str,
    raw_depth: np.ndarray,
    model_height: int = 224,
    model_width: int = 224,
    center_crop: bool = False,
    crop_scale: float = 0.9,
) -> CameraFrameGeometry:
    """Build camera geometry from a LIBERO/robosuite simulator state."""

    depth = np.asarray(raw_depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape [H, W] or [H, W, 1], got {raw_depth.shape}")

    raw_height, raw_width = depth.shape
    world_to_pixel = get_camera_transform_matrix(sim, camera_name, raw_height, raw_width)
    pixel_to_world = np.linalg.inv(world_to_pixel)
    real_depth = get_real_depth_map(sim, depth)

    return CameraFrameGeometry(
        camera_name=camera_name,
        raw_height=raw_height,
        raw_width=raw_width,
        model_height=model_height,
        model_width=model_width,
        world_to_pixel=world_to_pixel,
        pixel_to_world=pixel_to_world,
        real_depth=real_depth,
        center_crop=center_crop,
        crop_scale=crop_scale,
    )


def raw_xy_to_model_xy(
    raw_xy: np.ndarray,
    raw_height: int,
    raw_width: int,
    model_height: int = 224,
    model_width: int = 224,
    center_crop: bool = False,
    crop_scale: float = 0.9,
) -> np.ndarray:
    """Map raw LIBERO pixel xy coordinates to OpenVLA model-image xy coordinates."""

    raw_xy = np.asarray(raw_xy, dtype=np.float64)
    rotated_x = raw_width - 1 - raw_xy[..., 0]
    rotated_y = raw_height - 1 - raw_xy[..., 1]

    model_x = (rotated_x + 0.5) * model_width / raw_width - 0.5
    model_y = (rotated_y + 0.5) * model_height / raw_height - 0.5

    if center_crop:
        crop = math.sqrt(crop_scale)
        crop_w = model_width * crop
        crop_h = model_height * crop
        x0 = 0.5 * (model_width - crop_w)
        y0 = 0.5 * (model_height - crop_h)
        model_x = (model_x - x0 + 0.5) * model_width / crop_w - 0.5
        model_y = (model_y - y0 + 0.5) * model_height / crop_h - 0.5

    return np.stack([model_x, model_y], axis=-1)


def model_xy_to_raw_xy(
    model_xy: np.ndarray,
    raw_height: int,
    raw_width: int,
    model_height: int = 224,
    model_width: int = 224,
    center_crop: bool = False,
    crop_scale: float = 0.9,
) -> np.ndarray:
    """Map OpenVLA model-image xy coordinates back to raw LIBERO camera xy."""

    model_xy = np.asarray(model_xy, dtype=np.float64)
    x = model_xy[..., 0]
    y = model_xy[..., 1]

    if center_crop:
        crop = math.sqrt(crop_scale)
        crop_w = model_width * crop
        crop_h = model_height * crop
        x0 = 0.5 * (model_width - crop_w)
        y0 = 0.5 * (model_height - crop_h)
        x = x0 + (x + 0.5) * crop_w / model_width - 0.5
        y = y0 + (y + 0.5) * crop_h / model_height - 0.5

    rotated_x = (x + 0.5) * raw_width / model_width - 0.5
    rotated_y = (y + 0.5) * raw_height / model_height - 0.5
    raw_x = raw_width - 1 - rotated_x
    raw_y = raw_height - 1 - rotated_y
    return np.stack([raw_x, raw_y], axis=-1)


def make_patch_centers(model_height: int = 224, model_width: int = 224, patch_size: int = 14) -> np.ndarray:
    """Return xy patch centers for the OpenVLA model-image grid."""

    if model_height % patch_size != 0 or model_width % patch_size != 0:
        raise ValueError("Model image size must be divisible by patch_size")
    rows = model_height // patch_size
    cols = model_width // patch_size
    centers = []
    for row in range(rows):
        for col in range(cols):
            centers.append((col * patch_size + (patch_size - 1) / 2.0, row * patch_size + (patch_size - 1) / 2.0))
    return np.asarray(centers, dtype=np.float64)


def patch_ids_from_model_xy(
    model_xy: np.ndarray,
    model_height: int = 224,
    model_width: int = 224,
    patch_size: int = 14,
) -> np.ndarray:
    """Convert model-image xy coordinates to flattened patch ids, or -1 if out of bounds."""

    xy = np.asarray(model_xy, dtype=np.float64)
    rows = model_height // patch_size
    cols = model_width // patch_size
    x = xy[..., 0]
    y = xy[..., 1]
    valid = (x >= 0) & (y >= 0) & (x < model_width) & (y < model_height)
    patch_col = np.floor(x / patch_size).astype(np.int64)
    patch_row = np.floor(y / patch_size).astype(np.int64)
    patch_id = patch_row * cols + patch_col
    patch_id = np.where(valid, patch_id, -1)
    return patch_id.astype(np.int64)


def _nearest_depth(depth: np.ndarray, raw_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.rint(raw_xy[..., 0]).astype(np.int64)
    y = np.rint(raw_xy[..., 1]).astype(np.int64)
    valid = (x >= 0) & (y >= 0) & (x < depth.shape[1]) & (y < depth.shape[0])
    x_clip = np.clip(x, 0, depth.shape[1] - 1)
    y_clip = np.clip(y, 0, depth.shape[0] - 1)
    return depth[y_clip, x_clip], valid


def _pixel_depth_to_world(raw_xy: np.ndarray, depth: np.ndarray, pixel_to_world: np.ndarray) -> np.ndarray:
    xy = np.asarray(raw_xy, dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)
    camera_points = np.stack([xy[..., 0] * z, xy[..., 1] * z, z, np.ones_like(z)], axis=-1)
    world = camera_points @ pixel_to_world.T
    return world[..., :3]


def _project_world_to_raw_xy(world_points: np.ndarray, world_to_pixel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(world_points, dtype=np.float64)
    homo = np.concatenate([points, np.ones(points.shape[:-1] + (1,), dtype=np.float64)], axis=-1)
    projected = homo @ world_to_pixel.T
    z = projected[..., 2]
    xy = projected[..., :2] / np.maximum(z[..., None], 1e-8)
    return xy, z


def oracle_3d_patch_correspondence(
    prev: CameraFrameGeometry,
    curr: CameraFrameGeometry,
    patch_size: int = 14,
    depth_tolerance_m: float = 0.04,
) -> OracleCorrespondence:
    """Backproject current patch centers and reproject them into the previous frame."""

    centers = make_patch_centers(curr.model_height, curr.model_width, patch_size)
    curr_raw_xy = model_xy_to_raw_xy(
        centers,
        curr.raw_height,
        curr.raw_width,
        curr.model_height,
        curr.model_width,
        curr.center_crop,
        curr.crop_scale,
    )
    curr_depth, curr_depth_valid = _nearest_depth(curr.real_depth, curr_raw_xy)
    world_points = _pixel_depth_to_world(curr_raw_xy, curr_depth, curr.pixel_to_world)
    prev_raw_xy, prev_projected_depth = _project_world_to_raw_xy(world_points, prev.world_to_pixel)
    prev_depth, prev_depth_valid = _nearest_depth(prev.real_depth, prev_raw_xy)
    prev_model_xy = raw_xy_to_model_xy(
        prev_raw_xy,
        prev.raw_height,
        prev.raw_width,
        prev.model_height,
        prev.model_width,
        prev.center_crop,
        prev.crop_scale,
    )
    source_patches = patch_ids_from_model_xy(prev_model_xy, prev.model_height, prev.model_width, patch_size)

    depth_error = np.abs(prev_depth - prev_projected_depth)
    valid = curr_depth_valid & prev_depth_valid & (source_patches >= 0) & (depth_error <= depth_tolerance_m)

    target_ids = np.nonzero(valid)[0].astype(np.int64)
    return OracleCorrespondence(
        target_patches=target_ids.tolist(),
        source_patches=source_patches[valid].astype(np.int64).tolist(),
        confidence=np.ones(int(np.sum(valid)), dtype=np.float64).tolist(),
        depth_error=depth_error[valid].astype(np.float64).tolist(),
        num_candidates_before_filter=int(centers.shape[0]),
    )


def identity_patch_correspondence(
    model_height: int = 224,
    model_width: int = 224,
    patch_size: int = 14,
) -> OracleCorrespondence:
    """Identity correspondence for self-frame S0 sanity checks."""

    n_patches = (model_height // patch_size) * (model_width // patch_size)
    patch_ids = list(range(n_patches))
    return OracleCorrespondence(
        target_patches=patch_ids,
        source_patches=patch_ids,
        confidence=[1.0] * n_patches,
        depth_error=[0.0] * n_patches,
        num_candidates_before_filter=n_patches,
    )

