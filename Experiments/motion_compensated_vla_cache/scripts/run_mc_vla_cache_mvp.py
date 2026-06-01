#!/usr/bin/env python3
"""Run a lightweight MVP validation for motion-compensated VLA-Cache.

The runner intentionally avoids OpenVLA weights. It builds a synthetic RGB-D
rollout with known camera pose, computes oracle 3D patch correspondence, then
compares four cache policies with a small token/KV/action proxy:

1. original_grid: same-grid VLA-Cache style reuse.
2. mc_token: reuse motion-compensated visual tokens, then rebuild current KV.
3. mc_kv_no_rope: remap previous KV without position correction.
4. mc_kv_rope: remap previous KV with RoPE-like key position correction.

This is a smoke-testable bridge between the concept document and a future
OpenVLA integration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is optional for metrics.
    plt = None


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
DEFAULT_CONFIG = EXP_DIR / "configs" / "default.json"


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class Pose:
    rotation_c2w: np.ndarray
    translation_w: np.ndarray


@dataclass(frozen=True)
class SceneObject:
    object_id: int
    center0_xy: Tuple[float, float]
    velocity_xy: Tuple[float, float]
    size_xy: Tuple[float, float]
    z: float
    color: Tuple[float, float, float]
    kind: str


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    seg: np.ndarray
    pose: Pose


@dataclass
class ModelParams:
    token_w: np.ndarray
    token_b: np.ndarray
    layer_w: np.ndarray
    layer_b: np.ndarray
    key_w: np.ndarray
    key_b: np.ndarray
    value_w: np.ndarray
    value_b: np.ndarray
    task_query: np.ndarray
    action_w: np.ndarray
    action_b: np.ndarray
    rope_freq_x: np.ndarray
    rope_freq_y: np.ndarray


@dataclass
class Correspondence:
    prev_patch: np.ndarray
    confidence: np.ndarray
    coverage: np.ndarray
    valid: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MC-VLA-Cache MVP experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--similarity-thresholds", nargs="+", type=float, default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    for key in ["seed", "frames", "height", "width"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.patch_size is not None:
        config["patch_size"] = args.patch_size
    if args.scenarios is not None:
        config["scenarios"] = args.scenarios
    if args.methods is not None:
        config["methods"] = args.methods
    if args.similarity_thresholds is not None:
        config["similarity_thresholds"] = args.similarity_thresholds
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.run_name is not None:
        config["run_name"] = args.run_name
    return config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_run_name() -> str:
    return time.strftime("mc_vla_cache_%Y%m%d_%H%M%S")


def relative_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(pred - target) / max(np.linalg.norm(target), eps))


def cosine_flat(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a_flat = np.asarray(a, dtype=np.float64).reshape(-1)
    b_flat = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom < eps:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return (exp_x / np.sum(exp_x)).astype(np.float32)


def rotation_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    return (rot_y @ rot_x).astype(np.float32)


def make_camera(height: int, width: int) -> Camera:
    focal = 0.92 * float(width)
    return Camera(focal, focal, (width - 1) * 0.5, (height - 1) * 0.5, width, height)


def camera_pose_for(scenario: str, t: int) -> Pose:
    if scenario == "static_cam":
        yaw, pitch = 0.0, 0.0
        trans = (0.0, 0.0, 0.0)
    elif scenario == "pan_tilt":
        yaw, pitch = 0.075 * t, -0.045 * t
        trans = (0.0, 0.0, 0.0)
    elif scenario == "translate_xy":
        yaw, pitch = 0.0, 0.0
        trans = (0.300 * t, -0.200 * t, 0.0)
    elif scenario == "wrist_like":
        yaw, pitch = 0.060 * t, -0.040 * t
        trans = (0.110 * t, 0.050 * math.sin(0.7 * t), -0.040 * t)
    elif scenario == "dynamic_object":
        yaw, pitch = 0.035 * t, 0.0
        trans = (0.070 * t, -0.040 * t, 0.0)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return Pose(rotation_yaw_pitch(yaw, pitch), np.asarray(trans, dtype=np.float32))


def scene_objects_for(scenario: str) -> List[SceneObject]:
    move_a = (0.0, 0.0)
    move_b = (0.0, 0.0)
    if scenario == "dynamic_object":
        move_a = (0.020, -0.008)
        move_b = (-0.015, 0.012)
    return [
        SceneObject(1, (-0.34, -0.16), move_a, (0.22, 0.16), 2.15, (0.92, 0.25, 0.14), "ellipse"),
        SceneObject(2, (0.32, 0.20), move_b, (0.20, 0.18), 2.05, (0.12, 0.50, 0.92), "rect"),
        SceneObject(3, (-0.02, 0.46), (0.0, 0.0), (0.26, 0.08), 1.95, (0.10, 0.70, 0.38), "rect"),
    ]


def ray_grid(camera: Camera) -> np.ndarray:
    yy, xx = np.mgrid[0 : camera.height, 0 : camera.width].astype(np.float32)
    x = (xx - camera.cx) / camera.fx
    y = (yy - camera.cy) / camera.fy
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def world_texture(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = 0.48 + 0.18 * np.sin(22.0 * x + 0.5 * np.cos(17.0 * y))
    g = 0.45 + 0.15 * np.cos(18.0 * y - 0.2 * np.sin(13.0 * x))
    b = 0.50 + 0.16 * np.sin(15.0 * x + 16.0 * y)
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


def object_texture(obj: SceneObject, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    base = np.asarray(obj.color, dtype=np.float32)
    stripe = 0.82 + 0.18 * np.sin(28.0 * x + 17.0 * y + obj.object_id)
    return np.clip(base[None, None, :] * stripe[..., None] + 0.04, 0.0, 1.0).astype(np.float32)


def object_mask(obj: SceneObject, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
    cx = obj.center0_xy[0] + obj.velocity_xy[0] * t
    cy = obj.center0_xy[1] + obj.velocity_xy[1] * t
    sx, sy = obj.size_xy
    if obj.kind == "ellipse":
        return ((x - cx) / max(sx, 1e-6)) ** 2 + ((y - cy) / max(sy, 1e-6)) ** 2 <= 1.0
    if obj.kind == "rect":
        return (np.abs(x - cx) <= sx) & (np.abs(y - cy) <= sy)
    raise ValueError(f"Unknown object kind: {obj.kind}")


def intersect_z_plane(ray_w: np.ndarray, origin_w: np.ndarray, z_plane: float) -> Tuple[np.ndarray, np.ndarray]:
    denom = np.where(np.abs(ray_w[..., 2]) < 1e-6, 1e-6, ray_w[..., 2])
    scale = (z_plane - origin_w[2]) / denom
    points = origin_w[None, None, :] + ray_w * scale[..., None]
    return points.astype(np.float32), scale.astype(np.float32)


def render_frame(camera: Camera, scenario: str, t: int) -> Frame:
    pose = camera_pose_for(scenario, t)
    rays_c = ray_grid(camera)
    rays_w = np.einsum("ij,hwj->hwi", pose.rotation_c2w, rays_c)
    origin = pose.translation_w

    bg_points, bg_depth = intersect_z_plane(rays_w, origin, z_plane=3.0)
    rgb = world_texture(bg_points[..., 0], bg_points[..., 1])
    depth = bg_depth.copy()
    seg = np.zeros((camera.height, camera.width), dtype=np.int32)

    for obj in sorted(scene_objects_for(scenario), key=lambda o: o.z, reverse=True):
        points, obj_depth = intersect_z_plane(rays_w, origin, z_plane=obj.z)
        mask = object_mask(obj, points[..., 0], points[..., 1], t) & (obj_depth > 0.0) & (obj_depth < depth)
        if not np.any(mask):
            continue
        obj_rgb = object_texture(obj, points[..., 0], points[..., 1])
        rgb[mask] = obj_rgb[mask]
        depth[mask] = obj_depth[mask]
        seg[mask] = obj.object_id

    return Frame(rgb=rgb.astype(np.float32), depth=depth.astype(np.float32), seg=seg, pose=pose)


def render_sequence(config: Dict[str, Any], scenario: str) -> List[Frame]:
    camera = make_camera(config["height"], config["width"])
    return [render_frame(camera, scenario, t) for t in range(config["frames"])]


def patch_grid_shape(height: int, width: int, patch_size: int) -> Tuple[int, int, int]:
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("height and width must be divisible by patch_size")
    rows, cols = height // patch_size, width // patch_size
    return rows, cols, rows * cols


def patch_means(array: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = array.shape[:2]
    rows, cols, _ = patch_grid_shape(height, width, patch_size)
    if array.ndim == 2:
        reshaped = array.reshape(rows, patch_size, cols, patch_size)
        return reshaped.mean(axis=(1, 3)).reshape(rows * cols, 1)
    channels = array.shape[2]
    reshaped = array.reshape(rows, patch_size, cols, patch_size, channels)
    return reshaped.mean(axis=(1, 3)).reshape(rows * cols, channels)


def patch_stds(array: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = array.shape[:2]
    rows, cols, _ = patch_grid_shape(height, width, patch_size)
    channels = array.shape[2]
    reshaped = array.reshape(rows, patch_size, cols, patch_size, channels)
    return reshaped.std(axis=(1, 3)).reshape(rows * cols, channels)


def patch_rgb_vectors(rgb: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    rows, cols, _ = patch_grid_shape(height, width, patch_size)
    patches = rgb.reshape(rows, patch_size, cols, patch_size, 3)
    return patches.transpose(0, 2, 1, 3, 4).reshape(rows * cols, patch_size * patch_size * 3)


def patch_seg_hist(seg: np.ndarray, patch_size: int, num_ids: int = 4) -> np.ndarray:
    height, width = seg.shape
    rows, cols, num_patches = patch_grid_shape(height, width, patch_size)
    out = np.zeros((num_patches, num_ids), dtype=np.float32)
    patch_id = 0
    for row in range(rows):
        for col in range(cols):
            block = seg[row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size]
            for seg_id in range(num_ids):
                out[patch_id, seg_id] = float(np.mean(block == seg_id))
            patch_id += 1
    return out


def extract_patch_features(frame: Frame, patch_size: int) -> np.ndarray:
    mean_rgb = patch_means(frame.rgb, patch_size)
    std_rgb = patch_stds(frame.rgb, patch_size)
    mean_depth = patch_means(frame.depth, patch_size)
    seg_hist = patch_seg_hist(frame.seg, patch_size)
    return np.concatenate([mean_rgb, std_rgb, mean_depth / 3.0, seg_hist], axis=1).astype(np.float32)


def patch_positions(height: int, width: int, patch_size: int) -> np.ndarray:
    rows, cols, _ = patch_grid_shape(height, width, patch_size)
    pos = []
    for row in range(rows):
        for col in range(cols):
            x = col / max(cols - 1, 1)
            y = row / max(rows - 1, 1)
            pos.append((x, y))
    return np.asarray(pos, dtype=np.float32)


def make_model_params(config: Dict[str, Any], raw_dim: int) -> ModelParams:
    rng = np.random.default_rng(int(config["seed"]) + 1009)
    token_dim = int(config["token_dim"])
    kv_dim = int(config["kv_dim"])
    if kv_dim % 2 != 0:
        raise ValueError("kv_dim must be even for the RoPE proxy")
    action_dim = int(config["action_dim"])
    num_layers = int(config["num_layers"])

    def normal(shape: Sequence[int], scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    token_w = normal((raw_dim, token_dim), 1.0 / math.sqrt(raw_dim))
    token_b = normal((token_dim,), 0.03)
    layer_w = normal((num_layers, token_dim, token_dim), 1.0 / math.sqrt(token_dim))
    layer_b = normal((num_layers, token_dim), 0.02)
    key_w = normal((num_layers, token_dim, kv_dim), 1.0 / math.sqrt(token_dim))
    key_b = normal((num_layers, kv_dim), 0.02)
    value_w = normal((num_layers, token_dim, kv_dim), 1.0 / math.sqrt(token_dim))
    value_b = normal((num_layers, kv_dim), 0.02)
    task_query = normal((kv_dim,), 3.0 / math.sqrt(kv_dim))
    action_w = normal((kv_dim, action_dim), 1.0 / math.sqrt(kv_dim))
    action_b = normal((action_dim,), 0.02)

    pairs = kv_dim // 2
    rope_freq_x = np.linspace(3.0, 14.0, pairs, dtype=np.float32)
    rope_freq_y = np.linspace(4.0, 17.0, pairs, dtype=np.float32)
    return ModelParams(
        token_w,
        token_b,
        layer_w,
        layer_b,
        key_w,
        key_b,
        value_w,
        value_b,
        task_query,
        action_w,
        action_b,
        rope_freq_x,
        rope_freq_y,
    )


def encode_tokens(features: np.ndarray, params: ModelParams) -> np.ndarray:
    return np.tanh(features @ params.token_w + params.token_b).astype(np.float32)


def apply_rope(keys: np.ndarray, positions: np.ndarray, params: ModelParams, inverse: bool = False) -> np.ndarray:
    out = keys.copy().astype(np.float32)
    angles = positions[:, [0]] * params.rope_freq_x[None, :] + positions[:, [1]] * params.rope_freq_y[None, :]
    if inverse:
        angles = -angles
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    even = out[:, 0::2].copy()
    odd = out[:, 1::2].copy()
    out[:, 0::2] = even * cos_a - odd * sin_a
    out[:, 1::2] = even * sin_a + odd * cos_a
    return out


def rope_correct_keys(prev_keys: np.ndarray, prev_positions: np.ndarray, curr_positions: np.ndarray, params: ModelParams) -> np.ndarray:
    base = apply_rope(prev_keys, prev_positions, params, inverse=True)
    return apply_rope(base, curr_positions, params, inverse=False)


def compute_kv(tokens: np.ndarray, positions: np.ndarray, params: ModelParams) -> Tuple[np.ndarray, np.ndarray]:
    hidden = tokens.astype(np.float32)
    all_keys = []
    all_values = []
    for layer_idx in range(params.layer_w.shape[0]):
        hidden = np.tanh(hidden @ params.layer_w[layer_idx] + params.layer_b[layer_idx]).astype(np.float32)
        base_key = hidden @ params.key_w[layer_idx] + params.key_b[layer_idx]
        value = hidden @ params.value_w[layer_idx] + params.value_b[layer_idx]
        key = apply_rope(base_key, positions, params)
        all_keys.append(key.astype(np.float32))
        all_values.append(value.astype(np.float32))
    return np.stack(all_keys, axis=0), np.stack(all_values, axis=0)


def action_from_kv(keys: np.ndarray, values: np.ndarray, params: ModelParams) -> Tuple[np.ndarray, np.ndarray]:
    last_key = keys[-1]
    last_value = values[-1]
    logits = last_key @ params.task_query / math.sqrt(last_key.shape[1])
    attn = softmax(logits)
    context = attn @ last_value
    action = context @ params.action_w + params.action_b
    return action.astype(np.float32), attn.astype(np.float32)


def backproject_to_world(frame: Frame, camera: Camera) -> np.ndarray:
    yy, xx = np.mgrid[0 : camera.height, 0 : camera.width].astype(np.float32)
    z = frame.depth
    x = (xx - camera.cx) / camera.fx * z
    y = (yy - camera.cy) / camera.fy * z
    points_c = np.stack([x, y, z], axis=-1)
    return np.einsum("ij,hwj->hwi", frame.pose.rotation_c2w, points_c) + frame.pose.translation_w


def project_world_to_camera(points_w: np.ndarray, frame: Frame, camera: Camera) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points_w - frame.pose.translation_w
    points_c = np.einsum("ji,hwj->hwi", frame.pose.rotation_c2w, centered)
    z = points_c[..., 2]
    u = camera.fx * (points_c[..., 0] / np.maximum(z, 1e-6)) + camera.cx
    v = camera.fy * (points_c[..., 1] / np.maximum(z, 1e-6)) + camera.cy
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)


def bilinear_sample_scalar(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = np.clip(x, 0.0, width - 1.0)
    y = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = x - x0
    wy = y - y0
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def oracle_3d_correspondence(
    prev_frame: Frame,
    curr_frame: Frame,
    camera: Camera,
    patch_size: int,
    depth_tolerance: float,
    min_conf: float,
    min_coverage: float,
) -> Correspondence:
    height, width = curr_frame.rgb.shape[:2]
    rows, cols, num_patches = patch_grid_shape(height, width, patch_size)
    world_points = backproject_to_world(curr_frame, camera)
    prev_u, prev_v, prev_z = project_world_to_camera(world_points, prev_frame, camera)
    prev_x = np.rint(prev_u).astype(np.int32)
    prev_y = np.rint(prev_v).astype(np.int32)
    valid = (
        (prev_z > 0.0)
        & (prev_x >= 0)
        & (prev_x < width)
        & (prev_y >= 0)
        & (prev_y < height)
    )
    clipped_x = np.clip(prev_x, 0, width - 1)
    clipped_y = np.clip(prev_y, 0, height - 1)
    sampled_prev_depth = bilinear_sample_scalar(prev_frame.depth, prev_u, prev_v)
    depth_ok = np.abs(sampled_prev_depth - prev_z) <= depth_tolerance
    valid &= depth_ok

    prev_patch_pix = (clipped_y // patch_size) * cols + (clipped_x // patch_size)
    prev_patch = np.full((num_patches,), -1, dtype=np.int32)
    confidence = np.zeros((num_patches,), dtype=np.float32)
    coverage = np.zeros((num_patches,), dtype=np.float32)
    valid_patch = np.zeros((num_patches,), dtype=bool)

    for row in range(rows):
        for col in range(cols):
            patch_id = row * cols + col
            y0, y1 = row * patch_size, (row + 1) * patch_size
            x0, x1 = col * patch_size, (col + 1) * patch_size
            patch_valid = valid[y0:y1, x0:x1]
            if not np.any(patch_valid):
                continue
            votes = prev_patch_pix[y0:y1, x0:x1][patch_valid]
            counts = np.bincount(votes, minlength=num_patches)
            best = int(np.argmax(counts))
            valid_count = int(np.sum(patch_valid))
            cover = float(valid_count) / float(patch_size * patch_size)
            conf = float(counts[best]) / float(max(valid_count, 1))
            prev_patch[patch_id] = best
            confidence[patch_id] = conf
            coverage[patch_id] = cover
            valid_patch[patch_id] = (conf >= min_conf) and (cover >= min_coverage)
    return Correspondence(prev_patch=prev_patch, confidence=confidence, coverage=coverage, valid=valid_patch)


def patch_cosines(current_rgb: np.ndarray, prev_rgb: np.ndarray, patch_size: int) -> np.ndarray:
    curr = patch_rgb_vectors(current_rgb, patch_size)
    prev = patch_rgb_vectors(prev_rgb, patch_size)
    curr_norm = np.linalg.norm(curr, axis=1)
    prev_norm = np.linalg.norm(prev, axis=1)
    return np.sum(curr * prev, axis=1) / np.maximum(curr_norm * prev_norm, 1e-12)


def motion_compensated_patch_cosines(
    current_rgb: np.ndarray,
    prev_rgb: np.ndarray,
    patch_size: int,
    corr: Correspondence,
) -> np.ndarray:
    curr = patch_rgb_vectors(current_rgb, patch_size)
    prev = patch_rgb_vectors(prev_rgb, patch_size)
    sims = np.full((curr.shape[0],), -1.0, dtype=np.float32)
    for patch_id, source_id in enumerate(corr.prev_patch):
        if source_id < 0:
            continue
        sims[patch_id] = cosine_flat(curr[patch_id], prev[source_id])
    return sims


def task_veto_mask(attention: np.ndarray, ratio: float) -> np.ndarray:
    num_patches = attention.shape[0]
    count = int(round(num_patches * ratio))
    veto = np.zeros((num_patches,), dtype=bool)
    if count <= 0:
        return veto
    top_idx = np.argsort(-attention)[:count]
    veto[top_idx] = True
    return veto


def blend_kv_for_reuse(
    curr_keys: np.ndarray,
    curr_values: np.ndarray,
    prev_keys: np.ndarray,
    prev_values: np.ndarray,
    corr: Correspondence,
    selected: np.ndarray,
    params: ModelParams,
    positions: np.ndarray,
    early_recompute_layers: int,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mixed_keys = curr_keys.copy()
    mixed_values = curr_values.copy()
    layer_start = min(max(early_recompute_layers, 0), curr_keys.shape[0])
    selected_idx = np.flatnonzero(selected)
    if selected_idx.size == 0 or layer_start >= curr_keys.shape[0]:
        return mixed_keys, mixed_values

    if mode == "original_grid":
        source_idx = selected_idx
    else:
        source_idx = corr.prev_patch[selected_idx]

    for layer_idx in range(layer_start, curr_keys.shape[0]):
        if mode == "mc_kv_rope":
            corrected = rope_correct_keys(
                prev_keys[layer_idx, source_idx],
                positions[source_idx],
                positions[selected_idx],
                params,
            )
            mixed_keys[layer_idx, selected_idx] = corrected
        else:
            mixed_keys[layer_idx, selected_idx] = prev_keys[layer_idx, source_idx]
        mixed_values[layer_idx, selected_idx] = prev_values[layer_idx, source_idx]
    return mixed_keys, mixed_values


def key_cosine_on_selected(
    mixed_keys: np.ndarray,
    teacher_keys: np.ndarray,
    selected: np.ndarray,
    early_recompute_layers: int,
) -> float:
    idx = np.flatnonzero(selected)
    if idx.size == 0:
        return float("nan")
    start = min(max(early_recompute_layers, 0), mixed_keys.shape[0] - 1)
    scores = []
    for layer_idx in range(start, mixed_keys.shape[0]):
        scores.append(cosine_flat(mixed_keys[layer_idx, idx], teacher_keys[layer_idx, idx]))
    return float(np.mean(scores))


def evaluate_pair(
    config: Dict[str, Any],
    scenario: str,
    pair_idx: int,
    prev_frame: Frame,
    curr_frame: Frame,
    params: ModelParams,
    camera: Camera,
    positions: np.ndarray,
) -> List[Dict[str, Any]]:
    patch_size = int(config["patch_size"])
    thresholds = [float(x) for x in config["similarity_thresholds"]]
    methods = list(config["methods"])
    early_layers = int(config["early_recompute_layers"])
    min_conf = float(config["mc_min_conf"])
    min_coverage = float(config.get("mc_min_coverage", 0.0))
    depth_tol = float(config["depth_tolerance"])
    task_ratio = float(config["task_veto_ratio"])
    mc_similarity_threshold = float(config.get("mc_similarity_threshold", 0.0))

    prev_features = extract_patch_features(prev_frame, patch_size)
    curr_features = extract_patch_features(curr_frame, patch_size)
    prev_tokens = encode_tokens(prev_features, params)
    curr_tokens = encode_tokens(curr_features, params)
    prev_keys, prev_values = compute_kv(prev_tokens, positions, params)
    curr_keys, curr_values = compute_kv(curr_tokens, positions, params)
    teacher_action, teacher_attention = action_from_kv(curr_keys, curr_values, params)
    veto = task_veto_mask(teacher_attention, task_ratio)

    corr = oracle_3d_correspondence(prev_frame, curr_frame, camera, patch_size, depth_tol, min_conf, min_coverage)
    same_grid_sim = patch_cosines(curr_frame.rgb, prev_frame.rgb, patch_size)
    mc_sim = motion_compensated_patch_cosines(curr_frame.rgb, prev_frame.rgb, patch_size, corr)
    same_grid_valid = corr.valid & (corr.prev_patch == np.arange(corr.prev_patch.shape[0]))
    oracle_available = corr.valid & (~veto)

    rows = []
    num_patches = curr_tokens.shape[0]
    for threshold in thresholds:
        original_selected = (same_grid_sim >= threshold) & (~veto)
        mc_selected = corr.valid & (mc_sim >= mc_similarity_threshold) & (~veto)

        for method in methods:
            if method == "original_grid":
                selected = original_selected
                mixed_keys, mixed_values = blend_kv_for_reuse(
                    curr_keys,
                    curr_values,
                    prev_keys,
                    prev_values,
                    corr,
                    selected,
                    params,
                    positions,
                    early_layers,
                    mode=method,
                )
                action, _ = action_from_kv(mixed_keys, mixed_values, params)
                false_reuse = selected & (~same_grid_valid)
                key_cos = key_cosine_on_selected(mixed_keys, curr_keys, selected, early_layers)
                saving = float(np.mean(selected)) * (curr_keys.shape[0] - early_layers) / curr_keys.shape[0]
            elif method == "mc_token":
                selected = mc_selected
                mixed_tokens = curr_tokens.copy()
                idx = np.flatnonzero(selected)
                if idx.size > 0:
                    mixed_tokens[idx] = prev_tokens[corr.prev_patch[idx]]
                mixed_keys, mixed_values = compute_kv(mixed_tokens, positions, params)
                action, _ = action_from_kv(mixed_keys, mixed_values, params)
                false_reuse = selected & (~corr.valid)
                key_cos = key_cosine_on_selected(mixed_keys, curr_keys, selected, early_layers)
                saving = 0.10 * float(np.mean(selected))
            elif method in {"mc_kv_no_rope", "mc_kv_rope"}:
                selected = mc_selected
                mixed_keys, mixed_values = blend_kv_for_reuse(
                    curr_keys,
                    curr_values,
                    prev_keys,
                    prev_values,
                    corr,
                    selected,
                    params,
                    positions,
                    early_layers,
                    mode=method,
                )
                action, _ = action_from_kv(mixed_keys, mixed_values, params)
                false_reuse = selected & (~corr.valid)
                key_cos = key_cosine_on_selected(mixed_keys, curr_keys, selected, early_layers)
                saving = float(np.mean(selected)) * (curr_keys.shape[0] - early_layers) / curr_keys.shape[0]
            else:
                raise ValueError(f"Unknown method: {method}")

            reuse_count = int(np.sum(selected))
            false_count = int(np.sum(false_reuse))
            rows.append(
                {
                    "scenario": scenario,
                    "pair": pair_idx,
                    "threshold": threshold,
                    "method": method,
                    "num_patches": num_patches,
                    "reuse_count": reuse_count,
                    "reuse_ratio": float(reuse_count / num_patches),
                    "false_reuse_count": false_count,
                    "false_reuse_rate": float(false_count / max(reuse_count, 1)),
                    "oracle_available_ratio": float(np.mean(oracle_available)),
                    "original_valid_same_grid_ratio": float(np.mean(same_grid_valid & (~veto))),
                    "action_rel_l2": relative_l2(action, teacher_action),
                    "reused_key_cosine": key_cos,
                    "estimated_saving": saving,
                }
            )
    return rows


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["scenario"]), float(row["threshold"]), str(row["method"]))
        groups.setdefault(key, []).append(row)

    summary = []
    fields = [
        "reuse_ratio",
        "false_reuse_rate",
        "oracle_available_ratio",
        "original_valid_same_grid_ratio",
        "action_rel_l2",
        "reused_key_cosine",
        "estimated_saving",
    ]
    for (scenario, threshold, method), items in sorted(groups.items()):
        out: Dict[str, Any] = {"scenario": scenario, "threshold": threshold, "method": method, "pairs": len(items)}
        for field in fields:
            values = np.asarray([float(x[field]) for x in items if not math.isnan(float(x[field]))], dtype=np.float64)
            out[f"{field}_mean"] = float(np.mean(values)) if values.size else float("nan")
            out[f"{field}_std"] = float(np.std(values)) if values.size else float("nan")
        summary.append(out)
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_best_lines(summary: List[Dict[str, Any]]) -> List[str]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in summary:
        by_method.setdefault(str(row["method"]), []).append(row)
    lines = []
    for method, items in sorted(by_method.items()):
        action = np.mean([float(x["action_rel_l2_mean"]) for x in items])
        reuse = np.mean([float(x["reuse_ratio_mean"]) for x in items])
        false = np.mean([float(x["false_reuse_rate_mean"]) for x in items])
        saving = np.mean([float(x["estimated_saving_mean"]) for x in items])
        key_values = [float(x["reused_key_cosine_mean"]) for x in items if not math.isnan(float(x["reused_key_cosine_mean"]))]
        key_cos = float(np.mean(key_values)) if key_values else float("nan")
        lines.append(
            f"| {method} | {reuse:.3f} | {false:.3f} | {action:.4f} | {key_cos:.3f} | {saving:.3f} |"
        )
    return lines


def overlay_patch_mask(image: np.ndarray, mask: np.ndarray, patch_size: int, color: Tuple[float, float, float]) -> np.ndarray:
    out = image.copy()
    height, width = image.shape[:2]
    rows, cols, _ = patch_grid_shape(height, width, patch_size)
    color_arr = np.asarray(color, dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            patch_id = row * cols + col
            if not mask[patch_id]:
                continue
            y0, y1 = row * patch_size, (row + 1) * patch_size
            x0, x1 = col * patch_size, (col + 1) * patch_size
            out[y0:y1, x0:x1] = 0.55 * out[y0:y1, x0:x1] + 0.45 * color_arr
    return np.clip(out, 0.0, 1.0)


def make_debug_figure(
    path: Path,
    config: Dict[str, Any],
    scenario: str,
    pair_idx: int,
    prev_frame: Frame,
    curr_frame: Frame,
    corr: Correspondence,
    veto: np.ndarray,
    threshold: float,
) -> None:
    if plt is None:
        return
    patch_size = int(config["patch_size"])
    same_sim = patch_cosines(curr_frame.rgb, prev_frame.rgb, patch_size)
    mc_sim = motion_compensated_patch_cosines(curr_frame.rgb, prev_frame.rgb, patch_size, corr)
    original_selected = (same_sim >= threshold) & (~veto)
    mc_selected = corr.valid & (mc_sim >= threshold) & (~veto)
    original_img = overlay_patch_mask(curr_frame.rgb, original_selected, patch_size, (1.0, 0.25, 0.15))
    mc_img = overlay_patch_mask(curr_frame.rgb, mc_selected, patch_size, (0.1, 0.75, 0.25))
    veto_img = overlay_patch_mask(curr_frame.rgb, veto, patch_size, (0.15, 0.35, 1.0))

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    panels = [
        (prev_frame.rgb, "prev RGB"),
        (curr_frame.rgb, "curr RGB"),
        (original_img, "original same-grid reuse"),
        (mc_img, "3D MC reuse"),
        (veto_img, "task veto"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{scenario}, pair={pair_idx}, threshold={threshold}", fontsize=10)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(
    path: Path,
    config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
    elapsed: float,
) -> None:
    ensure_dir(path.parent)
    lines = [
        "# Motion-Compensated VLA-Cache MVP Report",
        "",
        f"- Run name: `{config['run_name']}`",
        f"- Scenarios: `{', '.join(config['scenarios'])}`",
        f"- Frames per scenario: `{config['frames']}`",
        f"- Image size: `{config['height']}x{config['width']}`",
        f"- Patch size: `{config['patch_size']}`",
        f"- Pair rows: `{len(rows)}`",
        f"- Runtime: `{elapsed:.2f}s`",
        "",
        "## Aggregate By Method",
        "",
        "| method | reuse_ratio | false_reuse_rate | action_rel_l2 | reused_key_cosine | estimated_saving |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(read_best_lines(summary))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `original_grid` reuses the previous KV at the same patch index, matching the failure mode of grid-level cache under camera ego-motion.",
            "- `mc_token` remaps visual tokens but rebuilds KV at the current position; it is a correspondence-quality diagnostic rather than the main speed path.",
            "- `mc_kv_no_rope` directly moves previous KV from source patch to target patch.",
            "- `mc_kv_rope` additionally applies a RoPE-like key position correction from source slot to target slot.",
            "",
            "Generated files:",
            "",
            "- `metrics/pair_metrics.csv`",
            "- `metrics/summary_by_method.csv`",
            "- `figures/debug_correspondence.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_debug_data(
    config: Dict[str, Any],
    frames: List[Frame],
    params: ModelParams,
    camera: Camera,
    positions: np.ndarray,
) -> Tuple[Frame, Frame, Correspondence, np.ndarray]:
    pair_idx = min(int(config["debug_pair"]), len(frames) - 1)
    prev_frame = frames[pair_idx - 1]
    curr_frame = frames[pair_idx]
    patch_size = int(config["patch_size"])
    curr_features = extract_patch_features(curr_frame, patch_size)
    curr_tokens = encode_tokens(curr_features, params)
    curr_keys, curr_values = compute_kv(curr_tokens, positions, params)
    _, attention = action_from_kv(curr_keys, curr_values, params)
    veto = task_veto_mask(attention, float(config["task_veto_ratio"]))
    corr = oracle_3d_correspondence(
        prev_frame,
        curr_frame,
        camera,
        patch_size,
        float(config["depth_tolerance"]),
        float(config["mc_min_conf"]),
        float(config.get("mc_min_coverage", 0.0)),
    )
    return prev_frame, curr_frame, corr, veto


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    if "run_name" not in config or not config["run_name"]:
        config["run_name"] = current_run_name()

    np.random.seed(int(config["seed"]))
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = EXP_DIR / output_root.relative_to("Experiments/motion_compensated_vla_cache") if str(output_root).startswith("Experiments/motion_compensated_vla_cache") else Path.cwd() / output_root
    output_dir = ensure_dir(output_root / str(config["run_name"]))
    metrics_dir = ensure_dir(output_dir / "metrics")
    figures_dir = ensure_dir(output_dir / "figures")

    camera = make_camera(int(config["height"]), int(config["width"]))
    positions = patch_positions(int(config["height"]), int(config["width"]), int(config["patch_size"]))

    dummy_frame = render_frame(camera, str(config["scenarios"][0]), 0)
    raw_dim = extract_patch_features(dummy_frame, int(config["patch_size"])).shape[1]
    params = make_model_params(config, raw_dim)

    start = time.time()
    all_rows: List[Dict[str, Any]] = []
    debug_payload: Optional[Tuple[str, int, Frame, Frame, Correspondence, np.ndarray]] = None

    for scenario in config["scenarios"]:
        frames = render_sequence(config, str(scenario))
        for pair_idx in range(1, len(frames)):
            all_rows.extend(
                evaluate_pair(
                    config,
                    str(scenario),
                    pair_idx,
                    frames[pair_idx - 1],
                    frames[pair_idx],
                    params,
                    camera,
                    positions,
                )
            )
        if scenario == config.get("debug_scenario"):
            prev_f, curr_f, corr, veto = prepare_debug_data(config, frames, params, camera, positions)
            debug_pair = min(int(config["debug_pair"]), len(frames) - 1)
            debug_payload = (str(scenario), debug_pair, prev_f, curr_f, corr, veto)

    if debug_payload is None:
        scenario = str(config["scenarios"][0])
        frames = render_sequence(config, scenario)
        prev_f, curr_f, corr, veto = prepare_debug_data(config, frames, params, camera, positions)
        debug_payload = (scenario, min(int(config["debug_pair"]), len(frames) - 1), prev_f, curr_f, corr, veto)

    summary = summarize(all_rows)
    elapsed = time.time() - start

    write_csv(metrics_dir / "pair_metrics.csv", all_rows)
    write_csv(metrics_dir / "summary_by_method.csv", summary)

    threshold = float(config["similarity_thresholds"][0])
    scenario, pair_idx, prev_f, curr_f, corr, veto = debug_payload
    make_debug_figure(figures_dir / "debug_correspondence.png", config, scenario, pair_idx, prev_f, curr_f, corr, veto, threshold)
    write_report(output_dir / "motion_compensated_vla_cache_report.md", config, all_rows, summary, elapsed)

    print(f"[ok] wrote metrics to {metrics_dir}")
    print(f"[ok] wrote report to {output_dir / 'motion_compensated_vla_cache_report.md'}")
    if plt is not None:
        print(f"[ok] wrote debug figure to {figures_dir / 'debug_correspondence.png'}")
    print("[summary]")
    for line in read_best_lines(summary):
        print(line)


if __name__ == "__main__":
    main()
