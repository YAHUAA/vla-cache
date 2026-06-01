"""Motion-compensation helpers for OpenVLA VLA-Cache experiments.

This module is intentionally lightweight. The first OpenVLA+LIBERO migration
uses RGB-only global 2D translation compensation because the current LIBERO
evaluation path only passes RGB frames into ``get_vla_action``. The API keeps
the correspondence representation explicit so a later depth/pose oracle can
replace the estimator without changing the cache-selection call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class PatchCorrespondence:
    """Current-patch to previous-patch correspondence."""

    target_patches: List[int]
    source_patches: List[int]
    confidence: List[float]
    similarity: List[float]
    shift_xy: Tuple[int, int]
    score: float
    num_candidates_before_topk: int

    def source_for_target(self) -> Dict[int, int]:
        return dict(zip(self.target_patches, self.source_patches))


def _to_rgb_float(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    return arr


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _overlap_slices(height: int, width: int, dx: int, dy: int) -> Optional[Tuple[slice, slice, slice, slice]]:
    """Return prev/current slices for a prev->current integer shift."""

    if dx >= 0:
        prev_x = slice(0, width - dx)
        curr_x = slice(dx, width)
    else:
        prev_x = slice(-dx, width)
        curr_x = slice(0, width + dx)

    if dy >= 0:
        prev_y = slice(0, height - dy)
        curr_y = slice(dy, height)
    else:
        prev_y = slice(-dy, height)
        curr_y = slice(0, height + dy)

    if (prev_x.stop - prev_x.start) <= 0 or (prev_y.stop - prev_y.start) <= 0:
        return None
    return prev_y, prev_x, curr_y, curr_x


def estimate_global_translation(
    prev_image: Image.Image,
    curr_image: Image.Image,
    search_radius: int = 28,
    step: int = 2,
) -> Tuple[int, int, float]:
    """Estimate integer prev->current translation with normalized MSE search."""

    prev_gray = _rgb_to_gray(_to_rgb_float(prev_image))
    curr_gray = _rgb_to_gray(_to_rgb_float(curr_image))
    height, width = prev_gray.shape
    if curr_gray.shape != prev_gray.shape:
        raise ValueError("prev_image and curr_image must have the same size")

    best_shift = (0, 0)
    best_score = float("inf")
    step = max(int(step), 1)
    radius = max(int(search_radius), 0)

    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            slices = _overlap_slices(height, width, dx, dy)
            if slices is None:
                continue
            prev_y, prev_x, curr_y, curr_x = slices
            prev_crop = prev_gray[prev_y, prev_x]
            curr_crop = curr_gray[curr_y, curr_x]
            if prev_crop.size < 64:
                continue
            diff = prev_crop - curr_crop
            score = float(np.mean(diff * diff))
            if score < best_score:
                best_score = score
                best_shift = (dx, dy)

    return best_shift[0], best_shift[1], best_score


def _patch_vectors(rgb: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by patch_size")
    rows = height // patch_size
    cols = width // patch_size
    patches = rgb.reshape(rows, patch_size, cols, patch_size, 3)
    return patches.transpose(0, 2, 1, 3, 4).reshape(rows * cols, patch_size * patch_size * 3)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _translation_patch_vote(
    patch_id: int,
    rows: int,
    cols: int,
    patch_size: int,
    dx: int,
    dy: int,
    samples_per_axis: int = 5,
) -> Tuple[int, float]:
    """Map a current patch to a previous patch by voting sampled points."""

    row, col = divmod(patch_id, cols)
    offsets = np.linspace(0.5, patch_size - 0.5, samples_per_axis, dtype=np.float32)
    votes: List[int] = []
    for oy in offsets:
        for ox in offsets:
            curr_x = col * patch_size + ox
            curr_y = row * patch_size + oy
            prev_x = curr_x - dx
            prev_y = curr_y - dy
            if prev_x < 0 or prev_y < 0 or prev_x >= cols * patch_size or prev_y >= rows * patch_size:
                continue
            prev_col = int(prev_x // patch_size)
            prev_row = int(prev_y // patch_size)
            votes.append(prev_row * cols + prev_col)

    if not votes:
        return -1, 0.0

    counts = np.bincount(np.asarray(votes, dtype=np.int32), minlength=rows * cols)
    source = int(np.argmax(counts))
    confidence = float(counts[source]) / float(samples_per_axis * samples_per_axis)
    return source, confidence


def find_motion_compensated_static_patches(
    curr_image: Image.Image,
    prev_image: Image.Image,
    patch_size: int = 14,
    top_k: int = 130,
    search_radius: int = 28,
    search_step: int = 2,
    min_confidence: float = 0.30,
    sim_threshold: float = 0.70,
    samples_per_axis: int = 5,
) -> PatchCorrespondence:
    """Find reusable current patches after global 2D motion compensation.

    Returns current patch IDs and their source patch IDs from the previous frame.
    ``sim_threshold`` is intentionally lower than the original same-grid 0.996
    threshold because a motion-compensated target patch can straddle old patch
    boundaries even when the world content is static.
    """

    curr_rgb = _to_rgb_float(curr_image)
    prev_rgb = _to_rgb_float(prev_image)
    if curr_rgb.shape != prev_rgb.shape:
        raise ValueError("curr_image and prev_image must have the same size")

    height, width = curr_rgb.shape[:2]
    rows = height // patch_size
    cols = width // patch_size
    num_patches = rows * cols
    dx, dy, shift_score = estimate_global_translation(prev_image, curr_image, search_radius, search_step)

    curr_vectors = _patch_vectors(curr_rgb, patch_size)
    prev_vectors = _patch_vectors(prev_rgb, patch_size)
    candidates = []

    for patch_id in range(num_patches):
        source_id, confidence = _translation_patch_vote(
            patch_id, rows, cols, patch_size, dx, dy, samples_per_axis=samples_per_axis
        )
        if source_id < 0 or confidence < min_confidence:
            continue
        similarity = _cosine(curr_vectors[patch_id], prev_vectors[source_id])
        if similarity < sim_threshold:
            continue
        combined_score = confidence * max(similarity, 0.0)
        candidates.append((patch_id, source_id, confidence, similarity, combined_score))

    candidates.sort(key=lambda item: item[-1], reverse=True)
    selected = candidates[: max(int(top_k), 0)]
    return PatchCorrespondence(
        target_patches=[int(item[0]) for item in selected],
        source_patches=[int(item[1]) for item in selected],
        confidence=[float(item[2]) for item in selected],
        similarity=[float(item[3]) for item in selected],
        shift_xy=(int(dx), int(dy)),
        score=float(shift_score),
        num_candidates_before_topk=len(candidates),
    )


def draw_motion_compensated_overlay(
    image: Image.Image,
    patch_ids: Sequence[int],
    patch_size: int = 14,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (46, 204, 113),
) -> Image.Image:
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cols = image.size[0] // patch_size
    for patch_id in patch_ids:
        row, col = divmod(int(patch_id), cols)
        top_left = (col * patch_size, row * patch_size)
        bottom_right = ((col + 1) * patch_size, (row + 1) * patch_size)
        draw.rectangle([top_left, bottom_right], fill=color + (int(255 * alpha),))
    return Image.alpha_composite(image, overlay).convert("RGB")


def remap_visual_kv_cache(
    past_key_values,
    target_token_indices: Iterable[int],
    source_token_indices: Iterable[int],
) -> bool:
    """Best-effort in-place visual KV slot remap for HuggingFace DynamicCache.

    The VLA-Cache transformers fork stores per-layer tensors under
    ``key_cache`` and ``value_cache``. If the runtime cache does not expose
    those attributes, this function returns ``False`` and the caller should use
    mask-only reuse. RoPE correction is intentionally not attempted here because
    the generic cache object does not expose the model's rotary embedding state.
    """

    if past_key_values is None:
        return False
    if not hasattr(past_key_values, "key_cache") or not hasattr(past_key_values, "value_cache"):
        return False

    target = list(int(x) for x in target_token_indices)
    source = list(int(x) for x in source_token_indices)
    if not target or len(target) != len(source):
        return False

    try:
        import torch

        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
        for layer_idx in range(len(key_cache)):
            key = key_cache[layer_idx]
            value = value_cache[layer_idx]
            if key is None or value is None:
                continue
            seq_dim = 2
            if key.shape[seq_dim] <= max(max(target), max(source)):
                continue
            source_tensor = torch.as_tensor(source, dtype=torch.long, device=key.device)
            target_tensor = torch.as_tensor(target, dtype=torch.long, device=key.device)
            key_src = key.index_select(seq_dim, source_tensor).clone()
            value_src = value.index_select(seq_dim, source_tensor).clone()
            key.index_copy_(seq_dim, target_tensor, key_src)
            value.index_copy_(seq_dim, target_tensor, value_src)
    except Exception:
        return False

    return True
