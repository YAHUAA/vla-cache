"""Run the OpenVLA KV contextualization layer-wise study on LIBERO frames.

This script implements the first executable pass of
Docs/OpenVLA_KV_Contextualization_Layerwise_Study_Design.md:

- S0 self-frame sanity and floor controls.
- Grid A: prompt x patch_semantics layer-wise K/V/H cosine.
- Grid B: episode_phase x patch_semantics layer-wise K/V/H cosine.

Large outputs default to /mnt/data0/zjh_data as required by the project rules;
repo-local outputs are limited to lightweight reports and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
OPENVLA_ROOT = REPO_ROOT / "src" / "openvla"
if str(OPENVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENVLA_ROOT))

from libero.libero import benchmark  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.libero.oracle_correspondence import (  # noqa: E402
    CameraFrameGeometry,
    OracleCorrespondence,
    capture_camera_geometry,
    identity_patch_correspondence,
    oracle_3d_patch_correspondence,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    OPENVLA_V01_SYSTEM_PROMPT,
    get_processor,
    process_image,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


PATCH_SIZE = 14
MODEL_IMAGE_SIZE = 224
NUM_VISUAL_PATCHES = 256
PROMPT_TOKEN = 29871


CSV_FIELDS = [
    "run_id",
    "task_suite",
    "task_id",
    "task_name",
    "episode",
    "step",
    "camera",
    "grid",
    "control",
    "prompt_condition",
    "phase",
    "phase_reason",
    "target_name",
    "target_eef_dist",
    "prev_target_eef_dist",
    "gripper_qpos",
    "prev_gripper_qpos",
    "gripper_delta",
    "patch_semantics",
    "modality",
    "layer",
    "sim_mean",
    "n_pairs",
    "n_oracle_pairs",
    "n_target_patches",
    "n_background_patches",
    "wrong_prompt",
]


PHASE_DIAGNOSTIC_FIELDS = [
    "run_id",
    "task_suite",
    "task_id",
    "task_name",
    "episode",
    "step",
    "camera",
    "phase",
    "phase_reason",
    "target_name",
    "target_eef_dist",
    "prev_target_eef_dist",
    "gripper_qpos",
    "prev_gripper_qpos",
    "gripper_delta",
    "gripper_open",
    "object_in_hand",
    "eef_x",
    "eef_y",
    "eef_z",
    "target_x",
    "target_y",
    "target_z",
]


@dataclass
class FrameSample:
    """A sampled LIBERO frame plus OpenVLA-aligned metadata."""

    obs: Mapping[str, np.ndarray]
    image: np.ndarray
    seg_model: np.ndarray
    geometry: CameraFrameGeometry
    step: int


@dataclass(frozen=True)
class PhaseInfo:
    phase: str
    reason: str
    target_name: str
    target_eef_dist: Optional[float]
    prev_target_eef_dist: Optional[float]
    gripper_qpos: float
    prev_gripper_qpos: float
    gripper_delta: float
    gripper_open: bool
    object_in_hand: bool
    eef_pos: Tuple[Optional[float], Optional[float], Optional[float]]
    target_pos: Tuple[Optional[float], Optional[float], Optional[float]]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def build_prompt(base_vla_name: str, task_label: str) -> str:
    if "openvla-v01" in base_vla_name:
        return f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def append_prompt_token(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.all(input_ids[:, -1] == PROMPT_TOKEN):
        return input_ids, attention_mask
    token = torch.full((input_ids.shape[0], 1), PROMPT_TOKEN, dtype=input_ids.dtype, device=input_ids.device)
    mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
    return torch.cat([input_ids, token], dim=1), torch.cat([attention_mask, mask], dim=1)


def teacher_forward(
    model,
    processor,
    prompt: str,
    image_array: np.ndarray,
    center_crop: bool,
    device: torch.device,
):
    """Run a clean multimodal forward and keep visual-token cache/hidden states."""

    if hasattr(model, "language_model"):
        model.language_model.config.proportion_attn_var = None
        model.language_model.config.reusable_patches = None

    image = Image.fromarray(image_array).convert("RGB")
    if center_crop:
        image = process_image(image)
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    input_ids, attention_mask = append_prompt_token(inputs["input_ids"], inputs["attention_mask"])
    with torch.inference_mode():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs["pixel_values"],
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )


def cache_layer(past_key_values, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    return past_key_values[layer_idx][0], past_key_values[layer_idx][1]


def num_cache_layers(past_key_values) -> int:
    if hasattr(past_key_values, "key_cache"):
        return len(past_key_values.key_cache)
    return len(past_key_values)


def cosine_layer_means(output_a, output_b, source_patches: Sequence[int], target_patches: Sequence[int]) -> Dict[str, List[float]]:
    """Compute per-layer mean cosine for K, V, and hidden states."""

    if len(source_patches) != len(target_patches):
        raise ValueError("source_patches and target_patches must have the same length")
    if not source_patches:
        n_layers = num_cache_layers(output_a.past_key_values)
        return {"K": [float("nan")] * n_layers, "V": [float("nan")] * n_layers, "H": [float("nan")] * n_layers}

    device = output_a.hidden_states[0].device
    source_tokens = torch.as_tensor(np.asarray(source_patches, dtype=np.int64) + 1, device=device, dtype=torch.long)
    target_tokens = torch.as_tensor(np.asarray(target_patches, dtype=np.int64) + 1, device=device, dtype=torch.long)

    out: Dict[str, List[float]] = {"K": [], "V": [], "H": []}
    n_layers = num_cache_layers(output_a.past_key_values)
    for layer_idx in range(n_layers):
        key_a, value_a = cache_layer(output_a.past_key_values, layer_idx)
        key_b, value_b = cache_layer(output_b.past_key_values, layer_idx)

        ka = key_a[0, :, source_tokens, :].permute(1, 0, 2).reshape(len(source_patches), -1).float()
        kb = key_b[0, :, target_tokens, :].permute(1, 0, 2).reshape(len(target_patches), -1).float()
        va = value_a[0, :, source_tokens, :].permute(1, 0, 2).reshape(len(source_patches), -1).float()
        vb = value_b[0, :, target_tokens, :].permute(1, 0, 2).reshape(len(target_patches), -1).float()
        ha = output_a.hidden_states[layer_idx + 1][0, source_tokens, :].float()
        hb = output_b.hidden_states[layer_idx + 1][0, target_tokens, :].float()

        out["K"].append(float(torch.nn.functional.cosine_similarity(ka, kb, dim=1).mean().detach().cpu()))
        out["V"].append(float(torch.nn.functional.cosine_similarity(va, vb, dim=1).mean().detach().cpu()))
        out["H"].append(float(torch.nn.functional.cosine_similarity(ha, hb, dim=1).mean().detach().cpu()))
    return out


def preprocess_segmentation_to_model(seg_raw: np.ndarray, center_crop: bool, crop_scale: float) -> np.ndarray:
    """Apply the OpenVLA image-space transform to an instance segmentation mask."""

    seg = np.asarray(seg_raw)
    if seg.ndim == 3:
        seg = seg[..., 0]
    seg = seg[::-1, ::-1].astype(np.int32)
    image = Image.fromarray(seg, mode="I")
    image = image.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), resample=Image.Resampling.NEAREST)
    if center_crop:
        crop = math.sqrt(crop_scale)
        crop_w = MODEL_IMAGE_SIZE * crop
        crop_h = MODEL_IMAGE_SIZE * crop
        x0 = int(round(0.5 * (MODEL_IMAGE_SIZE - crop_w)))
        y0 = int(round(0.5 * (MODEL_IMAGE_SIZE - crop_h)))
        x1 = int(round(x0 + crop_w))
        y1 = int(round(y0 + crop_h))
        image = image.crop((x0, y0, x1, y1)).resize(
            (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), resample=Image.Resampling.NEAREST
        )
    return np.asarray(image, dtype=np.int64)


def get_base_env(env):
    return getattr(env, "env", env)


def get_instance_id_sets(env, target_interest_index: int) -> Tuple[Dict[str, int], List[int], List[int], List[str]]:
    base = get_base_env(env)
    names = list(base.model.instances_to_ids.keys())
    instance_to_id = {name: idx + 1 for idx, name in enumerate(names)}
    robot_ids = [
        seg_id
        for name, seg_id in instance_to_id.items()
        if "Panda" in name or "Gripper" in name or "Mount" in name
    ]
    interest = list(getattr(base, "obj_of_interest", []))
    if not interest:
        target_names = []
    else:
        index = max(0, min(target_interest_index, len(interest) - 1))
        target_names = [interest[index]]
    target_ids = [instance_to_id[name] for name in target_names if name in instance_to_id]
    return instance_to_id, target_ids, robot_ids, target_names


def label_patch_semantics(
    seg_model: np.ndarray,
    target_ids: Sequence[int],
    robot_ids: Sequence[int],
    threshold: float = 0.5,
    background_mode: str = "all_non_target",
) -> Dict[str, List[int]]:
    rows = seg_model.shape[0] // PATCH_SIZE
    cols = seg_model.shape[1] // PATCH_SIZE
    target_set = set(int(x) for x in target_ids)
    robot_set = set(int(x) for x in robot_ids)

    labels = {"target": [], "background": []}
    for row in range(rows):
        for col in range(cols):
            patch_id = row * cols + col
            patch = seg_model[row * PATCH_SIZE : (row + 1) * PATCH_SIZE, col * PATCH_SIZE : (col + 1) * PATCH_SIZE]
            flat = patch.reshape(-1)
            total = float(flat.size)
            target_frac = float(np.isin(flat, list(target_set)).sum()) / total if target_set else 0.0
            robot_frac = float(np.isin(flat, list(robot_set)).sum()) / total if robot_set else 0.0
            if target_frac >= threshold and robot_frac < 0.05:
                labels["target"].append(patch_id)
                continue
            if robot_frac >= 0.05:
                continue
            if background_mode == "zero_only":
                background_frac = float((flat == 0).sum()) / total
            else:
                excluded = target_set | robot_set
                background_frac = float((~np.isin(flat, list(excluded))).sum()) / total if excluded else 1.0
            if background_frac >= threshold:
                labels["background"].append(patch_id)
    return labels


def obs_vector(obs: Mapping[str, np.ndarray], key: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if key not in obs:
        return (None, None, None)
    value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
    if value.size < 3:
        return (None, None, None)
    return (float(value[0]), float(value[1]), float(value[2]))


def target_distance_and_pos(
    obs: Mapping[str, np.ndarray],
    target_name: Optional[str],
) -> Tuple[Optional[float], Tuple[Optional[float], Optional[float], Optional[float]]]:
    if not target_name:
        return None, (None, None, None)
    rel_key = f"{target_name}_to_robot0_eef_pos"
    pos_key = f"{target_name}_pos"
    if rel_key in obs:
        rel = np.asarray(obs[rel_key], dtype=np.float64).reshape(-1)
        if rel.size >= 3:
            dist = float(np.linalg.norm(rel[:3]))
            eef = obs_vector(obs, "robot0_eef_pos")
            if all(value is not None for value in eef):
                target = tuple(float(eef[idx] + rel[idx]) for idx in range(3))
            else:
                target = (None, None, None)
            return dist, target
    if pos_key in obs and "robot0_eef_pos" in obs:
        target = np.asarray(obs[pos_key], dtype=np.float64).reshape(-1)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(-1)
        if target.size >= 3 and eef.size >= 3:
            return float(np.linalg.norm(target[:3] - eef[:3])), (
                float(target[0]),
                float(target[1]),
                float(target[2]),
            )
    return None, (None, None, None)


def infer_phase(
    obs: Mapping[str, np.ndarray],
    prev_obs: Optional[Mapping[str, np.ndarray]],
    target_name: Optional[str],
    eef_target_distance_threshold: float,
    gripper_open_threshold: float,
    gripper_delta_threshold: float,
) -> PhaseInfo:
    qpos = float(np.mean(obs.get("robot0_gripper_qpos", np.asarray([0.0]))))
    prev_qpos = qpos
    if prev_obs is not None:
        prev_qpos = float(np.mean(prev_obs.get("robot0_gripper_qpos", np.asarray([qpos]))))
    delta = qpos - prev_qpos
    dist, target_pos = target_distance_and_pos(obs, target_name)
    prev_dist = None
    if prev_obs is not None:
        prev_dist, _ = target_distance_and_pos(prev_obs, target_name)
    gripper_open = qpos >= gripper_open_threshold
    object_in_hand = dist is not None and dist <= eef_target_distance_threshold and not gripper_open
    phase = "unknown"
    reason = "insufficient_object_signal"

    if delta <= -gripper_delta_threshold:
        phase = "grasp"
        reason = "gripper_closing_transition"
    elif delta >= gripper_delta_threshold:
        phase = "place"
        reason = "gripper_opening_transition"
    elif (
        prev_dist is not None
        and dist is not None
        and prev_dist > eef_target_distance_threshold
        and dist <= eef_target_distance_threshold
    ):
        phase = "grasp"
        reason = "eef_entered_target_radius"
    elif dist is not None and dist > eef_target_distance_threshold:
        phase = "reach"
        reason = "eef_far_from_target"
    elif object_in_hand:
        phase = "transport"
        reason = "eef_near_target_and_gripper_closed"
    elif dist is not None and dist <= eef_target_distance_threshold and gripper_open:
        phase = "grasp"
        reason = "eef_near_target_and_gripper_open"

    return PhaseInfo(
        phase=phase,
        reason=reason,
        target_name=target_name or "",
        target_eef_dist=dist,
        prev_target_eef_dist=prev_dist,
        gripper_qpos=qpos,
        prev_gripper_qpos=prev_qpos,
        gripper_delta=delta,
        gripper_open=gripper_open,
        object_in_hand=object_in_hand,
        eef_pos=obs_vector(obs, "robot0_eef_pos"),
        target_pos=target_pos,
    )


def make_frame_sample(env, obs: Mapping[str, np.ndarray], cfg, args, step: int) -> FrameSample:
    image = get_libero_image(obs, MODEL_IMAGE_SIZE, camera_name=args.camera_name)
    seg_key = f"{args.camera_name}_segmentation_instance"
    depth_key = f"{args.camera_name}_depth"
    if seg_key not in obs or depth_key not in obs:
        raise KeyError(f"Expected observation keys {seg_key!r} and {depth_key!r}")
    seg_model = preprocess_segmentation_to_model(obs[seg_key], args.center_crop, args.crop_scale)
    geometry = capture_camera_geometry(
        env.sim,
        args.camera_name,
        obs[depth_key],
        model_height=MODEL_IMAGE_SIZE,
        model_width=MODEL_IMAGE_SIZE,
        center_crop=args.center_crop,
        crop_scale=args.crop_scale,
    )
    return FrameSample(obs=obs, image=image, seg_model=seg_model, geometry=geometry, step=step)


def write_rows(csv_path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_optional(value: Optional[float]) -> object:
    return "" if value is None else value


def empty_phase_csv_fields() -> Dict[str, object]:
    return {
        "phase_reason": "",
        "target_name": "",
        "target_eef_dist": "",
        "prev_target_eef_dist": "",
        "gripper_qpos": "",
        "prev_gripper_qpos": "",
        "gripper_delta": "",
    }


def phase_info_csv_fields(phase_info: PhaseInfo) -> Dict[str, object]:
    return {
        "phase_reason": phase_info.reason,
        "target_name": phase_info.target_name,
        "target_eef_dist": fmt_optional(phase_info.target_eef_dist),
        "prev_target_eef_dist": fmt_optional(phase_info.prev_target_eef_dist),
        "gripper_qpos": phase_info.gripper_qpos,
        "prev_gripper_qpos": phase_info.prev_gripper_qpos,
        "gripper_delta": phase_info.gripper_delta,
    }


def apply_phase_transition_windows(
    phase_info: PhaseInfo,
    *,
    step: int,
    last_grasp_step: Optional[int],
    transition_window: int,
) -> PhaseInfo:
    if last_grasp_step is None or phase_info.phase != "transport":
        return phase_info
    if step - last_grasp_step > transition_window:
        return phase_info
    return replace(
        phase_info,
        phase="grasp",
        reason=f"within_grasp_transition_window_after_step_{last_grasp_step}",
    )


def write_phase_diagnostic_row(
    csv_path: Path,
    *,
    args,
    task,
    task_id: int,
    episode_idx: int,
    step: int,
    phase_info: PhaseInfo,
) -> None:
    exists = csv_path.exists()
    eef_x, eef_y, eef_z = phase_info.eef_pos
    target_x, target_y, target_z = phase_info.target_pos
    row = {
        "run_id": args.run_id,
        "task_suite": args.task_suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "episode": episode_idx,
        "step": step,
        "camera": args.camera_name,
        "phase": phase_info.phase,
        "phase_reason": phase_info.reason,
        "target_name": phase_info.target_name,
        "target_eef_dist": fmt_optional(phase_info.target_eef_dist),
        "prev_target_eef_dist": fmt_optional(phase_info.prev_target_eef_dist),
        "gripper_qpos": phase_info.gripper_qpos,
        "prev_gripper_qpos": phase_info.prev_gripper_qpos,
        "gripper_delta": phase_info.gripper_delta,
        "gripper_open": int(phase_info.gripper_open),
        "object_in_hand": int(phase_info.object_in_hand),
        "eef_x": fmt_optional(eef_x),
        "eef_y": fmt_optional(eef_y),
        "eef_z": fmt_optional(eef_z),
        "target_x": fmt_optional(target_x),
        "target_y": fmt_optional(target_y),
        "target_z": fmt_optional(target_z),
    }
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PHASE_DIAGNOSTIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def rows_from_sims(
    *,
    args,
    task,
    task_id: int,
    episode_idx: int,
    step: int,
    grid: str,
    control: str,
    prompt_condition: str,
    phase: str,
    patch_semantics: str,
    sims: Mapping[str, Sequence[float]],
    n_pairs: int,
    n_oracle_pairs: int,
    n_target_patches: int,
    n_background_patches: int,
    phase_info: Optional[PhaseInfo] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    phase_fields = phase_info_csv_fields(phase_info) if phase_info is not None else empty_phase_csv_fields()
    for modality, values in sims.items():
        for layer_idx, sim in enumerate(values):
            rows.append(
                {
                    "run_id": args.run_id,
                    "task_suite": args.task_suite_name,
                    "task_id": task_id,
                    "task_name": task.name,
                    "episode": episode_idx,
                    "step": step,
                    "camera": args.camera_name,
                    "grid": grid,
                    "control": control,
                    "prompt_condition": prompt_condition,
                    "phase": phase,
                    **phase_fields,
                    "patch_semantics": patch_semantics,
                    "modality": modality,
                    "layer": layer_idx,
                    "sim_mean": sim,
                    "n_pairs": n_pairs,
                    "n_oracle_pairs": n_oracle_pairs,
                    "n_target_patches": n_target_patches,
                    "n_background_patches": n_background_patches,
                    "wrong_prompt": args.wrong_prompt,
                }
            )
    return rows


def subset_pairs(corr: OracleCorrespondence, allowed_target_patches: Sequence[int]) -> Tuple[List[int], List[int]]:
    allowed = set(int(x) for x in allowed_target_patches)
    src: List[int] = []
    dst: List[int] = []
    for source, target in zip(corr.source_patches, corr.target_patches):
        if int(target) in allowed:
            src.append(int(source))
            dst.append(int(target))
    return src, dst


def random_floor_pairs(n_patches: int, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    source = list(range(n_patches))
    target = list(range(n_patches))
    rng.shuffle(target)
    for idx, value in enumerate(target):
        if value == source[idx]:
            target[idx] = target[(idx + 1) % n_patches]
    return source, target


def clear_outputs(*outputs) -> None:
    for output in outputs:
        del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_controls(
    model,
    processor,
    args,
    task,
    task_id: int,
    episode_idx: int,
    sample: FrameSample,
    task_description: str,
    csv_path: Path,
    device: torch.device,
) -> None:
    prompt = build_prompt(str(args.pretrained_checkpoint), task_description)
    out_a = teacher_forward(model, processor, prompt, sample.image, args.center_crop, device)
    out_b = teacher_forward(model, processor, prompt, sample.image, args.center_crop, device)
    identity = identity_patch_correspondence(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, PATCH_SIZE)
    sims = cosine_layer_means(out_a, out_b, identity.source_patches, identity.target_patches)
    rows = rows_from_sims(
        args=args,
        task=task,
        task_id=task_id,
        episode_idx=episode_idx,
        step=sample.step,
        grid="S0_self",
        control="identity_same_frame_rerun",
        prompt_condition="P0",
        phase="control",
        patch_semantics="all",
        sims=sims,
        n_pairs=len(identity.target_patches),
        n_oracle_pairs=len(identity.target_patches),
        n_target_patches=0,
        n_background_patches=0,
    )
    floor_src, floor_dst = random_floor_pairs(NUM_VISUAL_PATCHES, args.seed + task_id * 1000 + episode_idx)
    floor_sims = cosine_layer_means(out_a, out_a, floor_src, floor_dst)
    rows.extend(
        rows_from_sims(
            args=args,
            task=task,
            task_id=task_id,
            episode_idx=episode_idx,
            step=sample.step,
            grid="S1_floor",
            control="random_same_frame_patch_pairs",
            prompt_condition="P0",
            phase="control",
            patch_semantics="all",
            sims=floor_sims,
            n_pairs=len(floor_src),
            n_oracle_pairs=len(floor_src),
            n_target_patches=0,
            n_background_patches=0,
        )
    )
    write_rows(csv_path, rows)
    clear_outputs(out_a, out_b)


def run_pair_grids(
    model,
    processor,
    args,
    task,
    task_id: int,
    episode_idx: int,
    prev_sample: FrameSample,
    curr_sample: FrameSample,
    task_description: str,
    phase_info: PhaseInfo,
    target_ids: Sequence[int],
    robot_ids: Sequence[int],
    csv_path: Path,
    device: torch.device,
) -> None:
    corr = oracle_3d_patch_correspondence(
        prev_sample.geometry,
        curr_sample.geometry,
        patch_size=PATCH_SIZE,
        depth_tolerance_m=args.depth_tolerance_m,
    )
    labels = label_patch_semantics(
        curr_sample.seg_model,
        target_ids=target_ids,
        robot_ids=robot_ids,
        threshold=args.patch_semantics_threshold,
        background_mode=args.background_mode,
    )
    prompt_by_condition = {
        "P0": build_prompt(str(args.pretrained_checkpoint), task_description),
        "P3": build_prompt(str(args.pretrained_checkpoint), args.wrong_prompt),
    }
    n_target = len(labels["target"])
    n_background = len(labels["background"])
    phase = phase_info.phase

    for condition, prompt in prompt_by_condition.items():
        prev_out = teacher_forward(model, processor, prompt, prev_sample.image, args.center_crop, device)
        curr_out = teacher_forward(model, processor, prompt, curr_sample.image, args.center_crop, device)
        rows: List[Dict[str, object]] = []
        for semantic in ("background", "target"):
            src, dst = subset_pairs(corr, labels[semantic])
            sims = cosine_layer_means(prev_out, curr_out, src, dst)
            rows.extend(
                rows_from_sims(
                    args=args,
                    task=task,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    step=curr_sample.step,
                    grid="A_prompt_patch",
                    control="oracle_3d_pair",
                    prompt_condition=condition,
                    phase="all",
                    patch_semantics=semantic,
                    sims=sims,
                    n_pairs=len(src),
                    n_oracle_pairs=len(corr.target_patches),
                    n_target_patches=n_target,
                    n_background_patches=n_background,
                    phase_info=phase_info,
                )
            )
            if condition == "P0" and (phase != "unknown" or args.include_unknown_phase):
                rows.extend(
                    rows_from_sims(
                        args=args,
                        task=task,
                        task_id=task_id,
                        episode_idx=episode_idx,
                        step=curr_sample.step,
                        grid="B_phase_patch",
                        control="oracle_3d_pair",
                        prompt_condition=condition,
                        phase=phase,
                        patch_semantics=semantic,
                        sims=sims,
                        n_pairs=len(src),
                        n_oracle_pairs=len(corr.target_patches),
                        n_target_patches=n_target,
                        n_background_patches=n_background,
                        phase_info=phase_info,
                    )
                )
        write_rows(csv_path, rows)
        clear_outputs(prev_out, curr_out)


def rollout_action(model, processor, cfg, obs: Mapping[str, np.ndarray], image: np.ndarray, task_description: str):
    observation = {
        "full_image": image,
        "prev_image": image,
        "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
    }
    action, _, _ = get_action(cfg, model, observation, task_description, processor=processor, last_caches=None)
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    return action.tolist()


def parse_task_ids(value: str, n_tasks: int) -> List[int]:
    if value == "all":
        return list(range(n_tasks))
    task_ids: List[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            task_ids.extend(range(int(start), int(end) + 1))
        else:
            task_ids.append(int(chunk))
    return [idx for idx in task_ids if 0 <= idx < n_tasks]


def maybe_plot(csv_path: Path, repo_output_dir: Path) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    values: Dict[Tuple[str, str, str, str, int], List[float]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["grid"] not in {"A_prompt_patch", "B_phase_patch", "S0_self", "S1_floor"}:
                continue
            key = (
                row["grid"],
                row["prompt_condition"] if row["grid"] == "A_prompt_patch" else row["phase"],
                row["patch_semantics"],
                row["modality"],
                int(row["layer"]),
            )
            try:
                value = float(row["sim_mean"])
            except ValueError:
                continue
            if math.isnan(value):
                continue
            values.setdefault(key, []).append(value)

    if not values:
        return None

    plot_path = repo_output_dir / "layerwise_kv_contextualization.png"
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True)
    for row_idx, modality in enumerate(("K", "V", "H")):
        ax_a = axes[row_idx, 0]
        ax_b = axes[row_idx, 1]
        for semantic in ("background", "target"):
            for condition in ("P0", "P3"):
                xs: List[int] = []
                ys: List[float] = []
                for layer in range(32):
                    key = ("A_prompt_patch", condition, semantic, modality, layer)
                    if key in values:
                        xs.append(layer)
                        ys.append(float(np.mean(values[key])))
                if xs:
                    ax_a.plot(xs, ys, label=f"{condition}/{semantic}")
        for semantic in ("background", "target"):
            for phase in ("reach", "grasp", "transport", "place"):
                xs = []
                ys = []
                for layer in range(32):
                    key = ("B_phase_patch", phase, semantic, modality, layer)
                    if key in values:
                        xs.append(layer)
                        ys.append(float(np.mean(values[key])))
                if xs:
                    ax_b.plot(xs, ys, label=f"{phase}/{semantic}")
        ax_a.set_title(f"Grid A {modality}")
        ax_b.set_title(f"Grid B {modality}")
        ax_a.set_ylabel("cosine")
        ax_a.grid(True, alpha=0.25)
        ax_b.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("layer")
    axes[-1, 1].set_xlabel("layer")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(fontsize=8)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return str(plot_path)


def summarize_csv(csv_path: Path) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "rows": 0,
        "controls": {},
        "grid_pair_counts": {},
        "phase_counts": {},
        "phase_reason_counts": {},
        "zero_pair_rows": {},
    }
    control_values: Dict[Tuple[str, str], List[float]] = {}
    pair_counts: Dict[Tuple[str, str], List[int]] = {}
    phase_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    zero_pair_rows: Counter[str] = Counter()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary["rows"] += 1
            grid = row["grid"]
            modality = row["modality"]
            try:
                sim = float(row["sim_mean"])
                n_pairs = int(row["n_pairs"])
            except ValueError:
                continue
            if grid == "B_phase_patch":
                phase_counts[row["phase"]] += 1
                reason_counts[row.get("phase_reason", "")] += 1
            if grid in {"A_prompt_patch", "B_phase_patch"} and n_pairs == 0:
                zero_pair_rows[f"{grid}/{row['patch_semantics']}"] += 1
            if math.isnan(sim):
                continue
            if grid in {"S0_self", "S1_floor"}:
                control_values.setdefault((grid, modality), []).append(sim)
            else:
                pair_counts.setdefault((grid, row["patch_semantics"]), []).append(n_pairs)

    summary["controls"] = {
        f"{grid}/{modality}": {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        for (grid, modality), vals in control_values.items()
        if vals
    }
    summary["grid_pair_counts"] = {
        f"{grid}/{semantic}": {
            "mean": float(np.mean(vals)),
            "min": int(np.min(vals)),
            "max": int(np.max(vals)),
        }
        for (grid, semantic), vals in pair_counts.items()
        if vals
    }
    summary["phase_counts"] = dict(phase_counts)
    summary["phase_reason_counts"] = dict(reason_counts)
    summary["zero_pair_rows"] = dict(zero_pair_rows)
    return summary


def write_report(args, csv_path: Path, repo_output_dir: Path, plot_path: Optional[str], summary: Mapping[str, object]) -> Path:
    report_path = repo_output_dir / "kv_contextualization_study_report.md"
    payload = {
        "run_id": args.run_id,
        "task_suite": args.task_suite_name,
        "task_ids": args.task_ids,
        "episodes": args.episodes_per_task,
        "camera": args.camera_name,
        "rollout_policy": args.rollout_policy,
        "center_crop": args.center_crop,
        "csv_path": str(csv_path),
        "plot_path": plot_path,
        "summary": summary,
    }
    with report_path.open("w") as f:
        f.write("# OpenVLA KV Contextualization Layer-wise Study\n\n")
        f.write(f"- Run ID: `{args.run_id}`\n")
        f.write(f"- Task suite: `{args.task_suite_name}`\n")
        f.write(f"- Task IDs: `{args.task_ids}`\n")
        f.write(f"- Episodes per task: `{args.episodes_per_task}`\n")
        f.write(f"- Camera: `{args.camera_name}`\n")
        f.write(f"- Rollout policy: `{args.rollout_policy}`\n")
        f.write(f"- Full CSV: `{csv_path}`\n")
        if plot_path:
            f.write(f"- Plot: `{plot_path}`\n")
        f.write("\n## Summary\n\n")
        f.write("```json\n")
        f.write(json.dumps(payload, indent=2, sort_keys=True))
        f.write("\n```\n")
    return report_path


def make_cfg(args):
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.pretrained_checkpoint,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        center_crop=args.center_crop,
        use_vla_cache=False,
        unnorm_key=args.task_suite_name,
    )


def run(args) -> None:
    set_seed_everywhere(args.seed)
    if os.path.isdir(args.pretrained_checkpoint):
        args.pretrained_checkpoint = str(Path(args.pretrained_checkpoint).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser()
    repo_output_dir = Path(args.repo_output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "layerwise_sim.csv"
    phase_diag_path = output_dir / "phase_diagnostics.csv"
    if csv_path.exists() and not args.append:
        csv_path.unlink()
    if phase_diag_path.exists() and not args.append:
        phase_diag_path.unlink()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[info] device={device}")
    cfg = make_cfg(args)
    original_cwd = Path.cwd()
    try:
        os.chdir(OPENVLA_ROOT)
        model = get_model(cfg)
        model.eval()
        processor = get_processor(cfg)
    finally:
        os.chdir(original_cwd)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = parse_task_ids(args.task_ids, task_suite.n_tasks)
    print(f"[info] task ids={task_ids}")

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(
            task,
            cfg.model_family,
            resolution=args.camera_resolution,
            camera_name=args.camera_name,
            camera_depths=True,
            camera_segmentations="instance",
        )
        env.seed(args.seed)
        try:
            env.reset()
            _, target_ids, robot_ids, target_names = get_instance_id_sets(env, args.target_interest_index)
            target_name = target_names[0] if target_names else None
            print(
                f"[task {task_id}] {task_description} | target={target_name} "
                f"target_ids={target_ids} robot_ids={robot_ids}"
            )
            for episode_idx in range(args.episodes_per_task):
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

                prev_sample: Optional[FrameSample] = None
                prev_obs: Optional[Mapping[str, np.ndarray]] = None
                last_grasp_step: Optional[int] = None
                samples_done = 0
                for step in range(args.max_rollout_steps + 1):
                    phase_info = infer_phase(
                        obs,
                        prev_obs,
                        target_name,
                        args.eef_target_distance_threshold,
                        args.gripper_open_threshold,
                        args.gripper_delta_threshold,
                    )
                    if phase_info.phase == "grasp":
                        last_grasp_step = step
                    phase_info = apply_phase_transition_windows(
                        phase_info,
                        step=step,
                        last_grasp_step=last_grasp_step,
                        transition_window=args.phase_transition_window,
                    )
                    if args.phase_diagnostics_only:
                        write_phase_diagnostic_row(
                            phase_diag_path,
                            args=args,
                            task=task,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            step=step,
                            phase_info=phase_info,
                        )
                        if step % max(1, args.phase_diagnostics_print_interval) == 0:
                            dist = "nan" if phase_info.target_eef_dist is None else f"{phase_info.target_eef_dist:.4f}"
                            print(
                                f"[task {task_id} ep {episode_idx}] step {step} phase={phase_info.phase} "
                                f"reason={phase_info.reason} qpos={phase_info.gripper_qpos:.5f} "
                                f"dq={phase_info.gripper_delta:.5f} dist={dist}"
                            )
                        if step >= args.max_rollout_steps:
                            break

                        prev_obs = obs
                        if args.rollout_policy == "model":
                            image = get_libero_image(obs, MODEL_IMAGE_SIZE, camera_name=args.camera_name)
                            action = rollout_action(model, processor, cfg, obs, image, task_description)
                        else:
                            action = get_libero_dummy_action(cfg.model_family)
                        obs, _, done, _ = env.step(action)
                        if done:
                            done_phase_info = infer_phase(
                                obs,
                                prev_obs,
                                target_name,
                                args.eef_target_distance_threshold,
                                args.gripper_open_threshold,
                                args.gripper_delta_threshold,
                            )
                            done_phase_info = replace(done_phase_info, phase="place", reason="env_done_after_action")
                            write_phase_diagnostic_row(
                                phase_diag_path,
                                args=args,
                                task=task,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                step=step + 1,
                                phase_info=done_phase_info,
                            )
                            break
                        continue

                    sample = make_frame_sample(env, obs, cfg, args, step)
                    if step == 0 and args.run_controls:
                        print(f"[task {task_id} ep {episode_idx}] S0/S1 controls at step {step}")
                        run_controls(
                            model,
                            processor,
                            args,
                            task,
                            task_id,
                            episode_idx,
                            sample,
                            task_description,
                            csv_path,
                            device,
                        )

                    should_sample_pair = (
                        prev_sample is not None
                        and step > 0
                        and step % args.sample_interval == 0
                        and samples_done < args.max_samples_per_episode
                    )
                    if should_sample_pair:
                        dist = "nan" if phase_info.target_eef_dist is None else f"{phase_info.target_eef_dist:.4f}"
                        print(
                            f"[task {task_id} ep {episode_idx}] pair step {prev_sample.step}->{step} "
                            f"phase={phase_info.phase} reason={phase_info.reason} "
                            f"qpos={phase_info.gripper_qpos:.5f} dq={phase_info.gripper_delta:.5f} dist={dist}"
                        )
                        run_pair_grids(
                            model,
                            processor,
                            args,
                            task,
                            task_id,
                            episode_idx,
                            prev_sample,
                            sample,
                            task_description,
                            phase_info,
                            target_ids,
                            robot_ids,
                            csv_path,
                            device,
                        )
                        samples_done += 1
                        if samples_done >= args.max_samples_per_episode:
                            break

                    if step >= args.max_rollout_steps:
                        break

                    prev_sample = sample
                    prev_obs = obs
                    if args.rollout_policy == "model":
                        action = rollout_action(model, processor, cfg, obs, sample.image, task_description)
                    else:
                        action = get_libero_dummy_action(cfg.model_family)
                    obs, _, done, _ = env.step(action)
                    if done:
                        if args.sample_done_phase and prev_sample is not None and samples_done < args.max_samples_per_episode:
                            done_sample = make_frame_sample(env, obs, cfg, args, step + 1)
                            done_phase_info = infer_phase(
                                obs,
                                prev_obs,
                                target_name,
                                args.eef_target_distance_threshold,
                                args.gripper_open_threshold,
                                args.gripper_delta_threshold,
                            )
                            done_phase_info = replace(done_phase_info, phase="place", reason="env_done_after_action")
                            print(
                                f"[task {task_id} ep {episode_idx}] terminal pair step {prev_sample.step}->{step + 1} "
                                f"phase={done_phase_info.phase} reason={done_phase_info.reason}"
                            )
                            run_pair_grids(
                                model,
                                processor,
                                args,
                                task,
                                task_id,
                                episode_idx,
                                prev_sample,
                                done_sample,
                                task_description,
                                done_phase_info,
                                target_ids,
                                robot_ids,
                                csv_path,
                                device,
                            )
                        break
        finally:
            env.close()

    if args.phase_diagnostics_only:
        print(f"[ok] wrote phase diagnostics to {phase_diag_path}")
        return

    plot_path = maybe_plot(csv_path, repo_output_dir)
    summary = summarize_csv(csv_path)
    report_path = write_report(args, csv_path, repo_output_dir, plot_path, summary)
    print(f"[ok] wrote CSV to {csv_path}")
    print(f"[ok] wrote report to {report_path}")
    if plot_path:
        print(f"[ok] wrote plot to {plot_path}")


def parse_args() -> argparse.Namespace:
    default_run_id = time.strftime("kv_contextualization_%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-checkpoint",
        default="/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial",
    )
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-rollout-steps", type=int, default=12)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--max-samples-per-episode", type=int, default=1)
    parser.add_argument("--rollout-policy", choices=["model", "dummy"], default="dummy")
    parser.add_argument("--center-crop", type=str2bool, default=True)
    parser.add_argument("--crop-scale", type=float, default=0.9)
    parser.add_argument("--load-in-8bit", type=str2bool, default=False)
    parser.add_argument("--load-in-4bit", type=str2bool, default=False)
    parser.add_argument("--run-controls", type=str2bool, default=True)
    parser.add_argument("--wrong-prompt", default="open the drawer and put the mug inside")
    parser.add_argument("--target-interest-index", type=int, default=0)
    parser.add_argument("--background-mode", choices=["all_non_target", "zero_only"], default="all_non_target")
    parser.add_argument("--patch-semantics-threshold", type=float, default=0.5)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.04)
    parser.add_argument("--eef-target-distance-threshold", type=float, default=0.12)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.035)
    parser.add_argument("--gripper-delta-threshold", type=float, default=0.005)
    parser.add_argument("--include-unknown-phase", type=str2bool, default=False)
    parser.add_argument("--phase-diagnostics-only", type=str2bool, default=False)
    parser.add_argument("--phase-diagnostics-print-interval", type=int, default=10)
    parser.add_argument("--phase-transition-window", type=int, default=5)
    parser.add_argument("--sample-done-phase", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--append", type=str2bool, default=False)
    parser.add_argument("--run-id", default=default_run_id)
    parser.add_argument(
        "--output-dir",
        default=f"/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_kv_contextualization/{default_run_id}",
    )
    parser.add_argument(
        "--repo-output-dir",
        default=str(REPO_ROOT / "Experiments" / "openvla_libero_mc_cache" / "outputs" / "kv_study"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
