"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import imageio
import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    # Use VLA-Cache for faster inference
    use_vla_cache: bool = True
    use_motion_compensated_cache: bool = False       # If True, replace same-grid static patch detection with 2D motion compensation
    mc_enable_kv_remap: bool = False                 # Experimental: remap DynamicCache visual KV slots when supported
    mc_patch_size: int = 14                          # ViT patch size used by OpenVLA's 224x224 image path
    mc_top_k: int = 130                              # Max motion-compensated candidate patches before attention veto
    mc_task_top_k: int = 120                         # Attention top-k patches excluded from reuse
    mc_search_radius: int = 28                       # Pixel search radius for global prev->current translation
    mc_search_step: int = 2                          # Pixel step for global translation search
    mc_min_confidence: float = 0.30                  # Patch-overlap confidence gate after translation compensation
    mc_similarity_threshold: float = 0.70            # RGB patch similarity gate after compensation
    mc_samples_per_axis: int = 5                     # Per-patch correspondence sampling density
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    camera_name: str = "agentview"                   # LIBERO camera name, e.g. agentview or robot0_eye_in_hand
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    num_tasks_to_eval: Optional[int] = None           # Optional smoke-test limit for task count
    task_ids: Optional[str] = None                    # Optional explicit task ids, e.g. "0", "0,3,5", "0-4", or "all"
    custom_bddl_file: Optional[str] = None             # Optional absolute or repo-relative BDDL file for one custom task
    custom_init_states_file: Optional[str] = None      # Optional torch-saved init states for the custom BDDL task
    custom_task_language: Optional[str] = None         # Optional language override for the custom BDDL task
    custom_task_name: str = "custom_libero_task"       # Name to use in summaries for a custom BDDL task
    custom_max_steps: Optional[int] = None             # Optional max action horizon for a custom BDDL task
    save_rollout_videos: bool = True                  # Whether to save per-episode MP4 rollout videos
    rollout_dir: Optional[str] = None                 # Optional directory for rollout videos
    summary_json_path: Optional[str] = None            # Optional machine-readable eval summary path

    # fmt: on


def parse_task_ids(task_ids: Optional[str], n_tasks: int) -> Optional[list[int]]:
    if task_ids is None:
        return None
    value = str(task_ids).strip()
    if not value:
        return None
    if value.lower() == "all":
        return list(range(n_tasks))

    selected: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending task id range: {chunk!r}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(chunk))

    invalid = [idx for idx in selected if idx < 0 or idx >= n_tasks]
    if invalid:
        raise ValueError(f"Task id(s) out of range for suite with {n_tasks} tasks: {invalid}")
    return list(dict.fromkeys(selected))


def read_bddl_language(bddl_file: Path) -> str:
    for line in bddl_file.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("(:language"):
            language = stripped[len("(:language") :].strip()
            if language.endswith(")"):
                language = language[:-1].strip()
            return language
    return bddl_file.stem.replace("_", " ")


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    use_custom_task = cfg.custom_bddl_file is not None
    if use_custom_task:
        assert cfg.custom_init_states_file is not None, "cfg.custom_init_states_file is required for custom BDDL eval!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite, or a one-task custom BDDL suite.
    custom_bddl_file = Path(cfg.custom_bddl_file).expanduser().resolve() if use_custom_task else None
    custom_init_states_file = (
        Path(cfg.custom_init_states_file).expanduser().resolve() if use_custom_task else None
    )
    if use_custom_task:
        task_suite = None
        custom_language = cfg.custom_task_language or read_bddl_language(custom_bddl_file)
        custom_task = SimpleNamespace(
            name=cfg.custom_task_name,
            language=custom_language,
            problem="LiberoCustom",
            problem_folder="",
            bddl_file=str(custom_bddl_file),
            init_states_file=str(custom_init_states_file),
        )
        num_tasks_in_suite = 1
    else:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        custom_task = None
        num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    print(f"Camera: {cfg.camera_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Camera: {cfg.camera_name}\n")
    if use_custom_task:
        print(f"Custom BDDL: {custom_bddl_file}")
        print(f"Custom init states: {custom_init_states_file}")
        log_file.write(f"Custom BDDL: {custom_bddl_file}\n")
        log_file.write(f"Custom init states: {custom_init_states_file}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    explicit_task_ids = parse_task_ids(cfg.task_ids, num_tasks_in_suite)
    if explicit_task_ids is not None:
        task_ids = explicit_task_ids
    elif cfg.num_tasks_to_eval is None:
        task_ids = range(num_tasks_in_suite)
    else:
        task_ids = range(min(num_tasks_in_suite, cfg.num_tasks_to_eval))

    task_ids = list(task_ids)
    print(f"Task IDs: {task_ids}")
    log_file.write(f"Task IDs: {task_ids}\n")

    episode_summaries = []
    task_summaries = []

    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = custom_task if use_custom_task else task_suite.get_task(task_id)

        # Get default LIBERO or custom initial states
        if use_custom_task:
            initial_states = torch.load(custom_init_states_file)
        else:
            initial_states = task_suite.get_task_init_states(task_id)
        if cfg.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"Requested {cfg.num_trials_per_task} trials, but only {len(initial_states)} init states are available"
            )

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(
            task,
            cfg.model_family,
            resolution=256,
            camera_name=cfg.camera_name,
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_images_heatmap = []
            prev_img = None
            last_caches = None
            episode_mc_steps = 0
            episode_mc_reuse_tokens = 0
            episode_mc_candidates = 0
            episode_mc_kv_remap_steps = 0
            episode_mc_shift_abs = 0.0
            episode_mc_shift_score = 0.0
            episode_latency_ms = []
            done = False
            
            
            if use_custom_task and cfg.custom_max_steps is not None:
                max_steps = cfg.custom_max_steps
            elif cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size, camera_name=cfg.camera_name)

                    # Save previous image
                    if prev_img is None:
                        prev_img = img
                    else:
                        prev_img = replay_images[-1]
                        
                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "prev_image": prev_img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }


                    # Query model to get action
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    action_start_time = time.perf_counter()
                    action, last_caches, result_image = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        last_caches=last_caches,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    episode_latency_ms.append((time.perf_counter() - action_start_time) * 1000.0)
                    mc_metrics = last_caches.get("mc_metrics", {}) if last_caches is not None else {}
                    if mc_metrics.get("cache_mode") in {"original_grid", "motion_compensated_2d"}:
                        episode_mc_steps += 1
                        episode_mc_reuse_tokens += int(mc_metrics.get("num_mc_reuse_tokens", 0))
                        episode_mc_candidates += int(mc_metrics.get("num_mc_candidates", 0))
                        episode_mc_kv_remap_steps += int(bool(mc_metrics.get("mc_kv_remap_applied", False)))
                        episode_mc_shift_abs += abs(float(mc_metrics.get("mc_shift_x", 0))) + abs(
                            float(mc_metrics.get("mc_shift_y", 0))
                        )
                        episode_mc_shift_score += float(mc_metrics.get("mc_shift_score", 0.0))
                    replay_images_heatmap.append(result_image)
                    # imageio.imwrite(f"rollouts/live_image.png", result_image)

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            rollout_video_path = None
            if cfg.save_rollout_videos:
                rollout_video_path = save_rollout_video(
                    replay_images_heatmap,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    log_file=log_file,
                    rollout_dir=cfg.rollout_dir,
                )

            # Save a replay video of the episode
            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            if episode_mc_steps > 0:
                avg_reuse_ratio = episode_mc_reuse_tokens / (episode_mc_steps * 256)
                avg_candidates = episode_mc_candidates / episode_mc_steps
                avg_shift_abs = episode_mc_shift_abs / episode_mc_steps
                avg_shift_score = episode_mc_shift_score / episode_mc_steps
                print(
                    "MC/VLA-Cache stats: "
                    f"steps={episode_mc_steps}, avg_reuse_ratio={avg_reuse_ratio:.3f}, "
                    f"avg_candidates={avg_candidates:.1f}, kv_remap_steps={episode_mc_kv_remap_steps}, "
                    f"avg_shift_l1_px={avg_shift_abs:.2f}, avg_shift_score={avg_shift_score:.3f}"
                )
            if episode_latency_ms:
                latency = np.asarray(episode_latency_ms, dtype=np.float64)
                print(
                    "Action latency ms: "
                    f"mean={latency.mean():.1f}, p50={np.percentile(latency, 50):.1f}, "
                    f"p90={np.percentile(latency, 90):.1f}, std={latency.std():.1f}, n={latency.size}"
                )
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            if episode_mc_steps > 0:
                avg_reuse_ratio = episode_mc_reuse_tokens / (episode_mc_steps * 256)
                avg_candidates = episode_mc_candidates / episode_mc_steps
                avg_shift_abs = episode_mc_shift_abs / episode_mc_steps
                avg_shift_score = episode_mc_shift_score / episode_mc_steps
                log_file.write(
                    "MC/VLA-Cache stats: "
                    f"steps={episode_mc_steps}, avg_reuse_ratio={avg_reuse_ratio:.3f}, "
                    f"avg_candidates={avg_candidates:.1f}, kv_remap_steps={episode_mc_kv_remap_steps}, "
                    f"avg_shift_l1_px={avg_shift_abs:.2f}, avg_shift_score={avg_shift_score:.3f}\n"
                )
            if episode_latency_ms:
                latency = np.asarray(episode_latency_ms, dtype=np.float64)
                log_file.write(
                    "Action latency ms: "
                    f"mean={latency.mean():.1f}, p50={np.percentile(latency, 50):.1f}, "
                    f"p90={np.percentile(latency, 90):.1f}, std={latency.std():.1f}, n={latency.size}\n"
                )
            episode_summary = {
                "task_id": int(task_id),
                "task_name": task.name,
                "task_description": task_description,
                "episode_idx": int(episode_idx),
                "success": bool(done),
                "action_steps": int(len(episode_latency_ms)),
                "cache_steps": int(episode_mc_steps),
                "rollout_video_path": rollout_video_path,
                "avg_reuse_ratio": (
                    float(episode_mc_reuse_tokens / (episode_mc_steps * 256)) if episode_mc_steps > 0 else None
                ),
                "avg_candidates": float(episode_mc_candidates / episode_mc_steps) if episode_mc_steps > 0 else None,
                "kv_remap_steps": int(episode_mc_kv_remap_steps),
                "avg_shift_l1_px": float(episode_mc_shift_abs / episode_mc_steps) if episode_mc_steps > 0 else None,
                "avg_shift_score": float(episode_mc_shift_score / episode_mc_steps) if episode_mc_steps > 0 else None,
            }
            if episode_latency_ms:
                latency = np.asarray(episode_latency_ms, dtype=np.float64)
                episode_summary["latency_ms_mean"] = float(latency.mean())
                episode_summary["latency_ms_p50"] = float(np.percentile(latency, 50))
                episode_summary["latency_ms_p90"] = float(np.percentile(latency, 90))
                episode_summary["latency_ms_std"] = float(latency.std())
            episode_summaries.append(episode_summary)
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        task_summaries.append(
            {
                "task_id": int(task_id),
                "task_name": task.name,
                "task_description": task_description,
                "episodes": int(task_episodes),
                "successes": int(task_successes),
                "success_rate": float(task_successes) / float(task_episodes),
            }
        )
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    if cfg.summary_json_path is not None:
        summary_json_path = Path(cfg.summary_json_path).expanduser()
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_json_path.open("w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "task_suite_name": cfg.task_suite_name,
                    "task_ids": task_ids,
                    "custom_bddl_file": str(custom_bddl_file) if use_custom_task else None,
                    "custom_init_states_file": str(custom_init_states_file) if use_custom_task else None,
                    "custom_task_language": custom_task.language if use_custom_task else None,
                    "camera_name": cfg.camera_name,
                    "use_vla_cache": cfg.use_vla_cache,
                    "use_motion_compensated_cache": cfg.use_motion_compensated_cache,
                    "mc_enable_kv_remap": cfg.mc_enable_kv_remap,
                    "num_trials_per_task": cfg.num_trials_per_task,
                    "seed": cfg.seed,
                    "total_episodes": int(total_episodes),
                    "total_successes": int(total_successes),
                    "total_success_rate": float(total_successes) / float(total_episodes) if total_episodes else 0.0,
                    "tasks": task_summaries,
                    "episodes": episode_summaries,
                    "local_log_filepath": local_log_filepath,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"Wrote summary JSON to {summary_json_path}")
        log_file.write(f"Wrote summary JSON to {summary_json_path}\n")
        log_file.flush()

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
