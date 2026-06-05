#!/usr/bin/env python
"""Generate fixed LIBERO init states for a custom BDDL task."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def read_bddl_language(bddl_file: Path) -> str:
    for line in bddl_file.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("(:language"):
            language = stripped[len("(:language") :].strip()
            if language.endswith(")"):
                language = language[:-1].strip()
            return language
    return bddl_file.stem.replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--metadata-json", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    openvla_root = repo_root / "src" / "openvla"
    sys.path.append(str(openvla_root))

    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env

    bddl_file = args.bddl_file.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    language = read_bddl_language(bddl_file)
    task = SimpleNamespace(language=language, problem_folder="", bddl_file=str(bddl_file))
    env, task_description = get_libero_env(
        task,
        "openvla",
        resolution=args.resolution,
        camera_name=args.camera_name,
    )

    states = []
    skipped_successes = 0
    attempts = 0
    max_attempts = max(args.num_states * 20, args.num_states + 10)
    while len(states) < args.num_states and attempts < max_attempts:
        env.seed(args.seed + attempts)
        obs = env.reset()
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
        success = bool(env.check_success())
        if success:
            skipped_successes += 1
            attempts += 1
            continue
        states.append(env.sim.get_state().flatten().copy())
        attempts += 1

    env.close()

    if len(states) != args.num_states:
        raise RuntimeError(
            f"Only generated {len(states)} valid states after {attempts} attempts; "
            f"skipped {skipped_successes} reset-success states"
        )

    states_array = np.asarray(states, dtype=np.float64)
    torch.save(states_array, output)
    print(f"Wrote {states_array.shape} init states to {output}")

    if args.metadata_json is not None:
        metadata_path = args.metadata_json.expanduser().resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "bddl_file": str(bddl_file),
            "camera_name": args.camera_name,
            "language": task_description,
            "num_states": int(states_array.shape[0]),
            "seed": args.seed,
            "settle_steps": args.settle_steps,
            "skipped_reset_successes": skipped_successes,
            "state_shape": list(states_array.shape),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
