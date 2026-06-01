#!/usr/bin/env python3
"""Smoke-test the RGB 2D motion-compensation patch selector.

This test does not load OpenVLA or LIBERO. It creates a textured image, shifts
it by a known integer offset, runs the same selector used by OpenVLA inference,
and writes a small report. It is meant to catch wiring and correspondence bugs
before launching a GPU rollout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
OPENVLA_SRC = EXP_DIR.parents[1] / "src" / "openvla"
sys.path.insert(0, str(OPENVLA_SRC))

from experiments.robot.motion_compensation import (  # noqa: E402
    draw_motion_compensated_overlay,
    find_motion_compensated_static_patches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test OpenVLA MC patch correspondence.")
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "outputs" / "smoke_motion")
    parser.add_argument("--shift-x", type=int, default=18)
    parser.add_argument("--shift-y", type=int, default=-12)
    parser.add_argument("--patch-size", type=int, default=14)
    return parser.parse_args()


def make_textured_image(size: int = 224) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rgb = np.stack(
        [
            0.48 + 0.20 * np.sin(xx / 5.7) + 0.10 * np.cos(yy / 9.1),
            0.45 + 0.18 * np.cos((xx + yy) / 8.3),
            0.50 + 0.16 * np.sin((xx - 2.0 * yy) / 10.7),
        ],
        axis=-1,
    )
    rgb = np.clip(rgb, 0.0, 1.0)
    image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse([42, 58, 92, 108], fill=(220, 62, 45))
    draw.rectangle([132, 118, 184, 166], fill=(42, 126, 220))
    draw.rectangle([78, 154, 122, 188], fill=(45, 180, 96))
    return image


def shift_image(image: Image.Image, dx: int, dy: int) -> Image.Image:
    arr = np.asarray(image)
    out = np.zeros_like(arr)
    height, width = arr.shape[:2]
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return Image.fromarray(out, mode="RGB")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prev_image = make_textured_image()
    curr_image = shift_image(prev_image, args.shift_x, args.shift_y)
    corr = find_motion_compensated_static_patches(
        curr_image,
        prev_image,
        patch_size=args.patch_size,
        top_k=130,
        search_radius=28,
        search_step=1,
        min_confidence=0.30,
        sim_threshold=0.30,
    )

    overlay = draw_motion_compensated_overlay(curr_image, corr.target_patches, patch_size=args.patch_size)
    prev_image.save(args.output_dir / "prev.png")
    curr_image.save(args.output_dir / "curr_shifted.png")
    overlay.save(args.output_dir / "mc_reuse_overlay.png")

    expected = (args.shift_x, args.shift_y)
    passed = corr.shift_xy == expected and len(corr.target_patches) > 0
    report = [
        "# Motion Compensation Smoke Report",
        "",
        f"- Expected shift: `{expected}`",
        f"- Estimated shift: `{corr.shift_xy}`",
        f"- Shift score: `{corr.score:.6f}`",
        f"- Candidates before top-k: `{corr.num_candidates_before_topk}`",
        f"- Selected patches: `{len(corr.target_patches)}`",
        f"- Status: `{'PASS' if passed else 'FAIL'}`",
        "",
        "Generated images:",
        "",
        "- `prev.png`",
        "- `curr_shifted.png`",
        "- `mc_reuse_overlay.png`",
    ]
    (args.output_dir / "smoke_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
