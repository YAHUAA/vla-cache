"""Summarize multiple LIBERO eval summary JSON files into a compact matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional


CSV_FIELDS = [
    "method",
    "task_ids",
    "successes",
    "episodes",
    "success_rate",
    "latency_ms_mean",
    "avg_reuse_ratio",
    "avg_candidates",
    "kv_remap_steps",
    "local_log_filepath",
]


def weighted_mean(episodes: Iterable[Mapping[str, object]], metric: str, weight: str) -> Optional[float]:
    total = 0.0
    denom = 0.0
    for episode in episodes:
        value = episode.get(metric)
        episode_weight = episode.get(weight)
        if value is None or episode_weight is None:
            continue
        episode_weight = float(episode_weight)
        if episode_weight <= 0:
            continue
        total += float(value) * episode_weight
        denom += episode_weight
    if denom == 0:
        return None
    return total / denom


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def method_from_path(path: Path) -> str:
    name = path.stem
    if name.endswith("_summary"):
        name = name[: -len("_summary")]
    return name


def summarize_file(path: Path) -> dict[str, object]:
    with path.open("r") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", [])
    return {
        "method": payload.get("method", method_from_path(path)),
        "task_ids": ",".join(str(idx) for idx in payload.get("task_ids", [])),
        "successes": int(payload.get("total_successes", 0)),
        "episodes": int(payload.get("total_episodes", 0)),
        "success_rate": float(payload.get("total_success_rate", 0.0)),
        "latency_ms_mean": weighted_mean(episodes, "latency_ms_mean", "action_steps"),
        "avg_reuse_ratio": weighted_mean(episodes, "avg_reuse_ratio", "cache_steps"),
        "avg_candidates": weighted_mean(episodes, "avg_candidates", "cache_steps"),
        "kv_remap_steps": int(sum(int(ep.get("kv_remap_steps", 0)) for ep in episodes)),
        "local_log_filepath": payload.get("local_log_filepath", ""),
    }


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[Mapping[str, object]]) -> str:
    lines = [
        "| method | task_ids | success | success_rate | latency mean ms | avg reuse | avg candidates | kv remap steps |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {task_ids} | {successes}/{episodes} | {success_rate:.1%} | {latency} | {reuse} | {candidates} | {kv} |".format(
                method=row["method"],
                task_ids=row["task_ids"],
                successes=row["successes"],
                episodes=row["episodes"],
                success_rate=row["success_rate"],
                latency=fmt_float(row["latency_ms_mean"]),
                reuse=fmt_float(row["avg_reuse_ratio"]),
                candidates=fmt_float(row["avg_candidates"]),
                kv=row["kv_remap_steps"],
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_paths = sorted(args.summary_dir.glob("*_summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No *_summary.json files found in {args.summary_dir}")

    rows = [summarize_file(path) for path in summary_paths]
    rows.sort(key=lambda row: str(row["method"]))

    output_csv = args.output_csv or args.summary_dir / "matrix_summary.csv"
    output_md = args.output_md or args.summary_dir / "matrix_summary.md"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    write_csv(output_csv, rows)
    table = markdown_table(rows)
    with output_md.open("w") as f:
        f.write("# LIBERO Eval Matrix Summary\n\n")
        f.write(table)
        f.write("\n")

    print(table)
    print(f"[ok] wrote {output_csv}")
    print(f"[ok] wrote {output_md}")


if __name__ == "__main__":
    main()
