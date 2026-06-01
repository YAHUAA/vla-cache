# Project Rules

- Applies to the whole repo. A deeper `AGENTS.md` overrides this file for its subtree.
- Keep changes focused; do not edit unrelated files or user changes.
- Prefer existing project structure, naming, and implementation style.
- Do not run destructive commands or overwrite files unless explicitly asked.
- After code changes, run the smallest relevant check when available; report if not run.
- 如果需要使用GPU，记得在沙箱外访问
- 针对该项目创建的虚拟环境，请放置在/mnt/data0/zjh_data/文件夹之下，并做好文件管理；

## Data And Weights

- Download datasets, pretrained weights, checkpoints, archives, generated data, and other large external files under `/mnt/data0/zjh_data`.
- Organize them clearly, e.g.:
  - `/mnt/data0/zjh_data/Embodied_Proj/datasets/<dataset_name>/`
  - `/mnt/data0/zjh_data/Embodied_Proj/weights/<model_or_method>/`
  - `/mnt/data0/zjh_data/Embodied_Proj/checkpoints/<experiment>/`
- Reuse existing files there when appropriate.
- Do not put large downloaded/generated files in the repo or commit them to git.
- Prefer configurable paths via CLI args, env vars, or config files.

## Experiments

- If a task targets one experiment directory, keep edits there unless shared code changes are required.
- Mention shared files before editing them.
- Do not refactor unrelated experiments as part of a local fix.
