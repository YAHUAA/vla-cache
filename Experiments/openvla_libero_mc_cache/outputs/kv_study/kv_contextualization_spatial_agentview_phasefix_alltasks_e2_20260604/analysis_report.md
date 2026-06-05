# OpenVLA KV Contextualization Layer-wise Study 结果分析报告

Run ID: `kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604`

数据位置：

- CSV: `/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_kv_contextualization/kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604/layerwise_sim.csv`
- 原始 summary: `kv_contextualization_study_report.md`
- 层级曲线图: `layerwise_kv_contextualization.png`

## 1. 实验范围

本轮实验是针对 `libero_spatial` 的第一轮 all-task agentview 版本，用于验证 layer-wise KV contextualization 的测量链路是否稳定，以及初步观察 target/background、prompt、phase 对 K/V/H 相似度的影响。

| 项目 | 数值 |
|---|---:|
| Task suite | `libero_spatial` |
| Tasks | `0-9` |
| 每个 task 的 episode 数 | 2 |
| 完成 episode 数 | 20 |
| Camera | `agentview` |
| Rollout policy | OpenVLA model |
| Max rollout steps | 160 |
| Sampling interval | 10 |
| 每个 episode 最大采样数 | 10 |
| CSV 总行数 | 116736 |
| 有效非 NaN 行数 | 104640 |
| NaN 行数 | 12096 |

NaN 行主要来自 target patch 在当前视角下不可见，或者经过 3D oracle depth consistency 过滤后没有有效对应 patch。后续统计 target 相关指标时，应显式过滤 `n_pairs > 0`。

## 2. Control 检查

sanity controls 通过。

| Control | K mean | V mean | H mean | 解释 |
|---|---:|---:|---:|---|
| S0 same-frame rerun | 1.000 | 1.000 | 1.000 | 同一帧重复前向，K/V/H 完全一致，说明抽取和 cosine 计算链路稳定。 |
| S1 random same-frame patch pairs | 0.669 | 0.231 | 0.402 | 随机 patch pair 提供无对应情况下的 floor，尤其 V/H 明显较低。 |

S0 全层精确为 1.0 是本轮实验最重要的正确性检查：说明 hidden state、K/V 抽取、patch 对齐和 cosine 聚合本身没有明显随机性或错位问题。S1 则说明相似度不是对任意 patch 都天然偏高，因此后续 oracle pair 的高相似度是有意义的。

## 3. 数据质量与覆盖情况

phase 修复后，Grid B 已经具备可分析的 phase 覆盖。

| Phase | 行数 | 唯一样本数 | 说明 |
|---|---:|---:|---|
| reach | 16704 | 87 | 主要来自接近目标前的阶段。 |
| grasp | 2304 | 12 | 来自 eef-target 距离首次进入阈值半径，以及其后的 transition window。 |
| transport | 17856 | 93 | 主要来自 eef 接近 target 且夹爪未打开的阶段。 |
| place | 768 | 4 | 仅来自 episode 成功 `done` 后追加的 terminal pair。 |

target patch 的有效对应仍然比较稀疏。

| Group | zero-pair rows |
|---|---:|
| Grid A target | 8064 |
| Grid B target | 4032 |

这与 `agentview` 视角有关：目标物体在部分时刻被遮挡、面积太小，或没有通过深度一致性过滤。该问题不会让整个实验失效，但会影响 target 相关结论的置信度。后续正式统计应同时报告有效样本数，并过滤 `n_pairs == 0`。

## 4. Grid A: Prompt x Patch Semantics

过滤无效 target 行后，P0 和 P3 在所有视觉 patch K/V/H slot 上完全一致。

| Patch group | Modality | P0 mean | P3 mean | P0-P3 |
|---|---:|---:|---:|---:|
| background | K | 0.994 | 0.994 | 0.000 |
| background | V | 0.968 | 0.968 | 0.000 |
| background | H | 0.971 | 0.971 | 0.000 |
| target | K | 0.972 | 0.972 | 0.000 |
| target | V | 0.924 | 0.924 | 0.000 |
| target | H | 0.918 | 0.918 | 0.000 |

按层段聚合后也得到相同现象：early、mid、late layers 的 P0-P3 差异均为 0。

| Patch group | Modality | Early 0-7 | Mid 8-23 | Late 24-31 |
|---|---:|---:|---:|---:|
| background | K | 0.998 | 0.995 | 0.988 |
| background | V | 0.987 | 0.974 | 0.939 |
| background | H | 0.987 | 0.977 | 0.943 |
| target | K | 0.989 | 0.978 | 0.942 |
| target | V | 0.964 | 0.943 | 0.847 |
| target | H | 0.963 | 0.936 | 0.837 |

解释：

1. 当前 Grid A 的 visual patch K/V/H slot 测量没有观察到 prompt-dependent effect。
2. 这不能直接解释为“语言 prompt 对 OpenVLA 没有影响”。
3. 更可能的原因是测量位置不对：当前比较的是视觉 patch token 自身的 K/V/H 状态；在 causal transformer 顺序下，后面的语言或动作 token 不一定能反向改变已经产生的视觉 token slot。
4. 因此，当前 Grid A 更适合作为 prompt-invariant control，而不是最终的 language contextualization probe。

如果要真正测 prompt contextualization，下一步应改测：

- action-token hidden states 在 P0/P3 下的差异；
- action-token query 对 target/background visual patches 的 attention 差异；
- P0/P3 下 action logits 或 action prediction 的变化；
- 或者把 visual slot 作为对照，同时另设 text/action-side probe。

尽管 prompt effect 为 0，Grid A 仍然显示 target/background 之间有区别：target patch 的跨帧相似度低于 background，尤其 late-layer V/H 更明显。

## 5. Grid B: Episode Phase x Patch Semantics

Grid B 显示出明确的 phase effect，最强信号出现在 target patch 的 V/H 上。

### 5.1 各 phase 的平均相似度

background patch 整体非常稳定。

| Phase | K | V | H |
|---|---:|---:|---:|
| reach | 0.993 | 0.963 | 0.966 |
| grasp | 0.996 | 0.980 | 0.982 |
| transport | 0.994 | 0.972 | 0.974 |
| place | 0.996 | 0.981 | 0.983 |

target patch 对 phase 更敏感。

| Phase | K | V | H |
|---|---:|---:|---:|
| reach | 0.985 | 0.952 | 0.950 |
| grasp | 0.959 | 0.874 | 0.869 |
| transport | 0.956 | 0.894 | 0.882 |
| place | 0.968 | 0.930 | 0.920 |

主要模式：

- background K/V/H 在不同 phase 中保持较高相似度。
- target K/V/H 在 grasp 和 transport 阶段明显下降。
- V/H 比 K 更敏感。
- place 高于 grasp/transport，但 place 只有 4 个唯一采样点，因此只能作为方向性观察，不能过度解释。

### 5.2 phase effect 最强的层

最大 phase spread 集中在 late layers，尤其 target V/H。

| Patch group | Modality | 最大 phase spread | Layer |
|---|---:|---:|---:|
| background | K | 0.010 | 30 |
| background | V | 0.046 | 30 |
| background | H | 0.055 | 31 |
| target | K | 0.072 | 30 |
| target | V | 0.136 | 31 |
| target | H | 0.141 | 31 |

这支持一个初步层级假设：

- early layers 更偏向保持图像局部对应关系；
- late layers 开始体现 object/phase/task interaction；
- target V/H 是当前测量中最强的 temporal/object contextualization 信号；
- 重点观察层可以先放在 27-31 层。

## 6. 当前可支持的结论

本轮实验可以支持：

1. 测量链路稳定：S0 精确为 1.0，S1 提供了合理 floor。
2. 3D oracle correspondence 可以在 10 个 LIBERO-Spatial task 上工作。
3. target patch 的跨帧相似度低于 background patch，尤其在 late-layer V/H 上。
4. phase effect 是存在的：target V/H 在 grasp/transport 阶段显著下降。
5. 当前最值得关注的 contextualization 层段是 late layers，约 27-31 层。

本轮实验还不能支持：

1. prompt-dependent visual-token effect。当前 P0/P3 在视觉 patch K/V/H slot 上完全一致。
2. 关于 place phase 的强结论。place 只有 4 个唯一采样点。
3. 最终统计显著性结论。当前是 all-task e2，不是计划中的 e5/full robustness pass。

## 7. 建议的下一步

1. 增加一个正式分析脚本，统一过滤 `n_pairs == 0`，按 layer/phase/semantic 聚合，并加入 bootstrap confidence interval。
2. 重设 prompt 条件的 probe：
   - 比较 P0/P3 下的 action-token hidden states；
   - 比较 action-token query 到 target/background visual patches 的 attention；
   - 比较 P0/P3 下的 action logits 或最终 action prediction；
   - 保留 visual K/V/H 作为 prompt-invariant control。
3. 跑完整 `EPISODES=5` 的 agentview 版本，以增强 phase 统计。
4. 跑 `CAMERA=robot0_eye_in_hand` wrist-camera 版本，验证 target visibility 和 motion magnitude 的稳健性。
5. 在所有 target-patch 图表和表格中同步报告有效 pair 数，避免被 zero-pair/NaN 行误导。

## 8. 总结

这轮 all-task/e2 实验成功验证了 geometry、phase、K/V/H 抽取和 layer-wise 聚合链路，也观察到了清晰的 late-layer target-phase effect。最强的变化出现在 target V/H 的 late layers，尤其 grasp/transport 阶段。

但是，当前 Grid A 的 visual patch K/V/H slot 比较没有测到 prompt effect，P0 和 P3 完全相同。因此，若研究目标是证明语言 prompt 如何 contextualize 视觉信息，下一步必须把 probe 从 visual-token slot 转向 action-token states、attention 或 action logits。
