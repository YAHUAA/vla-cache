# Motion-Compensated VLA-Cache: 完整思路与实验方案

## 1. 背景与问题

现有 VLA-Cache 的核心假设是：相邻控制帧中大量视觉 token 在同一图像网格位置上几乎不变，因此可以跳过这些静态 token 的重复计算，并复用上一帧的 KV cache。这个假设在固定相机、短时 manipulation 中通常成立，但在下面几类场景会快速失效：

- 机器人底座移动或头部相机转动，导致背景和目标在图像网格上整体平移、旋转或缩放。
- wrist camera 随末端执行器移动，画面中大部分像素都有明显 optical flow。
- 长程任务中视角持续变化，当前位置的 patch `j` 往往对应上一帧另一个 patch `i` 的世界内容。
- 任务物体、夹爪、遮挡边界和新出现区域是真动态区域，不能简单复用。

因此，原始 grid-level reuse 会遇到两个问题：

1. **漏复用**：同一个世界点移动到了新网格，逐位置 patch 差分认为它变了。
2. **错复用**：如果只用 motion compensation 判定当前 patch 静止，却不移动 cache，那么当前 patch `j` 仍会读取上一帧 `j` 位置的 KV，而真实对应内容可能在上一帧 `i`。

要让这个方向成立，motion compensation 不能只是一个更好的静态区域检测器，而应该输出 **跨帧 token correspondence**，并驱动视觉 token 或 KV cache 的重映射。

## 2. 核心命题

建议把方法定义为 **Motion-Compensated VLA-Cache, MC-VLA-Cache**：

> 在每个控制 step，先用 2D/3D 运动补偿把上一帧的视觉 token 或 KV cache 对齐到当前相机视角，再只重算未对齐区域、低置信区域、遮挡/新显露区域、动态物体区域和任务相关区域。

这个方法的目标不是简单提高静态 token 数量，而是让“静态”从图像坐标静态升级为 **世界坐标静态**：

```text
原始 VLA-Cache:
  current_patch[j] ~= prev_patch[j]  -> reuse prev KV[j]

MC-VLA-Cache:
  current_patch[j] corresponds to prev_patch[i] in the world
  and confidence(i -> j) is high
  and patch is not task-critical/dynamic
  -> reuse remapped prev KV[i] as current KV[j]
```

## 3. 应该复用什么

结论：**主线应该做 motion-compensated KV reuse，视觉 token reuse 作为诊断、消融和低风险 fallback。**

| 复用对象 | 优点 | 难点 | 预期收益 | 建议定位 |
|---|---|---|---|---|
| 像素/patch 输入 | 实现直观，可视化容易 | ViT 自注意力是全局的，只替换局部 patch 不一定省 encoder 计算 | 很低 | 不作为主线 |
| vision encoder 输出 token / projector 后视觉 token | 不碰 LLM KV，位置语义更容易控制，适合验证 correspondence 质量 | 如果仍完整跑 LLM，主耗时没省；若想局部跑 ViT，需要较大结构改造 | 低到中 | 第一阶段消融和 fallback |
| LLM per-layer KV cache | 直接省掉 VLA-Cache 当前最主要的 LLM token 计算，和现有代码路径最接近 | 需要 cache remap、位置编码校正、遮挡过滤、周期刷新 | 高 | 主贡献 |
| action token / action chunk reuse | 与视觉 cache 正交，可叠加 | 需要判定 action 稳态，容易影响闭环纠偏 | 中 | 后续扩展 |

原因如下：

- 现有仓库的 VLA-Cache 已经把主要加速点放在 LLM 侧 token pruning 和 `past_key_values` 复用上，代码入口集中在 `experiments/robot/openvla_utils.py` 和 `experiments/robot/vla_cache_utils.py`。
- OpenVLA 类模型中，视觉 encoder 和 projector 只是一部分成本；每个控制 step 的语言 backbone decoding 与 action token generation 仍然很重。
- 视觉 token reuse 更适合回答“几何对齐是否正确”和“复用后动作是否接近 full recompute”，但单独使用时不一定产生足够 latency gain。

更稳的最终形态是 **hybrid**：

1. 前几层或低置信区域重算，用来吸收位置和上下文变化。
2. 中高层对高置信、非任务关键、世界静态 token 做 KV remap reuse。
3. 每隔 `N` 帧或在 attention/动作发生突变时 full refresh。

## 4. 方法设计

### 4.1 Motion Compensation 模块

模块输入：

- 当前 RGB 图像 `I_t`，上一帧 RGB 图像 `I_{t-1}`。
- 可选深度 `D_t, D_{t-1}`。
- 可选相机内参 `K`、外参或相机位姿 `T_t, T_{t-1}`。
- 可选机器人状态、末端位姿、camera-to-robot calibration。
- 上一帧 attention map、上一帧 cache、上一帧视觉 token。

模块输出：

- `corr`: 当前 patch `j` 到上一帧 patch `i` 的对应关系。
- `conf`: 每个 correspondence 的置信度。
- `visible`: 是否通过 z-buffer / depth consistency 检查。
- `ego_flow`: 由相机运动解释的光流。
- `residual_flow`: 真实光流减去 ego flow 后的残差，用来发现动态物体。
- `reuse_mask`: 可复用的当前 token index。
- `refresh_mask`: 必须重算的当前 token index。

### 4.2 3D 对齐路径

仿真和 RGB-D 真实系统中优先使用 3D 路径，因为它最容易得到高质量 oracle：

```text
上一帧像素 u_{t-1}
  -> 用 D_{t-1}(u) 和 K 反投影到上一帧相机坐标
  -> 用 T_{world<-cam,t-1} 转到世界坐标
  -> 用 T_{cam,t<-world} 转到当前相机坐标
  -> 用 K 投影到当前像素 u_t
  -> z-buffer / depth consistency / patch overlap 过滤
```

对 ViT patch 级别，可以对每个 source patch 中的若干采样点投影到当前图像，统计落入 target patch 的面积比例或投票数，得到 `i -> j` 的 patch correspondence。

高置信复用条件建议为：

```text
conf(i, j) >= tau_conf
depth_consistency(i, j) <= tau_depth
residual_flow(j) <= tau_residual
patch_is_not_disoccluded(j)
patch_is_not_task_relevant(j)
```

### 4.3 2D 对齐路径

当深度或位姿不可用时，使用 2D 路径作为 fallback：

- 全局 homography / ECC：适合纯旋转、远景背景或近似平面桌面。
- 稀疏特征匹配 + RANSAC：适合纹理丰富场景。
- 光流模型：可处理非刚性局部位移，但计算成本和部署复杂度更高。
- IMU/odometry 辅助的 homography 初始化：适合移动机器人。

2D 路径要更保守：只把高置信、一致性强、远离遮挡边界的 token 放入 reuse mask。

### 4.4 动态区域与任务相关区域 veto

运动补偿后仍然不能复用全部静态世界点。建议保留原始 VLA-Cache 的 task relevance 思路，并加入几类 veto：

- **attention veto**：上一帧 text-to-vision attention top-k 和当前高 saliency 区域重算。
- **residual motion veto**：真实光流无法被相机运动解释的 patch 重算。
- **object/contact veto**：夹爪、目标物体、容器边界、接触区域重算。
- **occlusion veto**：z-buffer 竞争失败、新显露、深度突变区域重算。
- **uncertainty veto**：multi-match、低纹理、低 conf patch 重算。
- **periodic refresh**：每 `N=4/8/16` 帧强制全量刷新一次，限制 cache staleness。

## 5. KV Cache 重映射设计

### 5.1 为什么需要 remap

如果上一帧 patch `i` 在当前帧变成 patch `j`，仅把 `j` 加入 `reusable_patches` 是不够的。因为现有 cache 里 `j` 位置保存的是上一帧旧 `j` 内容，而不是旧 `i` 内容。正确做法是把上一帧 visual token `i` 的 cache 搬到当前 token 位置 `j`。

抽象接口：

```python
remapped_cache = remap_visual_kv_cache(
    past_key_values=last_cache,
    correspondences=[(prev_token_i, curr_token_j, confidence), ...],
    visual_token_start=1,
    num_visual_tokens=256,
    position_mode="rope_correct",
)
```

### 5.2 位置编码问题

LLaMA 类 decoder 通常对 query/key 使用 RoPE。上一帧 token `i` 的 key 已经带有位置 `pos_i` 的旋转，如果直接放到当前 token `j`，它的相对位置会错。

建议比较三个版本：

- **No position correction**：直接 remap KV，作为最低成本 baseline。
- **Key RoPE correction**：对 key 做 `R(pos_j) R(pos_i)^{-1}` 的旋转修正，value 不需要 RoPE 修正。
- **Early-layer recompute + late-layer remap**：前 `L0` 层重算，从中高层开始 remap，降低底层位置误差和上下文误差。

### 5.3 层级策略

建议默认配置：

- 第 0-1/2 层不复用，保证当前帧的低层视觉-语言交互重新建立。
- 中层使用较低 reuse ratio。
- 高层使用 attention entropy 和 confidence 共同决定 reuse ratio。
- 对 attention 高峰区域始终重算。

每层复用分数可以写成：

```text
reuse_score_l(j) =
  geometry_conf(j)
  * visibility(j)
  * static_prob(j)
  * (1 - task_relevance(j))
  * layer_reuse_budget(l)
```

选择 `top_k_l` 个 reuse score 最高的 token 进行该层 KV reuse。

## 6. 与现有 VLA-Cache 的关系

现有实现已经包含三个可复用组件：

- `find_static_patches`: 当前是逐网格 patch cosine similarity。
- `task_relevant_selection`: 用 attention top-k 排除任务关键 token。
- `get_layer_mask_schedule`: 用 attention entropy 调整 per-layer reuse ratio。

MC-VLA-Cache 可以最小侵入地替换第一步和扩展 cache 管理：

```text
原始:
  stable_patches = find_static_patches(image_t, image_t-1)
  remaining = stable_patches - task_relevant
  model.config.reusable_patches = remaining

改造:
  corr, conf, visible = estimate_patch_correspondence(obs_t, obs_t-1)
  reuse_candidates = filter_by_geometry_and_dynamics(corr, conf, visible)
  remaining = reuse_candidates - task_relevant
  prompt_cache = remap_visual_kv_cache(prompt_cache, corr)
  model.config.reusable_patches = current_token_indices(remaining)
```

关键差异是：`reusable_patches` 表示当前帧哪些 token 可以跳过，而 `remap_visual_kv_cache` 保证这些当前 token 的 KV 已经由正确的上一帧 token 搬运过来。

## 7. 实验总目标

验证三个假设：

1. **H1: 对齐质量**  
   相机自运动下，motion compensation 能显著恢复跨帧 token correspondence，reuse hit precision 高于原始逐网格差分。

2. **H2: 动作一致性**  
   在 full recompute 作为 teacher 的离线评测中，MC-KV reuse 的 action token / continuous action 偏差小于原始 VLA-Cache。

3. **H3: 在线收益**  
   在有明显 camera motion 的 LIBERO / SIMPLER / wrist-camera 任务中，MC-VLA-Cache 相比原始 VLA-Cache 有更高可复用率、更低 latency，并保持接近 full recompute 的成功率。

## 8. 实验设置

### 8.1 环境与模型

优先顺序：

1. **OpenVLA + LIBERO-Spatial**：和现有 `vla-cache` 路径最接近，先复现实验。
2. **OpenVLA-OFT + 双视角输入**：包含 primary + wrist，更能体现相机运动。
3. **LIBERO-Long / Goal / Object**：验证长程和语义阶段切换。
4. **SIMPLER 或真实机器人数据**：作为后续泛化验证。

大文件和生成数据遵守项目规则，放在：

```text
/mnt/data0/zjh_data/Embodied_Proj/datasets/mc_vla_cache/
/mnt/data0/zjh_data/Embodied_Proj/checkpoints/mc_vla_cache/
/mnt/data0/zjh_data/Embodied_Proj/outputs/mc_vla_cache/
```

### 8.2 Camera motion 条件

为避免一开始就被真实传感器噪声卡住，建议先做可控扰动：

| 条件 | 描述 | 用途 |
|---|---|---|
| `static_cam` | 原始 LIBERO 相机 | 确认不破坏原始收益 |
| `pan_tilt` | 相机 yaw/pitch 小幅摆动 | 测试全局视角变化 |
| `translate_xy` | 相机平移导致背景和物体位移 | 测试 parallax |
| `wrist_motion` | wrist camera 随末端移动 | 测试真实 manipulation 难点 |
| `long_horizon` | 多阶段任务，相机和语义焦点都变 | 测试 cache staleness |

### 8.3 对比方法

必须包含：

- `Full recompute`: 不使用 cache。
- `Original VLA-Cache`: 原始逐网格 patch diff + attention veto。
- `MC-token`: 只复用或替换 projector 后视觉 token，不做 KV remap。
- `MC-KV-no-rope`: 几何 remap KV，但不做位置修正。
- `MC-KV-rope`: 几何 remap KV，并对 key 做 RoPE position correction。
- `MC-KV-rope-refresh`: 上一项加 periodic refresh 和 semantic/action-change refresh。

可选：

- `Oracle-3D-MC`: 使用仿真真值深度、相机位姿和 segmentation。
- `Estimated-2D-MC`: 只用 RGB 估计 homography/flow。
- `Action-reuse + MC-KV`: 后续与 action reuse 叠加。

### 8.4 指标

效率指标：

- per-step latency, first-step latency, steady-state latency。
- CUDA time 和 CPU preprocessing time 分开统计。
- control frequency。
- TFLOPs 或 profiler 估计 FLOPs。
- token reuse ratio: primary / wrist / total。
- cache remap 开销。
- VRAM 峰值。

正确性指标：

- task success rate。
- action L2 / action token accuracy，相对 full recompute。
- episode return / completion step。
- intervention count 或 failure type。

cache 质量指标：

- correspondence precision / recall，相对 oracle 3D 或 full recompute matching。
- false reuse rate: 动态 token 被复用比例。
- false refresh rate: 静态 token 被重算比例。
- reused KV 与 full recompute KV 的 cosine similarity。
- reused token attention drift: 复用后 attention top-k 与 full recompute 的重合度。
- occlusion/disocclusion 检测准确率。

## 9. 分阶段实验路线

### Phase 0: 复现与基线固化

目标：

- 复现现有 VLA-Cache 在 `static_cam` 下的 latency、reuse ratio、success rate。
- 固定随机种子、任务列表、日志格式和可视化格式。

产出：

- `Full recompute` vs `Original VLA-Cache` 的基线表。
- 每 step 保存 `image_t`, `prev_image`, `stable_patches`, `attention_topk`, `reusable_patches`。

通过标准：

- 原始 VLA-Cache 在无相机扰动下不明显退化。
- 日志中可重放任意 episode 的 patch 选择过程。

### Phase 1: Motion compensation 诊断，不接 VLA

目标：

- 在有真值 depth/pose 的仿真数据上，比较逐网格 patch diff、2D homography、3D projection 的 correspondence 质量。
- 找到 `tau_conf`, `tau_depth`, `tau_residual` 的合理范围。

实验：

```text
static_cam / pan_tilt / translate_xy / wrist_motion
  x grid_diff / 2d_homography / 3d_depth_pose
  x patch_size 14
  x top_k {64, 96, 128, 160}
```

产出：

- correspondence precision/recall 曲线。
- patch overlay 视频：source patch、target patch、dynamic veto、occlusion veto。
- motion compensation runtime。

通过标准：

- 在 `pan_tilt` 和 `translate_xy` 下，3D-MC 的 static correspondence recall 明显高于 grid diff。
- false reuse rate 可被控制在 5%-10% 以下。

### Phase 2: 离线 teacher-forcing cache 正确性

目标：

- 不跑在线控制，先用 full recompute 作为 teacher，评估不同复用策略导致的 hidden/KV/action 偏差。

做法：

1. 对同一段 episode 保存 full recompute 的视觉 token、各层 KV、action logits/action。
2. 对相同输入运行 `MC-token`, `MC-KV-no-rope`, `MC-KV-rope`。
3. 比较每层 reused KV 与 teacher KV 的 cosine similarity。
4. 比较最终 action 与 teacher action 的误差。

关键消融：

- remap only vs remap + RoPE correction。
- 从第几层开始复用：`L0 in {0, 2, 4, 8}`。
- refresh interval: `N in {4, 8, 16, inf}`。
- attention veto top-k: `{64, 96, 120, 160}`。

通过标准：

- `MC-KV-rope` 的 action deviation 小于 `MC-KV-no-rope`。
- 在明显 camera motion 下，`MC-KV-rope` 小于原始 VLA-Cache 的 action deviation。

### Phase 3: 视觉 token reuse 消融

目标：

- 回答“复用图像 token 是否值得做”。
- 为 KV remap 提供低风险对照组。

实验：

- `MC-token-input`: 把 projector 后当前视觉 token `j` 替换为上一帧 token `i`。
- `MC-token-blend`: `token_j = alpha * token_i_prev + (1-alpha) * token_j_current`。
- `MC-token-stopgrad`: 只用于 inference，不训练。

观察：

- 如果动作几乎不变，说明几何 correspondence 可靠。
- 如果 latency 没明显下降，证明主线应转向 KV reuse。
- 如果 token 替换造成错误，说明 KV reuse 必须更保守，或需要前几层重算。

通过标准：

- action deviation 可控。
- 明确量化 token reuse 对 latency 的上限收益。

### Phase 4: MC-KV remap 实现

目标：

- 在现有 VLA-Cache 推理路径中加入 cache remap。

建议新增模块：

```text
experiments/robot/motion_compensation.py
experiments/robot/cache_remap_utils.py
```

建议新增配置：

```text
--use_vla_cache True
--use_motion_compensation True
--mc_mode {oracle_3d, depth_pose_3d, homography_2d, flow_2d}
--mc_reuse_target {token, kv, hybrid}
--mc_rope_correction True
--mc_min_conf 0.75
--mc_depth_thresh 0.03
--mc_residual_flow_thresh 2.0
--mc_refresh_interval 8
--mc_recompute_early_layers 2
```

实现要点：

- 对 single-image OpenVLA，visual token 范围通常是 `[1, 256]`。
- 对 OpenVLA-OFT 双图像，primary 和 wrist token 范围要分别处理，例如 primary `[1, 256]`，wrist `[257, 512]`。
- remap 后的 cache 要保证当前帧 token index 与 cache slot 对齐。
- 对一对多 correspondence，只保留最高 confidence，其他 target 重算。
- 对多对一 correspondence，只保留 z-buffer 最近或 overlap 最大的 source。
- 不确定区域宁愿重算，不要复用。

通过标准：

- cache remap 后模型能完整 rollout，不出现 shape/cache_position 错误。
- `MC-KV-rope-refresh` 在 `static_cam` 下不慢于原始 VLA-Cache 太多。
- 在 `pan_tilt/translate_xy` 下 token reuse ratio 和 action consistency 明显优于原始 VLA-Cache。

### Phase 5: 在线控制评测

目标：

- 验证真实闭环控制中是否加速且不掉成功率。

推荐任务矩阵：

```text
LIBERO-Spatial: 10 tasks x 5 rollouts
LIBERO-Object: 10 tasks x 5 rollouts
LIBERO-Goal: 10 tasks x 5 rollouts
LIBERO-Long: 10 tasks x 3 rollouts
```

如果资源有限，先做：

```text
LIBERO-Spatial: 10 tasks x 1 rollout smoke
LIBERO-Spatial: 10 tasks x 5 rollouts main
OpenVLA-OFT wrist setting: 5 tasks x 3 rollouts
```

报告表格：

| 方法 | camera motion | success | steady latency | speedup | reuse ratio | false reuse | action L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| Full recompute | pan_tilt | | | 1.00x | 0 | 0 | 0 |
| Original VLA-Cache | pan_tilt | | | | | | |
| MC-KV-rope-refresh | pan_tilt | | | | | | |

成功标准：

- `static_cam`: MC-VLA-Cache 与原始 VLA-Cache 接近，不引入明显额外开销。
- `camera_motion`: MC-VLA-Cache success rate 接近 full recompute，latency 接近或优于原始 VLA-Cache。
- 相比原始 VLA-Cache，MC-VLA-Cache 在有自运动时 false reuse 更低，reuse hit precision 更高。

## 10. 预期结果与可能结论

最理想结果：

- 无 camera motion 时，MC-VLA-Cache 退化为原始 VLA-Cache，只有很小额外开销。
- 有 camera motion 时，原始 VLA-Cache 的可复用率下降或错误复用上升；MC-VLA-Cache 能恢复世界静态 token 的复用。
- `MC-KV-rope-refresh` 在中等相机运动下保持成功率，并取得稳定 1.2x-1.5x 级别加速。

可能的中间结论：

- 视觉 token reuse 动作一致性好，但 latency 收益有限。这说明它适合作为安全 fallback 或蒸馏/训练信号，不适合作为主加速路径。
- 直接 KV remap 不稳定，但 early-layer recompute + late-layer remap 稳定。这会成为一个很好的方法点。
- 2D homography 在桌面 manipulation 可用，但 wrist camera 和近距离 parallax 必须用 depth/pose 或 flow residual。

## 11. 主要风险与规避

| 风险 | 影响 | 规避 |
|---|---|---|
| KV 受上下文和位置编码影响，remap 后不是严格等价 | 动作偏差或成功率下降 | RoPE key correction、前几层重算、周期刷新、teacher-forcing 先验验证 |
| dynamic object 被误判为 ego-motion 静态 | 错误复用目标物体或夹爪 | residual flow、segmentation/contact mask、attention veto |
| depth/pose 在真实系统不可用或噪声大 | 真实部署困难 | 先用 oracle 3D 证明上限，再做 estimated 2D/SLAM fallback |
| motion compensation 开销吃掉加速 | speedup 不明显 | patch 级聚合、GPU batched projection、降低 flow 模型频率 |
| 双视角 token index 容易错 | cache corruption | 单视角先做，双视角写单元测试和可视化 |
| 长程语义阶段变化导致 cache stale | 后期动作错误 | semantic/action-change refresh，attention entropy gating |

## 12. 最小可行版本

如果只做一个最小可行实验，建议这样收敛：

1. 使用 OpenVLA + LIBERO-Spatial。
2. 在仿真中保存 RGB、depth、camera pose、原始 action。
3. 构造 `pan_tilt` 和 `translate_xy` 两种相机扰动。
4. 实现 oracle 3D patch correspondence。
5. 先做 offline teacher-forcing，对比：
   - Full recompute
   - Original VLA-Cache
   - MC-token
   - MC-KV-no-rope
   - MC-KV-rope
6. 若 `MC-KV-rope` action deviation 最小，再接 online rollout。
7. 最终报告 success rate、latency、reuse ratio、false reuse、action deviation。

这个 MVP 的价值是：即使还没有真实 SLAM/flow 模块，也能判断“motion-compensated cache remap 是否真的有上限收益”。

## 13. 论文/报告角度的贡献点

可以把贡献写成三点：

1. **从 image-grid static 到 world-static reuse**：指出 VLA-Cache 在 camera ego-motion 下的 grid correspondence 失效问题，并用 motion compensation 恢复跨帧 token 对齐。
2. **Geometry-aware KV cache remapping**：不是只重估 reusable mask，而是把上一帧 KV 按 patch correspondence 搬运到当前 token slot，并处理位置编码和遮挡。
3. **Task- and dynamics-aware refresh policy**：结合 attention relevance、residual flow、occlusion 和 periodic refresh，避免复用任务关键或真实动态区域。

需要注意：VLN-Cache 已经在 navigation 场景讨论了 viewpoint shift 和 semantic dynamics，并提出 view-aligned remapping。因此本文的差异化应落在：

- VLA manipulation 而不是 VLN navigation。
- action-conditioned / attention-conditioned 的任务关键 token veto。
- primary + wrist camera 的近距离 parallax 和 contact dynamics。
- LLM KV cache slot remap、RoPE correction 和 VLA action consistency 评估。
- 在 OpenVLA / OpenVLA-OFT / LIBERO 控制闭环中的 latency-success tradeoff。

## 14. 参考资料

- VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching, arXiv:2502.02175, https://arxiv.org/abs/2502.02175
- VLA-Cache project page, https://vla-cache.github.io/
- VLN-Cache: Enabling Token Caching for VLN Models with Visual/Semantic Dynamics Awareness, arXiv:2603.07080, https://arxiv.org/abs/2603.07080
- Think Twice, Act Once: Token-Aware Compression and Action Reuse for Efficient Inference in Vision-Language-Action Models, arXiv:2505.21200, https://arxiv.org/abs/2505.21200
- VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language Inference, arXiv:2512.12977, https://arxiv.org/abs/2512.12977
