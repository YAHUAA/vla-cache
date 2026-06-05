# VLA KV Contextualization Study —— 实验计划 v2

Date: 2026-06-05
Status: 替代 `Motion_Compensated_VLA_Cache_Experiment_Plan.md` 的方法定位；
        在 `OpenVLA_KV_Contextualization_Layerwise_Study_Design.md` (2026-06-03)
        基础上加入"motion compensation 伪需求"的认定与双网格因子收敛。

---

## 0. TL;DR

放弃 "Motion-Compensated VLA-Cache" 的 framing。motion compensation 在主流 VLA
benchmark 和典型 humanoid manipulation 场景中**接近伪需求**：

- LIBERO / SIMPLER / RoboCasa 主视角固定，要做 ego-motion 必须人工注入扰动；
- humanoid 第一视角通常有 gaze stabilization，manipulation 阶段处于 quasi-static；
- wrist camera 运动大但在 OpenVLA 主输入里权重低；
- mobile manipulation 场景下成功率本身是首要矛盾，cache 加速次要。

但本轮讨论收敛出的**真正实验**——layer-wise KV contextualization 刻画——
独立于 motion compensation。它回答的是 VLA KV cache reuse 的物理上限：

> 同一世界点的 visual token KV，在第几层之前还可以被认为只是它自身的函数，
> 而不依赖整张图的其它部分与 language prompt？

这个问题在固定相机 + oracle segmentation tracking 下同样能测。motion compensation
退到工具层（甚至完全不出现在 paper 主文）。

本文档定义的研究范围：

- 两个二维网格：**prompt × patch_semantics** 与 **episode_phase × patch_semantics**。
- 一套共享的 oracle patch correspondence + KV 抽取 + cosine 测量协议。
- 一组因果干预 (KV substitution) 把表示相似度挂到行为不变性上。
- 一组统计分析（episode 级 bootstrap + 方差分解）把"轶事曲线"升级为"finding"。

---

## 1. 立场转变：为什么砍掉 motion compensation

### 1.1 原 framing 的问题

`Motion_Compensated_VLA_Cache_Experiment_Plan.md` 把方法卖点挂在 motion
compensation 上，存在以下结构性问题：

| 问题 | 解释 |
|---|---|
| **VLN-Cache 已覆盖 view-aligned remapping** | 换数据集和相机视角不构成方法贡献，reviewer 会判 incremental |
| **RoPE correction 是治标** | KV 真实依赖 = (位置, 周围 sequence 的 contextualization)；RoPE 只解决前者 |
| **prefill 占比未知** | OpenVLA 主耗时在 LLM decode，prefill 复用的收益天花板可能 < 20% |
| **patch 粒度太粗** | 14×14 patch 跨物体边界，几何对齐 false reuse 5–10% 已危险 |
| **LIBERO 相机固定** | 必须人工注入扰动，违反"做真实问题"的原则 |
| **kill-switch 缺位** | 没有 "先证明 baseline 在扰动下确实崩" 的两天实验 |

### 1.2 "伪需求" 的认定

用户最后一问触及本质：**第一视角人形机器人在 manipulation 时大部分时间相机
是稳的**。这一观察在三个层面成立：

- benchmark 层面：现有 VLA 评测体系几乎不包含 ego-motion；
- 硬件层面：头部 IMU + gaze stabilization 会主动抑制相机运动；
- 任务结构层面：manipulation 子阶段（reach/grasp/place）是 quasi-static，
  剧烈运动只发生在 locomotion / transition。

→ 把 cache 加速挂在 motion compensation 上，是给一个**不存在的瓶颈**做优化。

### 1.3 但科学问题仍然存在

VLA-Cache 的根本前提 ——"`patch_t[j] ≈ patch_{t-1}[j] ⇒ KV 可复用`"—— 在
contextualization 视角下**未经证明**。它假设 KV 是 patch 自身的函数，但实际上
KV 由 (patch + 周围 sequence + language prompt) 共同决定。

如果这个假设错，VLA-Cache 即使在静态相机下也会**默默地复用错 KV**——
这才是真正未被量化的科学缺口。把研究对象从 "ego-motion 下的 cache" 改成
"contextualization 下的 cache"，方法 scope 缩小，但科学价值升级。

---

## 2. 核心研究问题与假设

### 2.1 主问题

> 在 OpenVLA 7B (Llama-2 backbone, 32 层 decoder) 中，同一世界点的 visual token
> 在两次独立 forward 后，每层的 K 和 V 在不同条件下的相似度曲线是什么形状？

更具体地，刻画 `sim(L)` 这条函数，并研究它如何依赖于：

- **patch_semantics**：该 patch 落在 background 还是 target object 上；
- **prompt**：使用真实 task prompt vs 无关 prompt；
- **episode_phase**：reach / grasp / transport / place。

### 2.2 假设清单

| ID | 假设 | 用哪个量回答 |
|---|---|---|
| H1 | background patch 的 `sim(L)` 对 prompt 几乎不敏感（P0 ≈ P3 全层重合） | 网格 A, background 列 |
| H2 | target patch 存在 prompt 分叉层 `L_lang`，之前视觉主导、之后被 prompt 重塑 | 网格 A, target 列分叉点 |
| H3 | target patch 的 `sim(L)` 随 episode_phase 在某层 `L_phase` 之后显著分化，background 不分化 | 网格 B |
| H4 | `L_lang` 与 `L_phase` 接近 ⇒ contextualization 统一；显著不同 ⇒ 语言/视觉通路解耦 | A、B 对比 |
| H5 | KV substitution 引起的 action L2 误差曲线 `Δa(L)` 与 `1 − sim(L)` 在层级上单调相关 | §7 因果干预 |

H1–H4 是**表示层 finding**；H5 是**行为层验证**，确保 cosine 不是空指标。

### 2.3 这些假设的可发表性

无论曲线落在哪种形状，都是定量贡献：

- 若 `L_lang` 很低（如 2–4 层）→ prompt 在浅层就开始塑形 visual KV，
  KV reuse 的安全层范围极窄，VLA-Cache 隐含假设被证伪一半；
- 若 `L_lang` 很高（如 20+ 层）→ KV reuse 在中层有大块安全区，但要
  warn 高层；
- 若 H4 成立 → 统一 contextualization；不成立 → 提出 "language pathway vs
  visual pathway" 的解耦视图，这是一个独立 finding。

---

## 3. 实验设计

### 3.1 共享测量本体

对每对 patch correspondence `(i, j)`，每层 `L ∈ {0, ..., 31}`：

```
sim_K(L; i, j) = cos( K_prev[L][i].flatten(),  K_curr[L][j].flatten() )
sim_V(L; i, j) = cos( V_prev[L][i].flatten(),  V_curr[L][j].flatten() )
```

K 和 V **分开报告**。两者对下游误差的传播机制不同：

- K 错 → `softmax(QK^T)` 偏 → 关注对象错；
- V 错 → 关注对象对 → 取回信息错。

### 3.2 网格 0：sanity 对照（必须先跑）

```
I_prev = I_curr = I_t   (同一帧)
oracle correspondence: i = j  (identity)
```

预期：所有层 `sim(L) ≈ 1`，偏离量量化数值噪声 + projection 不变性误差。
如果不达标，**pipeline 有 bug，禁止跑后续网格**。

### 3.3 网格 A：prompt × patch_semantics

固定一对帧 `(I_{t-1}, I_t)`，跑 4 次 forward 只换 prompt：

| 条件 | prompt |
|---|---|
| P0 | 原 task prompt（如 LIBERO 自带 instruction） |
| P1 | 通用 prompt：`"do the task"` |
| P2 | 同 task family 错误目标 prompt（换 referent，如 black bowl → red bowl） |
| P3 | 完全无关 task 的 prompt（如 LIBERO-Long 的 instruction） |

按 patch_semantics ∈ {target, background} 分层报告 `sim_K(L)` 和 `sim_V(L)`。

读法：
- 在 target 列，P0 与 P3 在第几层开始显著分叉 = `L_lang(target)`；
- 在 background 列，P0 与 P3 应当全层重合（H1）；
- P1、P2 给出中间程度的扰动，用来检测分叉的渐进性。

### 3.4 网格 B：episode_phase × patch_semantics

固定原 task prompt P0，按 phase ∈ {reach, grasp, transport, place} 分组采样
帧对：

| phase | 选帧策略 |
|---|---|
| reach | gripper open 且与 target 欧氏距离 > 阈值的窗口内取连续帧 |
| grasp | gripper 正在闭合的 ±5 帧（用 gripper state 信号） |
| transport | object-in-hand 且远离 place 区域 |
| place | object-in-hand 且接近目标位置 / gripper 重新打开前后 |

按 patch_semantics ∈ {target, background} 分层报告。

读法：
- 在 target 列，4 条曲线分叉的最早层 = `L_phase(target)`；
- 在 background 列，4 条曲线应当近似重合（H3 对照）。

### 3.5 网格 A 与 B 的对照

把 `L_lang(target)` 与 `L_phase(target)` 画在同一张图上：

- 接近 → contextualization 是统一现象（H4 一侧）；
- 显著分离 → 语言与视觉/阶段两条通路在不同层接管 KV（H4 另一侧）。

---

## 4. 操作化定义（决定结论强度的细节）

### 4.1 patch_semantics

不能凭感觉勾。在 LIBERO 仿真里用 segmentation mask + prompt parsing：

- **target**：language prompt 里 named entity 对应的 object mask 占该 14×14 patch
  面积 ≥ 50%；
- **background**：桌面 / 墙 / 不在 prompt 中的物体，且 patch 内**不含**
  gripper / 任意 named entity / 物体边界；
- 显式排除并**不进入对照**：
  - gripper：每帧位置都变，自带 phase signal；
  - distractor：在 prompt 中未提及但场景中存在的物体（H2 的对照应该用最干净的
    background，不能混入 distractor）；
  - boundary：跨 object/background 边界的 patch（segmentation 面积 30–70% 之间）。

每帧的 patch 分类掩码作为元数据一起保存。

### 4.2 episode_phase

不靠时间四等分。LIBERO 提供 gripper open/close、object-in-hand、object position
信号，定义：

```python
def phase_of(step) -> Phase:
    if not step.object_in_hand and step.gripper_open:
        if dist(step.gripper, step.target) > REACH_THRESH:
            return "reach"
        else:
            return "grasp_pre"   # 进入 grasp 窗口的前半
    if step.gripper_closing_window:   # ±5 帧
        return "grasp"
    if step.object_in_hand and dist(step.gripper, step.place_target) > PLACE_THRESH:
        return "transport"
    if step.object_in_hand and dist(step.gripper, step.place_target) <= PLACE_THRESH:
        return "place"
```

阈值 `REACH_THRESH`, `PLACE_THRESH` 由 episode 长度归一化的 quantile 定，
不在不同任务间硬编码。

### 4.3 frame pair 选取

为消除 frame gap 这一隐藏变量，**固定 Δt = 1**（连续两 control step），
其余 gap 作为 §10 的扩展实验。

### 4.4 patch correspondence

oracle 3D projection：

```text
对每个当前帧 patch j 的中心像素:
  用 depth_t(u_j) 和 K 反投影到当前相机坐标
  用 T_world<-cam_t 转到世界坐标
  用 T_cam_{t-1}<-world 转回上一帧相机
  用 K 投影到上一帧像素，落入哪个 patch 即为 i
  用 z-buffer / depth consistency 过滤遮挡和新显露
```

固定相机时这个 mapping ≈ identity 但**仍需走完一遍**——为了和未来扩展到
ego-motion 的 pipeline 完全一致，也为了在 boundary patch 上正确触发遮挡过滤。

---

## 5. 模型与数据约定

### 5.1 模型

```
Model      : openvla/openvla-7b-finetuned-libero-spatial
Backbone   : Llama-2 7B, num_layers = 32
Image      : 224 × 224, patch_size = 14 → 16 × 16 = 256 visual tokens
Token idx  : visual_token_idx = patch_id + 1     (patch_id ∈ 0..255)
Precision  : bf16 forward；KV 抽取后转 fp32 再算 cosine
```

如需扩展 OpenVLA-OFT 双视角，primary `[1, 256]`、wrist `[257, 512]`
作为两组独立 patch 池。本文档先做 single-view。

### 5.2 数据规模

| 因子 | levels | 数量 |
|---|---|---|
| task family | Spatial, Object, Goal, Long | 4 |
| task / family | | 10 |
| episode / task | | 5 |
| 每 episode 取的 frame pair | Δt = 1，按 phase 分桶后每桶 ≤ 3 对 | ~12 |
| oracle 通过过滤的 patch pair / frame | | ~80 |

总量：`4 × 10 × 5 × 12 × 80 ≈ 192k patch pair`，cosine 测量本身廉价，
forward 才是瓶颈。每 episode 跑 4 次 forward（网格 A 的 4 个 prompt）+
1 次按 phase 采样的 forward 复用。

### 5.3 大文件路径

遵守项目规则：

```
/mnt/data0/zjh_data/Embodied_Proj/outputs/kv_contextualization_study/
    raw_kv/                # 每 episode 每 prompt 的 K, V 张量（bf16，压缩）
    correspondences/       # 每 frame pair 的 (i, j, conf, visible, semantics)
    sim_curves/            # 聚合后的 sim_K(L), sim_V(L) 数组
    figures/               # 论文用 figure 源 svg + png
```

代码侧仓库里只放 thin loader 和 plotting script。

---

## 6. 测量协议

### 6.1 KV 抽取

修改 OpenVLA forward 路径，在每层 self-attention 算完 K, V 后 hook 出
visual token 范围：

```python
# 伪代码
for layer_idx, layer in enumerate(model.language_model.model.layers):
    layer.self_attn.register_forward_hook(
        lambda mod, inp, out, l=layer_idx:
            save_kv(l, out.k_cache[:, visual_slice], out.v_cache[:, visual_slice])
    )
```

注意：

- 只抽 **visual token slice**，language token 不存（无意义且太大）；
- 抽取在 RoPE 之后（key 已经带位置旋转），后续 §7 因果干预再决定是否做
  RoPE 反旋；
- 保存格式：`[num_layers, num_heads, num_visual_tokens, head_dim]`，bf16，
  zstd 压缩。

### 6.2 相似度计算

对每个 patch pair `(i, j)`、每层 `L`、每对条件 `(c1, c2)`：

```python
sim_K = cosine( K_{c1}[L][i].flatten(),  K_{c2}[L][j].flatten() )
sim_V = cosine( V_{c1}[L][i].flatten(),  V_{c2}[L][j].flatten() )
```

flatten 在 `num_heads × head_dim` 维度上做。**不**对 head 取平均——
不同 head 的可复用性可能差异极大，留作 §10 扩展。

---

## 7. 统计分析

### 7.1 聚合单位

patch pair 在同一 episode 内**不独立**（共享 prompt、共享 context）。
所有 CI 在 episode 层做：

```
sim_K_episode_mean[e, L] = mean over patch pairs in episode e
然后 bootstrap over episodes:
  sample episodes with replacement, recompute mean, 1000 次
  报 sim_K_mean, sim_K_p2.5, sim_K_p97.5
```

### 7.2 方差分解

对 `sim_K(L)` 在每层 `L` 上做线性混合模型：

```
sim ~ patch_semantics + prompt + phase + (1 | task_family/task/episode)
分解: Var(task_family) + Var(task|family) + Var(episode|task) + Var(residual)
```

- 若 `Var(task_family)` 主导 → 单一曲线不存在，paper 必须按 task family 报；
- 若 `Var(episode|task)` 主导 → episode 漂移大，承认不稳定，加大 N；
- 若 `Var(residual)` 主导 → patch_semantics + prompt + phase 是真正的解释变量，
  H1–H4 站得住。

### 7.3 效应量

报 `sim(P0) − sim(P3)` 在每层的差值与 95% CI，**不报 p 值**。
分叉层 `L_lang` 定义为：

```
L_lang = min { L : sim(P0)[L] − sim(P3)[L] > Δ_min  AND  CI 不跨 0 }
Δ_min  = 0.05  (经验阈值，pilot 后可调)
```

---

## 8. 行为级因果干预（H5）

cosine 终究是代理指标。补一组**因果实验**，把表示相似性挂到行为不变性上：

```
对同一帧 t:
  baseline: 全部 fresh recompute → action_full
  treatment(L*): 仅在 layer L* 把 K[visual], V[visual] 替换为 K_prev, V_prev (oracle 对应)
                  其他层不动
  metric:
    Δa(L*) = || action_treatment(L*) − action_full ||_2
    top1_agree(L*) = (argmax action_treatment[0] == argmax action_full[0])
```

对每个 `L* ∈ {0, 2, 4, ..., 30}` 跑一遍。

预期：`Δa(L*)` 与 `1 − sim_V(L*)` 在层级上单调相关。如果不相关，cosine 不能
作为可复用性的指标，**整个 §3 的曲线解释要重做**。

干预实验需要两组：

- **with RoPE correction**：把 K 做 `R(pos_j) R(pos_i)^{-1}` 修正后再替换；
- **without RoPE correction**：直接替换。

两条 `Δa(L*)` 曲线的差距 = RoPE correction 的真实价值。这是 RoPE correction
唯一值得做的实验，不要在主线方法里夸大它的作用。

---

## 9. 通过 / 失败判定

| 阶段 | 通过标准 | 失败动作 |
|---|---|---|
| 网格 0 sanity | 所有层 `sim ≥ 0.99` | 修 pipeline |
| 网格 A | H1 在 background 列成立（P0-P3 差值全层 < 0.05） | 检查 patch_semantics 标注 |
| 网格 A | H2 在 target 列分叉，`L_lang` 有定义且 CI 不跨 0 | 若全层重合 → KV 不被 prompt 塑形，paper 角度转向"VLA 的 KV 比预期更视觉主导" |
| 网格 B | H3 在 target 列分叉 | 若不分叉 → episode_phase 不是 KV 漂移源 |
| 网格 A vs B | H4 给出明确结论（统一 or 解耦） | 两条都可发表 |
| 因果干预 | `Δa(L*)` 与 `1 − sim_V(L*)` Spearman ρ > 0.7 | cosine 不可作为代理，换 metric（per-head sim 或 attention drift） |

任一阶段失败都不是 sunk cost——失败结果本身就是 finding，可写成
"contrary-to-expectation" 一节。

---

## 10. 实施路线

### Phase 0：基础设施（1 周）

- KV hook + visual token 抽取，写好 bf16 → fp32 + zstd 序列化；
- 网格 0 sanity 跑通；
- patch_semantics 标注 pipeline（segmentation 复用 LIBERO 自带 mask）；
- episode_phase 分类器（用 gripper / object_in_hand 信号）。

### Phase 1：网格 A（1 周）

- 10 个 LIBERO-Spatial task，每 task 5 episode，每 episode 4 prompt forward；
- 产出 `sim_K(L)`, `sim_V(L)` 曲线 + 方差分解；
- 决定 `L_lang(target)` 是否存在。

### Phase 2：网格 B（1 周）

- 同 10 task 5 episode，按 phase 重新采样 frame pair；
- 产出 phase 分组曲线 + `L_phase(target)`；
- 与 Phase 1 对比，决定 H4。

### Phase 3：因果干预（1 周）

- 实现 KV substitution hook；
- 跑 with / without RoPE correction 两组；
- 把 `Δa(L*)` 曲线和 `sim_V(L*)` 曲线对齐，验证 H5。

### Phase 4：扩展（可选，2 周）

- 跨 task family 重复 Phase 1+2，做方差分解；
- per-head sim 拆分（看是否存在 "language-routing head" 与 "vision-routing head"）；
- Δt ∈ {1, 3, 5, 10} 的 frame gap 扩展（接 cache aging）；
- 切换到 OpenVLA-OFT 双视角验证 wrist patch 是否有不同的 `L_lang`。

---

## 11. 这个研究**不**依赖什么

明确划界，避免被误读为 motion-compensated cache 的 rebrand：

- 不依赖 motion compensation（固定相机 + identity correspondence 即可起步）；
- 不依赖 KV remap 实现（§3 只做 forward 抽取 + cosine，不动 cache）；
- 不依赖 RoPE correction（只在 §8 因果干预的一个 ablation 里出现）；
- 不依赖 ego-motion 扰动注入（避开 LIBERO 相机固定这一伪需求来源）；
- 不依赖 OpenVLA-OFT / 双视角 / mobile dataset（这些是 §10 可选扩展）。

→ 这是一个**最小独立可发表的科学问题**，与 cache 加速工程脱钩。

---

## 12. 可能的后续应用（只在主 study 成立后再启动）

只有当 §9 的判定走完且 H1–H5 给出明确曲线，下列工程方向才有意义：

| 方向 | 触发条件 | 内容 |
|---|---|---|
| **Patch-semantic-aware KV reuse** | H1 + H2 成立 | background patch 在所有层 reuse，target patch 仅在 `L < L_lang` reuse |
| **Phase-aware refresh schedule** | H3 成立 | 在 phase 切换帧（grasp 进入、object 释放）触发 partial refresh |
| **Per-head routing prune** | §10 per-head 扩展成立 | 对 "language-routing head" 永不 reuse、对 "vision-routing head" 全层 reuse |
| **Motion-compensated remap** | H5 + ego-motion benchmark 出现 | 才考虑把这个方向重新拿上桌 |

motion compensation 在这个列表里**最后才轮到**。

---

## 13. 风险

| 风险 | 影响 | 规避 |
|---|---|---|
| segmentation 噪声污染 target/background 标注 | 网格 A、B 的对照失效 | boundary patch 严格排除；标注一致性人工抽样检查 |
| sim cosine 不是动作的好代理 | §8 H5 失败 | 提前准备 per-head sim、attention drift 两个备选 metric |
| KV 数量级太大，磁盘 IO 瓶颈 | Phase 1 跑不动 | 只存 visual slice + bf16 + zstd；必要时只存 K 不存 V 的初版 |
| 跨 task family 方差吞掉 effect | H1–H4 失去普适性 | 先在 LIBERO-Spatial 内部把 finding 立住，再横向扩展 |
| `L_lang` 与 `L_phase` 都很高（在 28+） | KV reuse 安全区几乎不存在 | 仍然是 finding——量化了 VLA-Cache 的理论上限非常窄 |
| reviewer 质疑只在 OpenVLA 7B 上做 | 普适性不足 | §10 扩展到 OpenVLA-OFT 至少 1 个 size，承认 scope |

---

## 14. 与现有文档的关系

- **替代**：`Motion_Compensated_VLA_Cache_Experiment_Plan.md` 的方法定位部分。
  motion compensation 不再是主线，相关工程实现降级到 §12。
- **延续与扩展**：`OpenVLA_KV_Contextualization_Layerwise_Study_Design.md`
  (2026-06-03) 的双网格设计。本文档新增：
  - §1 motion compensation 伪需求认定的论证链；
  - §4.1 / §4.2 patch_semantics 与 episode_phase 的客观操作化定义；
  - §7 episode 级聚合 + 方差分解 + 效应量阈值；
  - §8 因果干预与 RoPE correction 的合理定位；
  - §11 明确不依赖项；
  - §12 后续应用的触发条件化。
- **不影响**：`OpenVLA_Source_Code_Walkthrough.md`、`vla_cache_implementation.md`
  作为代码侧参考保持不变。

---

## 附录 A：核心 finding 的可能形态

写在前面以约束研究方向。最终 paper 的中心句应当形如：

> 在 OpenVLA 7B 上，我们经验性地刻画了 visual token KV 由 "purely visual" 过渡到
> "prompt-conditioned" 的层 `L_lang`，并发现 target 与 background patch 的
> `L_lang` 显著不同（分别为 L_t 与 L_b，效应量 Δ_max）。结合 episode phase 分析，
> 我们进一步发现 KV contextualization 的两条通路（语言驱动 vs 场景驱动）
> [统一 / 解耦]，由此推出 patch-semantic-stratified 的 KV reuse 策略，
> 在 LIBERO-Spatial 上将 false reuse rate 从 X% 降至 Y%，同时保持 success rate。

注意：这个中心句**不出现 motion compensation**。
