# OpenVLA KV Contextualization Layer-wise Study —— 实验设计文档

Date: 2026-06-03

本设计文档承接 `Docs/MC_claude_log.md#389-449` 的收敛结论：把因子从 6 个砍到
2 个，只保留 **prompt × patch_semantics** 和 **episode_phase × patch_semantics**
两组主因子，各自产出一张 layer-wise sim 热图，独立刻画“语言”和“阶段”两条
contextualization 通路。

motion compensation 在本研究里**只作为工具**：用 oracle 几何对应把“两帧 patch
在像素上对应同一世界点”这件事确定下来，从而把几何估计误差从实验中剔除，单独
隔离出 contextualization 对 KV 可复用性的影响。

---

## 1. 研究问题

一句话：**同一世界点的 visual token KV，在第几层之前还可以被认为只是它自身的
函数，而不依赖整张图的其它部分与 language prompt？**

拆成两个可测子问题：

| 编号 | 子问题 | 用哪个网格回答 |
|---|---|---|
| RQ1 | language prompt 对 visual KV 的塑形从第几层开始？是否只塑形 target、不塑形 background？ | 网格 A: prompt × patch |
| RQ2 | episode 阶段切换（夹爪进画面 / 物体被遮挡 / 物体在手）对 visual KV 的塑形从第几层开始？ | 网格 B: phase × patch |

最终 finding 不是“L\* = 某个数字”，而是：

```text
VLA 中 vision KV 的可复用层范围是 prompt-conditioned 且 patch-semantic-stratified 的；
语言通路与视觉/阶段通路的“分叉层”是否一致，决定 contextualization 是统一的还是解耦的。
```

---

## 2. 核心假设

| ID | 假设 | 判定方式 |
|---|---|---|
| H1 | background patch 的 sim(L) 曲线对 prompt 几乎不敏感（P0≈P3 全层重合） | 网格 A，background 列 |
| H2 | target patch 存在一个 prompt 分叉层 L_lang，L_lang 之前视觉主导、之后被 prompt 重塑 | 网格 A，target 列分叉点 |
| H3 | target patch 的 sim(L) 随 episode_phase 在某层 L_phase 之后显著分化，background 不分化 | 网格 B，target vs background |
| H4 | L_lang 与 L_phase 接近 → contextualization 统一；显著不同 → 语言/视觉通路解耦 | 网格 A、B 分叉层对比 |

无论结果偏哪一侧，都是可发表的定量刻画——这正是 motion-compensated KV reuse
的物理上限来源。

---

## 3. 测量本体（与因子无关的共享协议）

### 3.1 模型与 token 约定

```text
Model      : openvla/openvla-7b-finetuned-libero-spatial
Backbone   : Llama-2 7B, 32 个 decoder layer (L = 0..31)
Image      : 224 x 224, patch 14 -> 16 x 16 = 256 visual tokens
Token index: token_idx = patch_id + 1   (patch_id ∈ 0..255)
```

### 3.2 两次“干净” teacher forward

对一对相邻控制步 (t-1, t)，各跑一次**完整 forward，不用 cache、不做任何
remap、不做 token reuse**：

```text
run_A = vla.predict_action(image_{t-1}, prompt, fresh DynamicCache,
                           output_attentions=True, output_hidden_states=True)
run_B = vla.predict_action(image_t,     prompt, fresh DynamicCache, ...)
```

从返回的 `last_caches['past_key_values']`（`DynamicCache`）抽取每层每个 visual
token 的 K、V：

```text
K[L][token_idx] = past_key_values.key_cache[L][:, :, token_idx, :]   # [num_heads, head_dim]
V[L][token_idx] = past_key_values.value_cache[L][:, :, token_idx, :]
```

> RoPE 说明：`key_cache` 里的 K 已经施加了 RoPE，含位置编码。位置是混杂变量。
> 因此本研究**同时测两条**：
> - `KV-cached`：直接用 cache 里的 post-RoPE K、原始 V（部署时真正会被复用的对象）。
> - `H-hidden`：用 `output_hidden_states` 的逐层隐状态 h_L（投影前、无 RoPE），
>   彻底剔除位置编码项，只看 contextualization 本身。
> 两条曲线的差就是“RoPE correction 能挽回多少”，顺带回答原 plan 里
> “要不要做 RoPE correction”的工程问题。

### 3.3 sim 定义（K / V 分开）

对每个 oracle 对应 patch pair (i, j) ∈ P 和每层 L：

```text
sim_K(L; i,j) = cos( K_prev[L][i].flatten(),  K_curr[L][j].flatten() )
sim_V(L; i,j) = cos( V_prev[L][i].flatten(),  V_curr[L][j].flatten() )
sim_H(L; i,j) = cos( h_prev[L][i],            h_curr[L][j] )           # 无 RoPE 对照
```

K 与 V **必须分开报**：K 错 → softmax(QKᵀ) 注意力分布偏；V 错 → 分布对但取回内容错。

### 3.4 上下界对照（每个 cell 都要带）

| 对照 | 构造 | 含义 |
|---|---|---|
| 上界 ceiling | 同一帧重跑两次（数值噪声/dropout-off 抖动） | sim 的天花板，≈1，验证 pipeline 无 bug |
| 下界 floor | 同一帧内随机 patch 对 | sim 的地板，无对应时的基线 |
| 自反 sanity（网格 0） | image_{t-1}=image_t，oracle 对应=identity | 必须全层贴近 1，否则先修 pipeline |

读图法：测量曲线从 ceiling 下滑到接近 floor 的那个 L，就是该 cell 的“失效层”。

---

## 4. oracle 几何对应（把 motion 变量剔除）

LIBERO 仿真可取 depth + camera pose，用 3D backproject→reproject 建立严格世界
对应，**不使用** RGB global translation（那是估计器，会引入误差）。

```text
对 image_t 的每个 patch j:
  取 patch 中心像素 + robot0_eye_in_hand_depth -> 反投影到世界点 X
  用 prev 帧相机 pose + intrinsics 把 X 投回 image_{t-1}
  落入哪个 patch 即 source patch i
  用 z-buffer / depth 一致性过滤遮挡和新显露
  -> 高质量 pair 集合 P = {(i, j)}
```

环境侧需要在 `get_libero_env` 打开（当前只传了 `camera_names`）：

```python
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": resolution, "camera_widths": resolution,
    "camera_names":         [camera_name],
    "camera_depths":        True,          # 新增：取 depth
    "camera_segmentations": "instance",    # 新增：取 instance/seg mask, 用于 patch_semantics
}
```

> 说明：本研究主体在 `agentview` 与 `robot0_eye_in_hand` 都可跑。agentview 相机近
> 静止，自反/连续帧对照充分；wrist camera 提供更强的几何扰动，用于把曲线“拉开”。
> 两个相机各跑一份，作为 motion magnitude 的弱/强两档（非主因子，仅做稳健性附录）。

---

## 5. 因子操作化（写死，不靠感觉）

### 5.1 patch_semantics（两类对照，其余显式排除）

用 instance segmentation mask 在 patch 网格上判定：

```text
target     : language prompt 提到的物体，其 seg 像素 >= 50% 占该 14x14 patch
background : 桌面/墙/不在 prompt 中的物体，且不与 gripper 接触
排除       : gripper / distractor / 物体边界 patch -> 不进入任一组
```

prompt→target 物体的映射用 LIBERO 每个 task 的 bddl 物体名 + 人工 mapping 表
固化（每个 task family 一张表），避免 parsing 歧义。

### 5.2 episode_phase（用客观信号切，不做时间四等分）

用 `obs["robot0_gripper_qpos"]`（夹爪开合）、`obs["robot0_eef_pos"]`（末端位置）、
以及 target 物体世界位姿（仿真可取）定义：

```text
reach     : gripper open 且 eef 到 target 距离 > d_thresh
grasp     : gripper 正在闭合的 ±5 帧 (qpos 由开到合的过渡窗)
transport : object-in-hand 且尚未进入 place 区域
place     : object-in-hand 且接近目标放置位，或 gripper 重新打开前后窗
```

阈值 `d_thresh`、过渡窗大小写进配置并随报告公布；切相位用客观信号，后续方差
分解才不被切分误差污染。

### 5.3 prompt 条件（本研究只用两档，收敛自 P0–P3）

| 条件 | prompt | 作用 |
|---|---|---|
| P0 | 原任务 prompt | baseline，完整 contextualization |
| P3 | 其它 task family 的无关 prompt | 完全无关时的 ceiling-of-language-effect |

> P1（空 prompt）、P2（同族错误目标）作为可选扩展放附录；主网格只跑 P0/P3，
> 因为 P0–P3 的差就是 prompt 效应的最大幅度，足以定位 L_lang。

---

## 6. 两个实验网格

两网格**共享同一套 patch pair 和同一套测量协议**，只是分组方式不同。

### 网格 A：prompt × patch_semantics（回答 RQ1）

```text
                 background     target
P0 (real prompt)  曲线 A00       曲线 A01
P3 (wrong prompt) 曲线 A10       曲线 A11
```

读法：
- target 列 A01 vs A11 开始分叉的层 = **L_lang(target)**。
- background 列 A00 vs A10 应近似全层重合 = 验证“背景不被 prompt 重塑”。

### 网格 B：episode_phase × patch_semantics（回答 RQ2）

```text
                 background     target
reach             曲线 B00       曲线 B01
grasp             曲线 B10       曲线 B11
transport         曲线 B20       曲线 B21
place             曲线 B30       曲线 B31
```

读法：
- target 列四条曲线开始分化的层 = **L_phase(target)**（阶段切换最影响哪些层）。
- background 列四条应近似重合（对照）。

每条曲线都各带 K / V / H 三个子图，以及 ceiling / floor 阴影带。

---

## 7. 采样规模（只跑 forward，无需 rollout 成功）

```text
task family : libero_spatial / object / goal / 10  (4 个)
每 family   : 10 task x 5 episode
每 episode  : 每 10 step 取一个 frame pair  -> ~15 pair-points / episode
每 frame pair: oracle 对应 + seg 过滤后 ~100 patch pair
合计        : 4 x 10 x 5 x 15 x 100 ≈ 3 x 10^5 patch pair
```

episode 用现成 OpenVLA checkpoint 做开环滚动采帧即可（成功与否不影响本研究，
因为我们只要帧序列 + depth + seg + state，不要任务成功）。

---

## 8. 统计与报告（拒绝“对所有 pair 求 mean”）

```text
聚合单位 : episode 级（pair 在 episode 内不独立，共享 context）
置信区间 : bootstrap over episode（不是 over pair），报 95% CI
方差分解 : per-layer sim 方差用线性混合模型 / ANOVA 分解
           Var(task) + Var(episode|task) + Var(pair|episode)
效应量   : 报 sim_P0 - sim_P3 每层差值 + CI（而非 p 值）
```

方差分解结论指向：

| 主导项 | 解释 | 对方法的含义 |
|---|---|---|
| Var(task) 主导 | 单一 L\* 不存在 | 必须按 task family 分别报 schedule |
| Var(episode\|task) 主导 | episode 漂移大 | paper 须承认不稳定 |
| Var(pair\|episode) 主导 | patch_semantics 是真解释变量 | §5.1 分层站得住，core finding 成立 |

“分叉层”定义（可操作）：取 |sim_P0(L) − sim_P3(L)| 首次超过 ceiling–floor 带宽
的 ε 比例（如 10%）且后续保持的最小 L，对每 episode 各求一次，再 bootstrap 报
L_lang / L_phase 的分布与 CI。

---

## 9. 行为级验证（cosine 是代理，需挂钩动作）

cosine 高不等于复用后动作正确。加一组因果干预，作为方法学内部一致性检验：

```text
对帧 t:
  baseline   : 全层 fresh recompute -> action_full
  treatment_L: 仅在 layer L 把 visual token 的 K,V 替换为 K_prev[i],V_prev[i] (oracle 对应)
  对每层 L 测:  || action_L - action_full ||  与 top-1 action-token agreement
```

期望这条 action-sensitivity(L) 曲线与 §6 的 sim(L) 曲线高度相关；若不相关，说明
cosine 指标不合适，需更换（这是对整套方法学的 sanity）。

实现上对应 `openvla_utils` 已有的 `remap_visual_kv_cache(...)` 分支，按单层注入。

---

## 10. 工程落点（最小改动）

| 改动 | 文件 | 内容 |
|---|---|---|
| 取 depth + seg | `experiments/robot/libero/libero_utils.py` | `get_libero_env` 加 `camera_depths`/`camera_segmentations` |
| 采帧并 dump | `experiments/robot/libero/run_libero_eval.py` | 新增 `--study_dump_dir`，每 N step 存 (rgb_t, rgb_{t-1}, depth, seg, eef, gripper, cam_pose, task_id, prompt) |
| oracle 对应 | 新增 `experiments/robot/libero/oracle_correspondence.py` | depth backproject + reproject + z-buffer -> P={(i,j)} |
| 双 forward + KV 抽取 | 新增 `experiments/study/kv_layerwise.py` | 两次 fresh forward，抽 `key_cache/value_cache/hidden_states`，算 sim_K/V/H |
| 因子标注 | 新增 `experiments/study/factor_labeling.py` | patch_semantics(seg) + episode_phase(gripper/eef) 标注 |
| 统计+画图 | 新增 `experiments/study/analyze_sim.py` | episode bootstrap、方差分解、两网格热图/曲线 |

复用现有：`predict_action(..., output_attentions=True, past_key_values=DynamicCache())`
已经能拿到逐层 cache（见 `openvla_utils.py:495`），只需加 `output_hidden_states=True`
并保存 visual token 段。

输出与权重遵守项目规则放 `/mnt/data0/zjh_data`，repo 内只放轻量结果：

```text
/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_study_frames/   # dump 的帧/depth/seg/KV
vla-cache/Experiments/openvla_libero_mc_cache/outputs/kv_study/    # 曲线/热图/CSV/报告
```

---

## 11. 执行顺序（先验管线，再跑网格）

```text
S0  自反 sanity（网格 0）：image_t==image_{t-1}，全层 sim 必须≈1 —— 不过不许往下
S1  上下界对照：ceiling(同帧重跑) / floor(随机 pair) 标定
S2  oracle 对应可视化：抽几对 (i,j) 画连线，肉眼确认 depth/pose/翻转/坐标系正确
S3  网格 A（prompt × patch），先 1 个 task family 跑通，再扩到 4 个
S4  网格 B（phase × patch），共用 S3 的 pair 池
S5  方差分解 + bootstrap CI，定位 L_lang / L_phase
S6  行为级 KV substitution 验证，校准 sim 与 action sensitivity
```

S0–S2 任何一步不过，先修 pipeline，否则后面所有曲线不可信（呼应
`MC_claude_log.md#441-444`）。

---

## 12. 通过标准 / 产出

通过标准：

```text
- S0 自反曲线全层 >= 0.99；ceiling/floor 分离明显
- 网格 A: background 列 P0≈P3（带内重合），target 列存在可定位的 L_lang
- 网格 B: target 列随 phase 分化、background 列不分化
- 方差分解能明确指认主导项；L_lang / L_phase 带 episode-bootstrap CI
- 行为级 substitution 曲线与 sim 曲线相关（Spearman 显著）
```

最终产出（论文素材）：

```text
Fig.1  网格 A 热图/曲线 + L_lang(target/background)
Fig.2  网格 B 热图/曲线 + L_phase(target/background)
Fig.3  K vs V vs H(无RoPE) 三条曲线对比（RoPE correction 能挽回多少）
Fig.4  action-sensitivity(L) 与 sim(L) 校准图
Tab.1  方差分解 + L_lang/L_phase ± CI（跨 4 个 task family）
```

由这些图直接推出 **prompt-aware, semantics-stratified layer schedule**，作为
motion-compensated cache 真正区别于 VLN-Cache 的方法贡献；motion compensation
退居工具层。

---

## 13. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| depth/pose 坐标系或图像翻转错误 | oracle 对应错，全study失效 | S2 强制可视化先行 |
| seg 把 distractor 混入 background / target 边界糊 | 曲线被污染 | §5.1 阈值写死 + 边界 patch 显式排除 |
| 相位切分误差 | 网格 B 分化被切分噪声盖住 | §5.2 用 gripper/eef 客观信号，非时间四等分 |
| cache 里 K 含 RoPE，位置混杂 | sim 下滑混入位置项 | 同时报 H(无RoPE)对照，分离 contextualization |
| agentview 相机近静止，曲线拉不开 | 看不到分叉 | 用 wrist camera 提供强几何扰动档 |
| episode 间漂移大 | 单一结论不稳 | episode-level bootstrap + 方差分解如实报告 |

---

## 14. 一句话

固定 motion compensation 用 oracle、固定其它变量，只把 **(prompt, patch_semantics)**
和 **(episode_phase, patch_semantics)** 当两组主因子，各产一张 layer-wise sim 热图，
独立刻画“语言”和“阶段”两条 contextualization 通路，并用行为级 KV substitution
校准——这就是支撑 “VLA vision KV 可复用层范围是 prompt-conditioned 且
patch-semantic-stratified” 这一 finding 的最小可执行 study。
