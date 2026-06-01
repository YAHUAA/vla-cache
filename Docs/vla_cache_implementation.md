# VLA-Cache 实现原理分析

## 整体架构

VLA-Cache 是一种**自适应 Token 缓存机制**，用于加速 Vision-Language-Action (VLA) 模型在机器人实时控制中的推理。核心思想是：在连续的控制帧之间，大部分视觉 tokens 对应的场景内容不变，其 key-value cache 可以直接复用，只需重新计算变化区域和任务关键区域的 tokens。

项目地址: [siyuhsu/vla-cache](https://github.com/siyuhsu/vla-cache)

---

## 1. 核心文件

| 文件 | 职责 |
|------|------|
| `src/openvla/experiments/robot/vla_cache_utils.py` | 静态 patch 检测、注意力驱动的任务相关性选择、层级 mask 调度 |
| `src/openvla/experiments/robot/openvla_utils.py` | VLA-Cache 推理管线入口 (`get_vla_action`) |
| `src/openvla/prismatic/extern/hf/modeling_prismatic.py` | HF 兼容模型层，KV-Cache 管理与裁剪 (`predict_action`) |
| transformers fork (`modeling_llama.py`) | **魔改的 LlamaModel.forward()**：Token 级剪枝与 FLOPs 追踪 |

---

## 2. 推理管线 (per-timestep)

每一帧的完整推理流程如下：

```
┌─────────────────────────────────────────────────────┐
│  Step 1: 静态 Patch 检测 (find_static_patches)       │
│  比较当前帧与上一帧的 patch 相似度                    │
│  选出 top-k 个余弦相似度 ≥ 0.996 的静态 patch        │
├─────────────────────────────────────────────────────┤
│  Step 2: 任务相关性选择 (task_relevant_selection)     │
│  利用上一帧的 attention map 找出 task-relevant tokens │
│  与静态 tokens 取差集 → remaining_static_tokens      │
│  差集中的 token 是"静态但不关键"的，可以被跳过        │
├─────────────────────────────────────────────────────┤
│  Step 3: 层级 Mask 调度 (get_layer_mask_schedule)     │
│  基于上一帧各层的 attention 熵                      │
│  为每层计算一个 reuse proportion (0~1)              │
│  熵越低 → 注意力越集中 → reuse proportion 越高      │
├─────────────────────────────────────────────────────┤
│  Step 4: 模型推理 (forward + predict_action)          │
│  LlamaModel: 在指定层对 non-essential tokens 剪枝    │
│  predict_action: 生成 action + 裁剪 KV-Cache          │
└─────────────────────────────────────────────────────┘
```

### Step 1: 静态 Patch 检测

```python
# vla_cache_utils.py: find_static_patches()
def find_static_patches(img_0, img_1, patch_size=14, top_k=150, sim_threshold=0.996):
    patches1 = patchify(img_0, patch_size)   # 224x224 → 16x16 grid, each 14x14
    patches2 = patchify(img_1, patch_size)
    similarity = calculate_patch_similarity(patches1, patches2)  # cosine similarity per patch
    # 筛选相似度 ≥ 0.996 的 patch，按相似度降序取 top_k
```

224x224 的图像被切分成 16x16 = 256 个 14x14 的 patch（与 ViT 的 patch size 对齐）。对每个 patch 计算两帧间的余弦相似度，相似度 ≥ 0.996 的 patch 被认为是**静态的**。

### Step 2: 任务相关性选择

```python
# vla_cache_utils.py: task_relevant_selection()
def task_relevant_selection(multihead_attention, image, significant_patches, top_k=120):
    attn_score = token_attention_merge(multihead_attention, layer_id=15)  # 取第15层的text→vision attention
    top_patches = get_top_attention_patches(attn_score, top_k)
    # 差集: 静态但低注意力的 patches = 可以被跳过的 tokens
    only_significant = set(significant_patches) - set(top_patches)
```

这里的关键逻辑：
- **attention map** 来自上一帧第 15 层（深层的语义注意力），反映 text tokens 到 vision tokens 的注意力分布
- **任务相关的 vision tokens** = attention 分数最高的 top-k（默认 120）个视觉 patch
- **可跳过的 tokens** = 静态 patch - 任务相关 patch = "静态但不重要"的 visual tokens

### Step 3: 层级 Mask 调度

```python
# vla_cache_utils.py: get_layer_mask_schedule()
def get_layer_mask_schedule(multihead_attention, apply_weighted_growth=True, growth_factor=0.55):
    # 计算每层 attention 的归一化熵
    # 熵越低 → reuse proportion 越高 → 剪枝越激进
    for layer_attn in multihead_attention[:-1]:
        token_entropy = -Σ(attn * log(attn))     # token-level entropy
        entropies.append(token_entropy.mean())    # 层平均熵
    
    norm_entropy = (entropies - min) / (max - min)
    reuse = 1.0 - norm_entropy                   # 低熵 = 高复用率
```

- 每层的 reuse proportion 不同，由上一帧该层的 attention entropy 决定
- `apply_weighted_growth=True` 使上层不能比下层 reuse 更多的 token（delta > 0 时乘以 0.55 平滑增长）
- 结果是：**浅层 reuse 少（多算），深层 reuse 多（少算）**

### Step 4: LlamaModel 中的 Token 剪枝

这是核心的 inference-time 优化，发生在 transformers fork 的 `modeling_llama.py` 中：

```python
# modeling_llama.py: LlamaModel.forward()
# 剪枝仅发生在特定层: self.pruning_loc = [2, 6, 9, 11]
# 仅当 inputs_embeds.shape[1] != 1 时（非 cached generation step）

for layer_idx, decoder_layer in enumerate(self.layers):
    if self.config.proportion_attn_var is not None 
       and inputs_embeds.size(1) != 1 
       and layer_idx in self.pruning_loc:
        
        reusable_patches = self.config.reusable_patches      # Step 2 中的 only_significant
        proportion = self.config.proportion_attn_var[layer_idx]  # Step 3 中的 reuse
        top_k = max(1, int(proportion * len(reusable_patches)))
        selected_reusable_patches = reusable_patches[:top_k]
        
        # 从 hidden_states, causal_mask, position_ids 中移除这 top_k 个 token
        mask = ~torch.isin(cache_position, selected_reusable_patches)
        hidden_states = hidden_states[..., mask, :]
        causal_mask = causal_mask[..., mask, :]
        ...
```

**关键设计**：

1. **剪枝层选择** `[2, 6, 9, 11]`：不在最浅层（需要全局上下文）和 token 生成单步时剪枝
2. **per-layer proportion**：不同层剪枝不同数量的 token，深层剪枝更多
3. **KV-Cache 保留**：被剪枝的 token 不参与当前层的计算，但其上一帧的 KV-Cache 仍然存在于 `past_key_values` 中，在 attention 计算时通过 cache 影响其他 token

---

## 3. KV-Cache 管理与 Action 生成

### predict_action 中的 Cache 裁剪

```python
# modeling_prismatic.py: OpenVLAForActionPrediction.predict_action()
def predict_action(self, input_ids, unnorm_key, **kwargs):
    results = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)
    
    # 关键：生成 action 后，裁剪掉 action tokens 的 KV-Cache
    max_cache_length = past_key_values._seen_tokens - self.get_action_dim(unnorm_key) + 1
    past_key_values.crop(max_length=max_cache_length)
    
    # 返回裁剪后的 cache + attention map，供下一帧使用
    last_caches = {"past_key_values": past_key_values, "attentions": attentions[0]}
    return actions, last_caches
```

- 生成 action 后，`past_key_values` 包含场景 tokens + 7 个 action tokens
- `crop()` 裁掉 action tokens 的最后 6 个（保留第 1 个 action token 作为下一帧生成的 context），避免 action tokens 污染下一帧的场景理解

### 跨帧 Cache 传递

```python
# openvla_utils.py: get_vla_action() (简化)
def get_vla_action(cfg, vla, ..., last_caches=None):
    # 上一帧的 cache 作为本帧的初始 past_key_values
    prompt_cache = last_caches['past_key_values'] if last_caches else DynamicCache()
    prev_attn = last_caches['attentions'] if last_caches else None
    
    if cfg.use_vla_cache and prompt_cache is not None:
        # VLA-Cache 三步骤 (见上文)
        stable_patches = find_static_patches(image, prev_image)
        result_image, remaining = task_relevant_selection(prev_attn, image, stable_patches)
        vla.language_model.config.reusable_patches = mask_indices
        vla.language_model.config.proportion_attn_var = get_layer_mask_schedule(prev_attn)
    
    # 用复用后的 cache 推理
    action, last_caches = vla.predict_action(..., past_key_values=prompt_cache)
    return action, last_caches, result_image
```

---

## 4. 加速原理总结

```
                    无 Cache (每帧从头计算)        VLA-Cache (复用 KV + 剪枝)
                    ┌─────────────────────┐      ┌─────────────────────┐
Vision Tokens (256) │ ████████████████████ │      │ 剪枝掉 ~130 个 tokens │
                    │ 全部通过 32 层 LLM   │  →   │ 仅 ~126 个通过深层    │
Text Tokens  (~15)  │ ████████████████████ │      │ ████████████████████ │
                    └─────────────────────┘      └─────────────────────┘
FLOPs  (per step)   1.91 TFLOPs                  1.49 TFLOPs (-22%)
Latency (per step)  ~109 ms                      ~83 ms   (-24%)
Latency (first step) ~210 ms                     ~222 ms  (+5.7%, cache init overhead)
```

三个层次的优化：

| 层次 | 机制 | 效果 |
|------|------|------|
| **KV-Cache 复用** | 上一帧的场景理解通过 `past_key_values` 保留，本帧无需重新编码静态区域 | 避免冗余计算 |
| **静态 Token 剪枝** | 视觉上无变化的 patch tokens 在特定层被跳过前向计算 | 减少 attention 和 FFN 的 token 数量 |
| **层级自适应比例** | 基于 attention 熵为每层设置不同的剪枝比例，深层剪枝更多 | 在保持任务精度前提下最大程度减少计算 |

速度提升来自：attention 计算复杂度 O(n²) 和 FFN 复杂度 O(n) 都随 token 数减少而降低。每帧节省 ~24% 的推理时间。

---

## 5. 关键代码路径

```
run_libero_eval.py
  └─ eval_libero()
       └─ get_model() → robot_utils.py/get_model() → openvla_utils.py/get_vla()
       └─ for each episode:
            └─ get_action() → robot_utils.py/get_action() → openvla_utils.py/get_vla_action()
                 │
                 ├─ [VLA-Cache ON] find_static_patches() / task_relevant_selection()
                 │     └─ get_layer_mask_schedule()
                 │
                 ├─ vla.predict_action() → modeling_prismatic.py
                 │     └─ self.generate() → transformers generation
                 │           └─ LlamaModel.forward() → modeling_llama.py (魔改)
                 │                 └─ [pruning_loc layers] token-level pruning
                 │
                 └─ past_key_values.crop() → 裁剪 action tokens → 返回 last_caches
```
