# OpenVLA 源码拆解：如何构建一个 VLA 模型

本文基于本仓库中的 `vla-cache/src/openvla` 源码整理。这个目录是 OpenVLA 主代码，同时本地分支加入了 VLA-Cache 推理改动；`vla-cache/src/openvla-oft` 是后续 OFT 路线，本文只在必要处提示，不作为主线展开。

## 1. 一句话理解 OpenVLA

OpenVLA 把机器人策略建模成一个自回归 VLM：

```text
图像 + 语言指令
  -> 视觉编码器产生 image patch features
  -> projector 映射到 LLM hidden dim
  -> 插入到 LLM token embedding 序列中
  -> LLM 逐 token 生成动作 token
  -> ActionTokenizer 把动作 token 反离散化成连续动作
  -> 用数据集统计量反归一化，得到机器人可执行 action
```

也就是说，OpenVLA 没有单独设计一个 policy head。它复用了语言模型的 next-token prediction 机制，把连续动作离散成 LLM vocabulary 尾部的一组 token，让 LLM “说出”动作。

## 2. 源码地图

最重要的目录如下：

```text
vla-cache/src/openvla/
  prismatic/
    conf/
      models.py                  # 基础 VLM 配置：视觉 backbone、LLM、projector、图像尺寸
      vla.py                     # VLA 训练配置：数据混合、冻结策略、batch/lr/FSDP
    models/
      materialize.py             # 从配置实例化 vision backbone、LLM backbone、VLM
      backbones/
        vision/                  # CLIP/SigLIP/DINO/DINO-SigLIP 等视觉编码器
        llm/                     # Llama/Vicuna/Mistral/Phi wrapper 和 prompt builder
      vlms/prismatic.py          # VLM 主体：图像 patch 插入 LLM embedding 序列
      vlas/openvla.py            # OpenVLA wrapper：predict_action + action 反归一化
    vla/
      action_tokenizer.py        # 连续动作 <-> 离散动作 token
      materialize.py             # 构造 VLA dataset、ActionTokenizer、collator
      datasets/
        datasets.py              # RLDS batch -> OpenVLA 训练样本
        rlds/                    # RLDS/OXE 数据读取、归一化、混合采样
    training/
      materialize.py             # 训练策略注册，目前主用 FSDP
      strategies/base_strategy.py# 通用训练 loop + VLA 专用训练 loop
    extern/hf/
      modeling_prismatic.py      # HuggingFace AutoModel 版本，推理/LoRA 微调用
      processing_prismatic.py    # HF Processor：图像处理 + tokenizer
      configuration_prismatic.py # HF config：vision/LLM/name 到具体模型的映射
  vla-scripts/
    train.py                     # FSDP 全量训练/全量微调入口
    finetune.py                  # HF + PEFT LoRA 微调入口
    deploy.py                    # REST API 服务示例
  experiments/robot/
    openvla_utils.py             # 评测加载、VLA-Cache 推理封装
    robot_utils.py               # 环境侧 action 调用和 gripper 后处理
    libero/run_libero_eval.py    # LIBERO benchmark 评测入口
```

## 3. 模型由哪些组件拼起来

### 3.1 Vision Backbone

入口在 `prismatic/models/materialize.py`：

- `VISION_BACKBONES` 注册了 `clip-vit-l`、`siglip-vit-so400m`、`dinov2-vit-l`、`dinosiglip-vit-so-224px` 等。
- `get_vision_backbone_and_transform()` 返回两个东西：视觉模型 wrapper 和对应的 image transform。

基础接口在 `prismatic/models/backbones/vision/base_vision.py`：

- `VisionBackbone` 要提供 `forward()`、`embed_dim`、`num_patches`、`default_image_resolution`。
- `TimmViTBackbone` 用 `timm.create_model(..., pretrained=True, num_classes=0)` 加载 ViT。
- 视觉特征默认取倒数第二层 patch features。
- 支持三种图像处理策略：`resize-naive`、`resize-crop`、`letterbox`。

OpenVLA 7B 主配置使用 DINOv2 + SigLIP 的 fused backbone，对应 `prismatic/models/backbones/vision/dinosiglip_vit.py`：

- 输入图像分别走 DINO 和 SigLIP transform。
- `forward()` 分别得到两个 patch 序列，然后在 feature dim 上拼接。
- `embed_dim = dino_dim + siglip_dim`。
- `num_patches` 保持两路一致，比如 224px、patch size 14 时通常是 16 x 16 = 256 个视觉 patch。

### 3.2 LLM Backbone

入口同样在 `prismatic/models/materialize.py`：

- `LLM_BACKBONES` 注册 Llama-2、Vicuna、Mistral、Phi。
- `get_llm_backbone_and_tokenizer()` 返回 LLM wrapper 和 tokenizer。

核心接口在 `prismatic/models/backbones/llm/base_llm.py`：

- `HFCausalLLMBackbone` 用 HF `AutoConfig` / 具体 `ForCausalLM` 类加载模型。
- 训练时关闭 `use_cache`，推理时打开。
- `embed_input_ids()` 取 LLM 输入 embedding，这一步很关键，因为图像 patch 是直接拼进 embedding 序列，而不是变成文本 token。

Llama/Vicuna 具体实现在 `prismatic/models/backbones/llm/llama2.py`：

- Llama 派生模型都会额外加 `<PAD>` token，并 resize embedding 到 64 的倍数。
- `prompt_builder_fn` 根据模型类型返回 `PurePromptBuilder`、`LLaMa2ChatPromptBuilder` 或 `VicunaV15ChatPromptBuilder`。

OpenVLA 主模型使用 pure Llama-2 风格 prompt。`PurePromptBuilder` 的格式是：

```text
In: What action should the robot take to <instruction>?
Out: <action_tokens></s>
```

### 3.3 Projector

Projector 在 `prismatic/models/vlms/prismatic.py` 初始化：

- `linear`：单层线性映射。
- `gelu-mlp`：`vision_dim -> llm_dim -> llm_dim`。
- `fused-gelu-mlp`：fused vision backbone 使用更宽的 MLP。

实现位于 `prismatic/util/nn_utils.py`。它的职责很单纯：把 vision patch feature 变成 LLM hidden size，使其可以直接拼进 LLM input embedding。

### 3.4 PrismaticVLM：图像 token 如何接入 LLM

核心在 `prismatic/models/vlms/prismatic.py` 的 `PrismaticVLM.forward()`。

关键流程：

```python
patch_features = vision_backbone(pixel_values)
projected_patch_embeddings = projector(patch_features)
input_embeddings = llm_backbone.embed_input_ids(input_ids)

multimodal_embeddings = cat([
    input_embeddings[:, :1, :],        # BOS
    projected_patch_embeddings,        # image patches
    input_embeddings[:, 1:, :],        # rest text tokens
])

llm_backbone(inputs_embeds=multimodal_embeddings, labels=multimodal_labels)
```

注意点：

- 图像 patch 被插在第一个 token 后面，通常就是 BOS 后面。
- labels 也要同步插入一段 `IGNORE_INDEX = -100`，视觉 patch 位置不算语言建模 loss。
- 推理生成时，如果已有 `past_key_values` 且本轮只输入 1 个 token，就直接走 LLM cached forward，不再重复视觉编码。
- 训练和推理共用同一个 forward 结构，所以动作预测本质还是 causal LM loss。

### 3.5 OpenVLA wrapper

`prismatic/models/vlas/openvla.py` 中的 `OpenVLA` 继承 `PrismaticVLM`，只额外增加两件事：

- `norm_stats`：数据集 action 统计量，用于反归一化。
- `action_tokenizer`：把生成的 action token decode 回连续动作。

`predict_action()` 做了完整推理：

```text
PIL image + instruction
  -> prompt builder
  -> tokenizer
  -> image transform
  -> generate max_new_tokens = action_dim
  -> 取最后 action_dim 个 token
  -> ActionTokenizer decode 到 [-1, 1]
  -> 按 dataset_statistics q01/q99 反归一化
```

这里的 `action_dim` 不是写死的，而是从 `norm_stats[unnorm_key]["action"]["q01"]` 的长度推断。

## 4. 动作为什么能变成 token

核心文件是 `prismatic/vla/action_tokenizer.py`。

OpenVLA 默认：

- 每个 action 维度被 clip 到 `[-1, 1]`。
- 用 `bins=256` 做均匀离散。
- 将离散 bin 映射到 tokenizer vocabulary 尾部的 256 个低频 token。
- token id 计算方式近似是 `tokenizer.vocab_size - discretized_bin`。

训练时，一个 7 DoF action 会变成 7 个 token。推理时，LLM 连续生成 7 个 token，再映射回 7 个连续值。

反解逻辑：

```text
predicted_action_token_ids
  -> discretized_actions = tokenizer.vocab_size - token_ids
  -> clip 到合法 bin center index
  -> bin_centers[index] 得到 normalized action
```

然后用数据集统计量反归一化：

```text
action = 0.5 * (normalized_action + 1) * (q99 - q01) + q01
```

如果某个维度在 `mask` 中是 False，比如 gripper 不参与归一化，则保留 normalized value。

## 5. 数据流：RLDS/OXE 如何变成训练 batch

### 5.1 数据集注册

OpenVLA 的默认数据格式是 RLDS/TFDS。关键文件：

- `prismatic/vla/datasets/rlds/oxe/configs.py`：每个数据集的 observation/action schema。
- `prismatic/vla/datasets/rlds/oxe/transforms.py`：把各数据集原始字段标准化。
- `prismatic/vla/datasets/rlds/oxe/mixtures.py`：定义数据混合和采样权重。
- `prismatic/vla/datasets/rlds/oxe/materialize.py`：把 mixture name 转成 dataset kwargs 和 weights。

`configs.py` 中只允许当前代码支持的 action encoding，例如：

- `EEF_POS`：`delta_xyz + rpy + gripper`，共 7 维。
- `EEF_R6`：`delta_xyz + rotation_6d + gripper`，共 10 维。

`make_oxe_dataset_kwargs()` 会设置两个重要 mask：

- `absolute_action_mask`：哪些动作维度是绝对动作，主要影响 action chunk 结束处的 padding。
- `action_normalization_mask`：哪些维度参与归一化。默认 gripper 不归一化。

### 5.2 数据读取和归一化

核心在 `prismatic/vla/datasets/rlds/dataset.py`：

- `make_dataset_from_rlds()` 从 TFDS builder 读取轨迹。
- `standardize_fn` 先把不同数据集字段统一成 `observation`、`task`、`action`。
- `normalize_action_and_proprio()` 把 action 归一化到 `[-1, 1]`，OpenVLA 用的是 `NormalizationType.BOUNDS_Q99`。
- `apply_trajectory_transforms()` 做 trajectory 级处理，比如跳过无语言标签、chunk observation/action。
- `apply_frame_transforms()` 做 frame 级处理，比如 decode image、resize、image augmentation。
- `make_interleaved_dataset()` 按 mixture 权重混合多个数据集。

当前 VLA 训练默认：

```text
window_size = 1
future_action_window_size = 0
load_camera_views = ("primary",)
load_depth = False
load_proprio = False
load_language = True
```

也就是说，vanilla OpenVLA 输入主要是单张第三人称 RGB 图像和语言，不输入 proprio，也不预测 action chunk。

### 5.3 RLDSBatchTransform：一条样本如何变成 LM 样本

`prismatic/vla/datasets/datasets.py` 中的 `RLDSBatchTransform.__call__()` 是最关键的数据转换。

它从 RLDS batch 取：

- `dataset_name`
- `action`
- `observation["image_primary"]`
- `task["language_instruction"]`

然后构造一轮对话：

```python
conversation = [
    {"from": "human", "value": f"What action should the robot take to {lang}?"},
    {"from": "gpt", "value": action_tokenizer(action)},
]
```

再 tokenize，并把 labels 中非动作部分全部设成 `IGNORE_INDEX`：

```python
labels[: -(len(action) + 1)] = IGNORE_INDEX
```

含义是：模型只为动作 token 和可选 stop token 付 loss，不为 prompt、图像 patch、普通文本付 loss。

### 5.4 Collator

`prismatic/util/data_utils.py` 中的 `PaddedCollatorForActionPrediction` 负责：

- padding `input_ids` 和 `labels`。
- 生成 `attention_mask`。
- stack `pixel_values`。
- 透传 `dataset_names`，用于训练时统计每个数据集的 action accuracy。

它明确要求 VLA 样本必须都有图像，和普通 VLM 训练中可能混入 language-only 样本的 collator 不同。

## 6. 训练入口

### 6.1 全量训练/全量微调：`vla-scripts/train.py`

这是 Prismatic 原生训练路线，适合多 GPU FSDP。

主流程：

```text
TrainConfig
  -> 读取 VLAConfig
  -> 初始化 torch.distributed
  -> load base VLM 或 load existing VLA checkpoint
  -> 根据冻结配置确定 stage
  -> freeze_backbones(stage)
  -> get_vla_dataset_and_collator()
  -> save dataset_statistics.json
  -> get_train_strategy(FSDP)
  -> run_setup()
  -> run_vla_training()
```

`prismatic/conf/vla.py` 定义训练配置，典型字段：

- `base_vlm`：基座 VLM，例如 `prism-dinosiglip-224px+7b`。
- `data_mix`：数据混合名，例如 `bridge`、`oxe_magic_soup_plus_minus`。
- `freeze_vision_backbone`、`freeze_llm_backbone`、`unfreeze_last_llm_layer`：决定训练哪些模块。
- `expected_world_size`：预期 GPU 数，代码会强校验。
- `global_batch_size`、`per_device_batch_size`：决定有效 batch。
- `train_strategy`：通常是 `fsdp-full-shard`。

`train.py` 中 stage 的含义：

| stage | 训练内容 |
| --- | --- |
| `vla-full-train` | vision backbone + projector + LLM 全部训练 |
| `vla-train` | 冻结 vision，训练 projector + LLM |
| `vla-sandwich-train` | 训练 vision + projector + LLM 最后一层 |
| `vla-last-layer-train` | 只训练 LLM 最后一层相关模块 |

### 6.2 训练 loop：`run_vla_training()`

实现在 `prismatic/training/strategies/base_strategy.py`。

每个 batch：

1. 前向：`vlm(input_ids, attention_mask, pixel_values, labels)`。
2. `output.loss.backward()`。
3. 从 logits 中取 action token 对齐位置。
4. 计算 action token accuracy。
5. 把预测 token 和 GT token 都 decode 成连续动作，计算 L1 loss。
6. 梯度裁剪、optimizer step、scheduler step。
7. 按 `save_interval` 保存 checkpoint。

一个容易忽略的细节：计算 action logits 时要跳过图像 patch 插入造成的 offset：

```python
action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
action_gt = batch["labels"][:, 1:]
mask = action_gt > action_tokenizer.action_token_begin_idx
```

这里的 `num_patches` 对应插入到 BOS 后面的视觉 token 数。

### 6.3 LoRA 微调：`vla-scripts/finetune.py`

这是更轻量的路线，使用 HuggingFace AutoClasses + PEFT：

```text
AutoProcessor.from_pretrained(vla_path)
AutoModelForVision2Seq.from_pretrained(vla_path)
  -> 可选 4bit quantization
  -> 可选 LoRA target_modules="all-linear"
  -> DDP wrapper
  -> RLDSDataset
  -> ActionTokenizer
  -> PaddedCollatorForActionPrediction
  -> 训练并保存 processor + model
```

适合在已有 OpenVLA checkpoint 上做新任务、新 embodiment 的参数高效微调。

注意：

- LoRA checkpoint 保存后，脚本会尝试把 adapter merge 回 base model，方便推理。
- 微调后必须保存 `dataset_statistics.json`，否则推理时无法按 `unnorm_key` 反归一化。

## 7. HuggingFace 推理版本

`prismatic/extern/hf/` 是为了让 OpenVLA 能通过 HF `AutoProcessor` 和 `AutoModelForVision2Seq` 加载。

主要文件：

- `configuration_prismatic.py`：把 `vision_backbone_id` 映射到 TIMM model id，把 `llm_backbone_id` 映射到 HF LLM。
- `processing_prismatic.py`：实现 `PrismaticImageProcessor` 和 `PrismaticProcessor`。
- `modeling_prismatic.py`：复刻 `PrismaticVLM.forward()`，并定义 `OpenVLAForActionPrediction`。

HF forward 的核心也一样：

```text
pixel_values
  -> PrismaticVisionBackbone
  -> PrismaticProjector
  -> 拼到 input_embeddings 的 BOS 后面
  -> language_model(inputs_embeds=..., labels=...)
```

HF 版本的 `OpenVLAForActionPrediction.predict_action()` 会：

1. 如果 prompt 末尾没有 Llama 的特殊空白 token `29871`，手动补上。
2. 调用 `generate(max_new_tokens=action_dim)`。
3. 取最后 `action_dim` 个 token。
4. 反离散化和反归一化。

本地 `vla-cache/src/openvla` 版本被 VLA-Cache 改过，`predict_action()` 返回的是：

```python
actions, last_caches
```

而不是纯 `actions`。它还依赖 `return_dict_in_generate=True`、`output_attentions=True` 和带 VLA-Cache 改动的 `transformers` fork。`experiments/robot/openvla_utils.py` 是当前本地评测代码的正确调用示例。

## 8. VLA-Cache 本地改动

本仓库不是完全干净的 upstream OpenVLA，它包含 VLA-Cache 相关代码。

相关文件：

- `README_VLA_Cache.md`
- `prismatic/extern/hf/modeling_prismatic.py`
- `experiments/robot/openvla_utils.py`
- `experiments/robot/vla_cache_utils.py`
- `vla_cache_scripts/`

推理流程增加了：

1. 用当前图像和上一帧图像比较，找静态视觉 patch：`find_static_patches()`。
2. 用上一轮 attention 过滤掉任务相关 patch：`task_relevant_selection()`。
3. 把可复用 patch index 写入 `vla.language_model.config.reusable_patches`。
4. 根据上一轮 attention entropy 生成层级 cache reuse schedule：`get_layer_mask_schedule()`。
5. 下次生成时把 `past_key_values` 传回模型。

这部分依赖本地 `pyproject.toml` 中指定的 custom `transformers` fork：

```text
transformers @ git+https://github.com/siyuhsu/transformers.git@vla-cache-openvla
```

如果只想理解或复现 vanilla OpenVLA，可以先忽略 VLA-Cache，按 `OpenVLA` / `PrismaticVLM` / `ActionTokenizer` 主线看。

## 9. 从零构建一个 VLA 的最小路线

### Step 1：选一个 VLM 骨架

最简单是复用已有 Prismatic 配置：

- 基座视觉：`dinosiglip-vit-so-224px`
- 基座 LLM：`llama2-7b-pure`
- projector：`no-align+fused-gelu-mlp`
- VLM config：`prism-dinosiglip-224px+7b`

如果要换 backbone：

- 在 `prismatic/models/materialize.py` 注册新的 vision 或 LLM backbone。
- 在 `prismatic/conf/models.py` 增加新的 `ModelConfig`。
- 如果要支持 HF 导出，还要同步 `prismatic/extern/hf/configuration_prismatic.py` 中的映射。

### Step 2：确定动作空间

OpenVLA 默认适合：

```text
action = [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
```

如果你的机器人不是 7 维，可以做，但要保证：

- RLDS transform 输出统一 action shape。
- `dataset_statistics.json` 中 `action.q01/q99` 长度正确。
- 评测环境的 action 后处理同步修改。
- gripper 这类不连续维度要正确设置 `action_normalization_mask`。

动作维度由统计量推断，所以模型生成 token 数会随 `unnorm_key` 对应的 action 维度变化。

### Step 3：准备数据

推荐转成 RLDS/TFDS，目录放在项目规则要求的数据盘，例如：

```text
/mnt/data0/zjh_data/Embodied_Proj/datasets/<dataset_name>/
```

然后添加：

1. `prismatic/vla/datasets/rlds/oxe/configs.py`：声明 image/state/action schema。
2. `prismatic/vla/datasets/rlds/oxe/transforms.py`：写 standardization transform。
3. `prismatic/vla/datasets/rlds/oxe/mixtures.py`：添加 mixture name。
4. `prismatic/conf/vla.py`：添加 VLAConfig，并注册到 `VLARegistry`。

### Step 4：把动作写进 prompt

训练样本最终长这样：

```text
In: What action should the robot take to put the carrot in the bowl?
Out: <act_1><act_2><act_3><act_4><act_5><act_6><act_7></s>
```

其中 `<act_i>` 不是特殊 token 名字，而是 tokenizer vocabulary 尾部的普通 token id 被复用为 action bin。

### Step 5：训练

全量/FSDP 路线：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "<your-vla-id>" \
  --data_root_dir /mnt/data0/zjh_data/Embodied_Proj/datasets/<rlds_root> \
  --run_root_dir /mnt/data0/zjh_data/Embodied_Proj/checkpoints/<experiment> \
  --wandb_project "<project>" \
  --wandb_entity "<entity>"
```

LoRA 路线：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /mnt/data0/zjh_data/Embodied_Proj/datasets/<rlds_root> \
  --dataset_name "<dataset_or_mixture_name>" \
  --run_root_dir /mnt/data0/zjh_data/Embodied_Proj/checkpoints/<experiment> \
  --adapter_tmp_dir /mnt/data0/zjh_data/Embodied_Proj/checkpoints/<experiment>/adapter-tmp \
  --lora_rank 32 \
  --batch_size 16 \
  --learning_rate 5e-4 \
  --image_aug True
```

如果需要 GPU，根据仓库规则应在沙箱外运行。

### Step 6：推理部署

标准 HF 推理思路：

```python
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="<dataset_name>", do_sample=False)
```

本地 VLA-Cache 版本请参考 `experiments/robot/openvla_utils.py`，因为这里 `predict_action()` 会额外返回 cache。

## 10. 最容易踩坑的地方

1. Prompt 必须和训练一致。OpenVLA 主版本是 `In: ...\nOut:`，v0.1/Vicuna 是 `USER: ... ASSISTANT:`。
2. 图像 patch 是插在 BOS 后面，不是文本里显式的 `<image>` token。
3. 训练 loss 只看 action token。`labels[: -(len(action) + 1)] = -100` 是 VLA 训练的关键。
4. action token 使用 vocabulary 尾部 token。换 tokenizer 时要确认“低频 token 在 vocab 尾部”这个假设是否成立。
5. `dataset_statistics.json` 是推理必需品。缺它就无法按 `unnorm_key` 反归一化。
6. gripper 维度通常不做 q01/q99 归一化，但评测环境可能需要再把 `[0,1]` 转成 `[-1,1]` 并翻转符号。
7. vanilla OpenVLA 不做 action chunk，控制频率建议低一些。README 中建议数据采集频率约 5-10 Hz。
8. 本地 VLA-Cache 修改过 HF 推理返回值。`deploy.py` 保留了 upstream 风格，若直接用本地 HF model 可能需要按 `openvla_utils.py` 的调用方式调整。
9. 全量训练的 `expected_world_size` 会强校验 GPU 数，不匹配会直接 assert。
10. TensorFlow RLDS 读取代码会禁用 TF GPU，避免和 PyTorch 抢 GPU，这是正常行为。

## 11. 建议阅读顺序

按下面顺序读源码，负担最小：

1. `prismatic/vla/action_tokenizer.py`
2. `prismatic/vla/datasets/datasets.py`
3. `prismatic/models/vlms/prismatic.py`
4. `prismatic/models/vlas/openvla.py`
5. `prismatic/conf/models.py`
6. `prismatic/conf/vla.py`
7. `vla-scripts/train.py`
8. `prismatic/training/strategies/base_strategy.py`
9. `vla-scripts/finetune.py`
10. `prismatic/extern/hf/modeling_prismatic.py`
11. `experiments/robot/openvla_utils.py`

读完前 4 个文件，基本就能明白 OpenVLA 的核心思想：VLA 不是给 VLM 外接一个传统 policy head，而是把动作离散化成语言模型可以生成的 token，再用普通 causal LM loss 训练。

## 12. 最小心智模型

可以把 OpenVLA 记成下面这个公式：

```text
policy(image, instruction)
  = denormalize(
      detokenize_action(
        LLM.generate(
          [BOS, Projector(VisionEncoder(image)), TextTokens(prompt)]
        )
      )
    )
```

训练时：

```text
loss = CrossEntropy(
  logits_on_action_token_positions,
  ground_truth_action_token_ids
)
```

所以构建一个 VLA 的本质工作是：

- 选好视觉编码器和 LLM。
- 让 projector 对齐视觉 patch 和 LLM hidden space。
- 设计稳定的 action tokenization 和 normalization。
- 把机器人数据统一成 image + language + action。
- 严格保持训练和推理的 prompt、图像预处理、action 后处理一致。
