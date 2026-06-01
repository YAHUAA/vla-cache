适合本方法的数据集，核心不是“有没有视频”，而是有没有 **移动相机视角**，尤其是：

- wrist / eye-in-hand camera：随机械臂末端运动
- head / body camera：随移动底盘或头部运动
- egocentric wearable camera：人戴在头/胸/眼镜上，适合预训练但不一定有机器人 action

**最推荐**
| 数据集 | 摄像头是否移动 | 为什么适合 |
|---|---|---|
| **DROID** | 强，wrist-mounted Zed Mini 随手腕运动；还有外部 stereo cameras | 真实机器人、大规模、操作场景多，最适合验证 camera ego-motion 下的 VLA-cache；官方页写有 76k demos/350h，并包含 wrist camera setup：[DROID](https://droid-dataset.github.io/) |
| **RH20T** | 强，1-2 个 in-hand cameras + 8-10 个 global RGB-D cameras | 有多视角、RGB-D、力觉、动作和校准信息，适合做 3D motion compensation 上限实验：[RH20T](https://rh20t.github.io/) |
| **RLBench / Colosseum** | 强，包含 eye-in-hand camera；仿真可拿 depth/seg/pose | 最适合先做 oracle 3D 对齐，因为相机 pose、depth、seg 都容易拿到：[RLBench](https://arxiv.org/abs/1909.12271), [Colosseum](https://huggingface.co/datasets/colosseum/colosseum-challenge/blob/main/README.md) |
| **CALVIN** | 中到强，有 static camera + gripper camera RGB-D | gripper camera 是移动相机，长程语言任务也适合测 cache staleness：[CALVIN dataset](https://deepwiki.com/mees/calvin/2-dataset) |
| **ManiSkill** | 强，可用 wrist camera / egocentric RGB-D / moving camera | 仿真可控，适合系统性扫相机运动强度：[ManiSkill wristcam](https://maniskill.readthedocs.io/en/v3.0.0b21/robots/panda_wristcam/), [ManiSkill paper](https://arxiv.org/abs/2107.14483) |

**也很适合，但要筛子集**
| 数据集 | 适用性 |
|---|---|
| **Open X-Embodiment / RT-X** | 聚合了很多真实机器人数据，部分子集有 wrist camera / mobile robot / head camera。要按 observation keys 过滤，不是所有子集都适合：[Open X-Embodiment](https://robotics.growbotics.ai/projects/datasets/open-x-embodiment-oxe) |
| **RT-1 / RT-2** | Google mobile manipulator，head-mounted camera + mobile base，天然有相机自运动；但直接可用性取决于公开子集和格式：[RT-1 data summary](https://claru.ai/models/rt-1) |
| **BridgeData V2** | 主视角多为 fixed third-person，但部分 episode 有 wrist camera；只建议用 wrist-camera subset：[BridgeData V2](https://arxiv.org/abs/2308.12952) |
| **RoboSet** | 有 `image_wrist`，适合做真实 wrist camera 测试：[TFDS RoboSet](https://www.tensorflow.org/datasets/catalog/robo_set) |
| **ALOHA / Mobile ALOHA** | wrist cameras 很典型，Mobile ALOHA 还有移动底盘，适合验证移动操作；但语言/VLA格式可能要自己整理：[Mobile ALOHA](https://mobile-aloha.github.io/), [ALOHA camera docs](https://docs.trossenrobotics.com/aloha_docs/1.0/operation/data_collection.html) |
| **AgiBot World / humanoid datasets** | head/hand 多相机 + 移动双臂/人形机器人，很适合这个方向，但要确认开放版本里 camera calibration、pose、action 格式：[AgiBot World](https://arxiv.org/abs/2503.06669) |

**不太适合直接验证，但可做预训练/辅助**
| 数据集 | 原因 |
|---|---|
| **Ego4D / Ego-Exo4D** | 摄像头强移动，但主要是人类 egocentric 视频，没有机器人 action；适合训练/评估视觉对齐、flow、world-static token 检测，不适合直接做 VLA action success：[Ego4D](https://ego4d-data.org/docs/), [Ego-Exo4D](https://docs.ego-exo4d-data.org/) |

**对我们当前 repo 的判断**
现在跑的 `OpenVLA + LIBERO-Spatial` 默认用 `agentview_image`，基本是固定相机，所以不是最能体现你这个方法优势的设置。LIBERO/robomimic 其实可以导出 `robot0_eye_in_hand` 视角，文档里也有 `agentview robot0_eye_in_hand` 这种 camera_names 用法：[robomimic cameras](https://robomimic.github.io/docs/v0.4/datasets/robosuite.html)。下一步更推荐把实验切到：

1. `LIBERO/robomimic + robot0_eye_in_hand`
2. `RLBench/CALVIN/ManiSkill` 做 3D oracle
3. `DROID/RH20T` 做真实 moving-camera 验证

最强论文路线：**先仿真 oracle 3D 证明上限，再用 DROID/RH20T 证明真实 wrist camera 有收益。**