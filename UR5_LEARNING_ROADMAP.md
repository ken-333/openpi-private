# UR5 + π0.5 学习路线

任务目标：用 SpaceMouse 采集 UR5 数据，训练 π0.5 完成"抓取指定物品放入指定颜色盘子"。

---

## 硬件配置

- 机器人：UR5（多台）
- 遥操作：3DConnexion SpaceMouse
- GPU：2× RTX 4090（Linux 训练机）
- 相机：固定俯视相机（base）+ 腕部相机（wrist）

---

## 整体数据流

```
SpaceMouse 6轴输入
    ↓ pyspacemouse
Cartesian 速度 [vx, vy, vz, vrx, vry, vrz]
    ↓ ur_rtde speedl()
UR5 实时运动
    ↓ 每帧记录 getActualQ() + 夹爪 + 两路图像
HDF5 文件 (episode_0000.hdf5 ...)
    ↓ convert_ur5_to_lerobot.py
LeRobot Dataset (HuggingFace 格式)
    ↓ compute_norm_stats.py
归一化统计
    ↓ train.py
π0.5 LoRA 微调模型 checkpoint
    ↓ serve_policy.py (WebSocket server)
UR5 Robot Client (推理部署)
```

---

## Step 1｜数据采集脚本

**文件：** `examples/ur5/collect_data.py`

**功能：**
- 读取 SpaceMouse → 映射为 UR5 Cartesian 速度
- 实时控制 UR5（`speedl`）
- 每帧记录：`joints(6,)`、`gripper(1,)`、`base_rgb`、`wrist_rgb`
- 键盘控制：`s` 保存 episode，`d` 丢弃，`q` 退出
- 每条 episode 保存为一个 HDF5 文件

**关键依赖：** `pyspacemouse`, `ur_rtde`, `opencv-python`, `h5py`

**HDF5 文件结构：**
```
episode_0000.hdf5
├── /observations/qpos        (T, 7)   joints + gripper
├── /observations/images/base (T, H, W, 3)
├── /observations/images/wrist(T, H, W, 3)
├── /action                   (T, 7)   下一帧的 qpos（t 时刻 action = t+1 时刻 state）
└── /task_instruction         string   如 "pick up the red block and place it in the blue plate"
```

**采集标准：**
- 频率：50Hz 控制，10Hz 记录（每 5 帧取 1 帧）
- 数量：每个任务变体先采 50 条，跑通 pipeline 后扩到 200+
- 分辨率：640×480（训练时 resize 到 224×224）

---

## Step 2｜数据转换脚本

**文件：** `examples/ur5/convert_ur5_to_lerobot.py`

**参考：** `examples/aloha_real/convert_aloha_data_to_lerobot.py`

**功能：**
- 读取所有 `episode_*.hdf5`
- 创建 `LeRobotDataset`，定义 features（state, action, images）
- 逐帧写入，`save_episode(task=task_instruction)`
- 最后 `dataset.consolidate()`，可选 `push_to_hub()`

**LeRobot features 定义：**
```python
features = {
    "observation.state":              {"dtype": "float32", "shape": (7,)},
    "observation.images.base":        {"dtype": "video",   "shape": (3, 480, 640)},
    "observation.images.wrist":       {"dtype": "video",   "shape": (3, 480, 640)},
    "action":                         {"dtype": "float32", "shape": (7,)},
}
```

---

## Step 3｜Policy Transforms

**文件：** `src/openpi/policies/ur5_policy.py`

**参考：** `src/openpi/policies/libero_policy.py`

**需要写两个 dataclass：**

`UR5Inputs`：把 LeRobot dataset 的字段映射成模型期望的格式
- `state` = concat(joints, gripper) → shape (7,)，padding 到 (32,)（π0.5 action_dim=32）
- `image` = {"base_0_rgb": ..., "left_wrist_0_rgb": ..., "right_wrist_0_rgb": zeros}
- `image_mask` = right wrist 设 False（π0.5 模式）
- 注意：LeRobot 存图像格式是 float32 (C,H,W)，模型要 uint8 (H,W,C)，需要转换

`UR5Outputs`：把模型输出的 action（32维）截取前 7 维
- `actions = data["actions"][:, :7]`

---

## Step 4｜Training Config

**文件：** `src/openpi/training/config.py`（末尾追加到 `_CONFIGS` 列表）

**使用 π0.5 + LoRA：**
```python
model=pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",        # 冻结 PaliGemma，只训 LoRA 层
    action_expert_variant="gemma_300m_lora",   # 同上
),
freeze_filter=pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter(),
ema_decay=None,   # LoRA 微调关掉 EMA
```

**加载官方 π0.5 checkpoint：**
```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "gs://openpi-assets/checkpoints/pi05_base/params"
),
```

**数据 delta action 处理（关节 6 维用 delta，夹爪不用）：**
```python
delta_action_mask = _transforms.make_bool_mask(6, -1)
# 训练时：DeltaActions(delta_action_mask)
# 推理时：AbsoluteActions(delta_action_mask)
```

---

## Step 5｜训练

```bash
# 计算归一化统计（必须在训练前做）
uv run scripts/compute_norm_stats.py --config pi05_ur5_lora

# 启动训练（JAX，自动使用两张 4090）
CUDA_VISIBLE_DEVICES=0,1 uv run scripts/train.py pi05_ur5_lora \
    --exp-name pick_place_v1

# 监控（wandb 或 tensorboard）
# loss 应在 5k steps 内开始明显下降
# 参考收敛：30k steps，约 4-6 小时（双 4090）
```

---

## Step 6｜部署推理

**架构：**
```
[GPU 训练机]                      [UR5 控制机]
serve_policy.py                   ur5_inference.py
WebSocket server :8000    ←→      openpi-client runtime
```

```bash
# 启动 policy server
uv run scripts/serve_policy.py --env UR5 \
    --checkpoint-dir checkpoints/pi05_ur5_lora/pick_place_v1
```

Robot client 需要实现 `Environment` 接口（参考 `examples/droid/main.py`）：
- `get_observation()` → 返回 joints, gripper, base_rgb, wrist_rgb, prompt
- `apply_action(action)` → 解析 7 维 action，控制 UR5 + 夹爪

---

## 任务设计：抓取 + 放置

**Language instruction 格式：**
```
"pick up the {object} and place it in the {color} plate"
```

**变体举例（每个变体采 50 条）：**
| object | plate color | instruction |
|--------|-------------|-------------|
| red block | blue | "pick up the red block and place it in the blue plate" |
| blue ball | red | "pick up the blue ball and place it in the red plate" |
| yellow cup | green | "pick up the yellow cup and place it in the green plate" |

**建议先从 1 个 object + 2 个 plate color 开始，跑通 pipeline 再扩展。**

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 动作抖动 | 没用 delta action | 确认 DeltaActions transform 已加 |
| 模型忽略语言指令 | prompt_from_task=False | 改 True，确认 HDF5 里有 task_instruction |
| 推理首帧很慢 | JAX JIT 编译 | 推理循环前先 warmup 一次 |
| VRAM OOM | batch_size 过大 | LoRA 下 batch=32 足够，可降到 16 |
| norm stats 不匹配 | 用了官方 ur5e stats | 用自己数据重跑 compute_norm_stats.py |

---

## 当前进度

- [ ] Step 1：写 collect_data.py
- [ ] Step 2：写 convert_ur5_to_lerobot.py
- [ ] Step 3：写 ur5_policy.py
- [ ] Step 4：加 TrainConfig
- [ ] Step 5：训练并监控 loss
- [ ] Step 6：真机部署
- [ ] 清理：删除 laptop 上临时安装的库（pyspacemouse, opencv-python, h5py, numpy）windows系统的laptop装了（远程写代码用）  linux省略这一步