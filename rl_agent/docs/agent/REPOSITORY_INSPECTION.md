# 仓库检查报告

检查日期：2026-07-15（Asia/Shanghai）。

## 工作区与运行环境

- 工作区根目录本身不是 Git 仓库；`../unitree_rl_gym` 是独立 Git 仓库，修改前提交为 `276801e`。检查前已经存在未跟踪的 `isaacgym/` 和 `rsl_rl/` 目录。
- 用户指定的 Agent 目录在实现前为空。
- Conda 环境 `rl_agent`：Python 3.8.20、PyTorch 2.3.1，CUDA 检查结果为可用。
- 本地 Isaac Gym 从 `isaacgym/_bindings/linux-x86_64/gym_38.so` 加载，目录结构对应 NVIDIA Isaac Gym Preview 4。
- Agent 初始缺少 Pydantic、pytest、OpenCV、pandas 和 PyArrow；当前依赖已经写入 `requirements.txt` 和 `pyproject.toml`。

## 实际 Unitree 集成点

- 训练入口：`../unitree_rl_gym/legged_gym/scripts/train.py` 中的 `train`，负责创建环境和 runner，再调用 `OnPolicyRunner.learn`。
- 播放/评估入口：`../unitree_rl_gym/legged_gym/scripts/play.py` 中的 `play`，会关闭噪声与域随机化、加载 runner 并执行推理。
- 已注册机器人：`legged_gym/envs/__init__.py` 中的 `go2`、`h1`、`h1_2` 和 `g1`。Go2 使用 `GO2RoughCfg`、`GO2RoughCfgPPO` 与 `LeggedRobot`。
- Go2 机器人与配置：`legged_gym/envs/go2/go2_config.py`；包含 12 个动作、位置控制、Go2 URDF、机身接触终止和实验名 `rough_go2`。
- 基础观测：`legged_gym/envs/base/legged_robot.py` 中的 `LeggedRobot.compute_observations`；拼接机身线速度/角速度、投影重力、三个命令、关节位置/速度和上一步动作，共 48 个策略观测值。
- 奖励：`LeggedRobot._prepare_reward_function` 会把每个非零 `rewards.scales.<name>` 动态映射到 `_reward_<name>`。基础实现位于 `legged_robot.py`，人形机器人还提供机器人专用奖励。Go2 继承基础奖励尺度，并覆盖扭矩与关节限制相关尺度。
- Episode 奖励：`LeggedRobot.reset_idx` 填充 `extras['episode']['rew_<name>']`；`OnPolicyRunner.log` 把所有 episode 条目写到 TensorBoard 的 `Episode/` 命名空间。
- 终止逻辑：`LeggedRobot.check_termination`；包含禁止部位接触力、滚转/俯仰阈值和超时。
- 命令生成：`LeggedRobot._resample_commands`；x/y 速度和 heading/yaw 范围来自 `LeggedRobotCfg.commands`。
- PPO：`legged_robot_config.py` 中的 `LeggedRobotCfgPPO`；包含 PPO 超参数、每环境 24 步、默认 1500 iteration、每 50 iteration 保存一次。
- Runner 与日志：`../unitree_rl_gym/rsl_rl/rsl_rl/runners/on_policy_runner.py`；TensorBoard 标签包括 `Episode`、`Loss`、`Policy`、`Perf` 和 `Train`。
- Checkpoint：`OnPolicyRunner.save`；使用 Torch 字典保存 `model_state_dict`、`optimizer_state_dict`、`iter` 和 `infos`，文件名为 `model_<iteration>.pt`。
- 恢复机制：`TaskRegistry.make_alg_runner` 和 `helpers.get_load_path`；CLI 参数包括 `--resume`、`--load_run` 和 `--checkpoint`。
- 原有录制能力：`play.py` 声明了 `RECORD_FRAMES`，但没有可工作的同步录制实现。Agent 因此新增独立离屏摄像机录制器。
- 检查时发现的已有日志：`logs/rough_go2/Jul14_21-17-34_/events.out.tfevents...`。
- 实现 Agent 前没有发现 pytest 测试套件。

## OpenCLI

2026-07-18 的 `opencli doctor` 报告版本 1.8.6、daemon 正常、Chrome 扩展 1.0.22 已连接、profile 已连接。`opencli chatgpt status -f json` 报告 ChatGPT 已连接且已登录。当前 CLI 提供 `chatgpt ask/new/read/send/status`，以及 `browser` 的 state/find/fill/upload/wait/eval/tab/bind/close 等命令。实现中使用 browser 接口完成语义交互；原生图片上传被 Chrome 拒绝时自动切换到分块 DataTransfer。
