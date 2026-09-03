# Parameter Adjustment Agent

面向 Unitree 机器人的自然语言强化学习参数调整与训练编排工具。

本项目把自然语言描述的 Unitree 机器人任务转换为经过校验的奖励候选，执行分阶段训练，生成同步 rollout 证据，并完成视觉评论、确定性评估和可恢复的实验管理。生产环境中的智能推理只能通过 OpenCLI 操作已登录的 ChatGPT 网页完成；项目不使用任何模型 API 或 API Key。

## 项目目录结构

Agent 与 Unitree 训练工程位于同一个父目录中。仓库只纳入 Agent 源码；Unitree 工程、Isaac Gym SDK、训练 checkpoint、视频和实验日志不会上传。代码和配置只保存相对路径，因此克隆后仍可复现：

```text
parameter_adjustment_agent/
├── README.md                    # 项目总说明
├── agent.html                   # 中文架构导览页
├── rl_agent/                    # 自然语言强化学习训练 Agent
└── unitree_rl_gym/              # 本地依赖，不纳入本仓库
```

`rl_agent` 的主要文件和目录如下：

```text
rl_agent/
├── pyproject.toml               # Python 包、依赖和 pytest 配置
├── requirements.txt             # 依赖版本清单
├── .env.example                 # 可选环境变量示例，不保存密钥
├── .gitignore                   # 排除实验、缓存、构建与运行产物
├── 启动上位机.sh                # 使用相对路径启动桌面上位机
├── config/
│   ├── agent.yaml               # 训练预算、机器人、实验目录等 Agent 配置
│   └── opencli.yaml             # OpenCLI 会话、超时和附件阈值配置
├── docs/agent/                  # 架构、安全、界面、实验格式和故障排查文档
├── rl_training_agent/           # Agent 的 Python 主包
│   ├── __main__.py              # `python -m rl_training_agent` 模块入口
│   ├── cli.py                   # plan/train/resume/evaluate/play 等命令
│   ├── settings.py              # YAML、环境变量和相对路径配置加载
│   ├── desktop_launcher.py      # Chrome 应用窗口启动器
│   ├── desktop_ui.py            # Tk 兼容桌面界面
│   ├── web_ui.py                # 上位机本地服务、作业、状态和日志 API
│   ├── environment/             # Unitree 环境、奖励与物理量能力检查（含指标别名规范化）
│   ├── orchestration/           # 训练编排、状态机、预算和恢复策略
│   ├── providers/               # OpenCLI ChatGPT 与离线 Mock Provider
│   ├── prompts/                 # 任务、奖励、诊断和视觉评论提示词
│   ├── rewards/                 # 奖励配置编译、校验、调度和诊断修订
│   ├── schemas/                 # 任务、奖励、实验、指标和视觉 Pydantic Schema
│   ├── training/                # 训练控制器、进程管理、train/rollout/play 入口
│   ├── evaluation/              # 确定性验收、失败模式和证据融合
│   ├── metrics/                 # TensorBoard、PPO、奖励和轨迹指标
│   ├── visual/                  # 视频、抽帧、事件、标注和接触图处理
│   ├── storage/                 # 实验存储、文件锁和实验谱系
│   ├── utils/                   # 原子文件操作与安全路径工具
│   └── web/                     # 上位机 HTML、CSS 和 JavaScript 前端
├── tests/                       # 离线单元测试和集成测试
├── artifacts/                   # 运行时公共产物，不属于源码
│   ├── environment_manifest.json # 最近一次环境能力清单
│   ├── opencli_test/            # OpenCLI 文本与附件探测结果
│   └── ui_jobs/                 # 每个上位机作业的元数据和增量日志
└── experiments/                 # 任务、候选、checkpoint、rollout 和最终报告
    └── task-<任务编号>/
        ├── state.json           # 可恢复状态机与阶段历史
        ├── task_spec.json       # 结构化任务规格
        ├── loop_status.json     # 当前闭环轮次、奖励版本、决策和剩余预算
        ├── loop_history.json    # 每轮联合验收及诊断决策历史
        ├── lineage.json         # 初始候选和每次修订的父子谱系
        ├── provider_records/    # 发给 GPT 的需求文档和原始回复
        ├── candidates/          # 候选配置、日志、指标和 model_*.pt
        └── final/               # 最终 checkpoint、配置、奖励计划和视觉证据
```

源码上传时应保留 `config/`、`docs/`、`patches/`、`rl_training_agent/`、`tests/`、顶层配置文件和启动脚本。`artifacts/`、`experiments/`、`rl_training_agent.egg-info/`、`__pycache__/` 和 `.pytest_cache/` 都是运行或安装生成内容，已由 `.gitignore` 排除；如果需要同时交付训练模型，应单独打包所需的 `model_*.pt`、对应 `config.yaml` 和评估视频。

## 安装

克隆本仓库与 Unitree 训练工程，并应用 Agent 所需的 Unitree 定制补丁：

```bash
git clone https://github.com/Madong9/parameter_adjustment_agent.git
cd parameter_adjustment_agent
git clone https://github.com/unitreerobotics/unitree_rl_gym.git
git -C unitree_rl_gym checkout 276801e
git -C unitree_rl_gym apply ../rl_agent/patches/unitree_rl_gym.patch

# 按 Unitree 官方说明安装 Isaac Gym 和 rsl_rl 后，安装 Agent
conda activate rl_agent
cd rl_agent
python -m pip install -e '.[test]'
```

补丁内容与用途见 `rl_agent/patches/README.md`。所有纳入版本控制的路径均为相对路径。`config/agent.yaml` 使用 `../unitree_rl_gym` 指向训练项目；公共产物和实验均保存在 Agent 目录内。

## 启动桌面上位机软件

本项目提供独立的软件窗口，可输入自然语言动作目标、选择 Unitree 机型、切换离线演练或真实训练，并查看实时阶段、总体进度、运行日志与最近实验。在文件管理器中可双击 `启动上位机.sh`。

如果终端提示符已经以 `/rl_agent$` 结尾，说明当前就在 Agent 目录，直接运行：

```bash
conda activate rl_agent
python -m rl_training_agent desktop
```

只有当前位于上一级 `parameter_adjustment_agent` 目录时，才需要执行 `cd rl_agent`：

```bash
conda activate rl_agent
cd rl_agent
python -m rl_training_agent desktop
```

桌面入口会使用系统已有的 Chrome/Chromium 创建一个没有地址栏、标签栏和书签栏的独立应用窗口。它使用隔离的临时浏览器配置，不读取个人浏览数据或扩展；训练服务仅监听随机的本机端口。这样可以获得稳定的中文字体、缩放和响应式布局，同时不新增 Qt 或 Electron 依赖。界面日志保存在 `artifacts/ui_jobs/<job_id>/training.log`，实验结果保存在 `experiments/<task_id>/`。软件不会连接或控制实体机器人。完整说明见 `docs/agent/DESKTOP_UI.md`。

项目同时保留浏览器版作为远程或轻量备用入口：

```bash
python -m rl_training_agent ui --no-browser
```

随后访问 `http://127.0.0.1:8765/`。浏览器版说明见 `docs/agent/WEB_UI.md`。

## 配置 OpenCLI 与 Chrome

安装 OpenCLI Chrome 扩展并连接 OpenCLI，在 Chrome 中打开 `https://chatgpt.com/` 并完成登录。默认使用绑定模式：绑定前请将目标 ChatGPT 标签页切换到前台。若希望 Agent 自行管理标签页，可在 `config/opencli.yaml` 中设置 `bind_existing_tab: false` 和 `owned_session: true`。不要在仓库中保存 Cookie、令牌或 API Key。

```bash
opencli list
opencli doctor
python -m rl_training_agent doctor
python -m rl_training_agent opencli-test
```

当生产条件不满足时（包括真实图片上传探测失败），`doctor` 会返回非零退出码；这不会影响离线 dry-run。

集成测试会生成一张无敏感信息的红色圆形图片，上传后要求 ChatGPT 返回严格 JSON，再用 Pydantic 校验，并把请求与回复记录到 `artifacts/opencli_test/`。若 Chrome 拒绝 OpenCLI 的本地路径上传并返回 `Not allowed`，Provider 会自动切换到分块 DataTransfer 上传，不需要手动更新扩展或重新选择文件。

## 离线 dry-run

以下流程不需要 GPU 或 ChatGPT 登录，会完整执行任务设计、三个候选、配置编译、模拟分阶段训练、指标生成、三路 MP4、同步 Parquet、事件检测、接触图、视觉报告、诊断、最终选择和状态持久化：

```bash
python -m rl_training_agent train \
  --task "测试机器狗稳定向前行走" \
  --dry-run \
  --provider mock
```

## 检查、规划、训练与恢复

```bash
python -m rl_training_agent inspect-env --robot go2
python -m rl_training_agent plan --task "训练一只机器狗原地向上跳跃并稳定落地" --robot go2
python -m rl_training_agent train --task "训练一只机器狗原地向上跳跃并稳定落地" --robot go2
python -m rl_training_agent status --task-id TASK_ID
python -m rl_training_agent resume --task-id TASK_ID
python -m rl_training_agent report --task-id TASK_ID
```

真实训练复用 Unitree 的实际任务注册表和 `OnPolicyRunner`，在内存中应用 Agent 生成的奖励尺度、奖励参数、课程阶段和终止条件，并写入全新的候选目录。课程边界会按本次训练迭代数等比例缩放，阶段切换时同步更新奖励、命令范围和目标参数。自动训练仅限 GPU 仿真，不会调用实机部署代码。

## 自动训练闭环

真实训练不是“一次生成奖励、训练一次就结束”。每个作业会自动执行下面的闭环：

```text
自然语言动作目标
  → 精简环境能力、机器人配置、注册奖励和可计算指标
  → GPT 生成任务规格及多个奖励候选
  → 本地规范化、白名单校验和配置编译
  → 冒烟训练、候选筛选、多随机种子完整训练
  → 多视角 rollout、完整轨迹指标和视觉评论
  → 确定性联合验收与 GPT 结构化诊断
  → 奖励/课程修订并续训或重训
  → 再次 rollout 和验收，直到完成或达到真实阻塞条件
```

诊断阶段会再次把当前奖励计划、可注册奖励及参数、命令空间、可计算指标、逐项奖励均值、PPO 数据、视觉结论、物理指标、历史轮次和剩余预算精简后交给 GPT。GPT 只能新增已注册奖励，或修改白名单字段；本地随后重新执行奖励符号、权重上限、课程边界、任务指标覆盖和配置编译校验。普通的动作未达标、奖励投机或视觉/数值不通过会继续自动修订，不再直接进入人工审核。

只有 Provider/登录不可用、模型要求不可计算的物理量、必须由用户决定的任务歧义、修订无法通过安全校验，或迭代/修订预算耗尽时，状态才会进入 `HUMAN_REVIEW`。上位机会显示“闭环第 N 轮、奖励 vN、当前决策和训练阶段”；机器可读明细保存在 `loop_status.json` 和 `loop_history.json`。

奖励校验会根据环境函数的真实语义检查符号：`orientation`、`base_height`、`collision` 等返回非负代价值，必须使用负权重。全部候选未通过数值健康或硬门槛时，流程会直接停止，不会再从负无穷得分中强行选择一个候选。Go2 的后腿站立任务使用 `rear_leg_stand` 和 `rear_leg_walk`；用前腿倒立行走的任务使用 `front_leg_stand` 和 `front_leg_walk`。两类行走奖励都由对应双腿支撑姿态门控，可避免四足爬行或普通行走利用速度奖励。相同自然语言任务再次运行时会创建独立 `run-<编号>-candidate-*` 目录，旧 checkpoint 会保留，但不会混入新训练。

上位机只有在 Agent 最终状态为 `COMPLETED` 时才显示“训练完成”。这表示视觉目标、任务物理指标和硬安全约束在同一轮中全部通过，不只是 PPO 进程跑完。进程正常退出但状态为 `HUMAN_REVIEW` 时会显示“等待人工复核”，并进一步区分“训练尚未开始”和“训练完成但未通过最终验收”。

## 查看已经训练好的策略效果

`evaluate` 会加载指定的真实 checkpoint，在 Isaac Gym 中运行单环境确定性 rollout，并录制前视、侧视和俯视三路 MP4，同时保存轨迹与逐项奖励数据。该命令不会重新训练，也不会控制实体机器人；`--provider mock` 只表示本步骤不需要 GPT，不会把 checkpoint 替换成模拟数据。

### 使用 Isaac Gym Viewer 实时观看

如果希望像 Unitree 原生 `play.py` 一样直接打开仿真窗口，而不是先录制视频，请在 Ubuntu 图形桌面终端中运行：

```bash
conda activate rl_agent
python -m rl_training_agent play \
  --task-id task-2c93da8f8c \
  --checkpoint experiments/task-2c93da8f8c/candidates/candidate-01-v01/Jul17_16-05-08_candidate-01-v01-seed-1/model_3000.pt
```

该命令默认只创建 1 个环境，并直接加载指定策略。关闭 Isaac Gym Viewer 窗口或在启动命令的终端按 `Ctrl+C` 即可退出。若终端没有可用的 Ubuntu 图形会话（例如纯 SSH），Viewer 无法打开，此时使用下面的 `evaluate` 离屏录像方式。

需要同时观察少量并行环境时，可增加 `--num-envs`，允许范围为 1 到 16：

```bash
python -m rl_training_agent play \
  --task-id task-2c93da8f8c \
  --checkpoint experiments/task-2c93da8f8c/candidates/candidate-01-v01/Jul17_16-05-08_candidate-01-v01-seed-1/model_3000.pt \
  --num-envs 4
```

### 离屏录制视频

如果终端已经位于 `rl_agent` 目录，可以直接查看本次“机器狗后腿站立行走”任务的 Seed 1 最终策略：

```bash
conda activate rl_agent
python -m rl_training_agent evaluate \
  --task-id task-2c93da8f8c \
  --checkpoint experiments/task-2c93da8f8c/candidates/candidate-01-v01/Jul17_16-05-08_candidate-01-v01-seed-1/model_3000.pt \
  --provider mock
```

成功后，视频和数据位于：

```text
experiments/task-2c93da8f8c/final/evaluation_rollout/
├── front.mp4
├── side.mp4
├── overview.mp4
├── trajectory.parquet
├── rewards.parquet
├── behavior_evidence.json
├── contact_sheet_annotated.png
├── contact_sheet_multiview.png
├── visual_attachment_manifest.json
├── visual_report_enhanced.json
└── metadata.json
```

录像完成后，可对现有 rollout 单独执行增强视觉审核，不会重新训练：

```bash
python -m rl_training_agent visual-audit \
  --task-id task-2c93da8f8c \
  --rollout final/evaluation_rollout \
  --provider opencli
```

该命令会扫描完整轨迹，生成 command/实际速度、全部非足刚体接触力、连续直立区间和关键事件窗口，并生成均匀采样、连续标注及同帧三视角接触图。结果保存为 `visual_report_enhanced.json`；原有 checkpoint 和训练日志不会改变。

优先观看侧视视频：

```bash
xdg-open experiments/task-2c93da8f8c/final/evaluation_rollout/side.mp4
```

也可以分别打开前视和俯视视频：

```bash
xdg-open experiments/task-2c93da8f8c/final/evaluation_rollout/front.mp4
xdg-open experiments/task-2c93da8f8c/final/evaluation_rollout/overview.mp4
```

其他任务或候选的通用命令如下：

```bash
python -m rl_training_agent evaluate \
  --task-id TASK_ID \
  --checkpoint experiments/TASK_ID/candidates/CANDIDATE/RUN/model_3000.pt \
  --provider mock
```

闭环完成后，可以直接加载统一复制出的最终 checkpoint：

```bash
python -m rl_training_agent play \
  --task-id TASK_ID \
  --checkpoint experiments/TASK_ID/final/checkpoint.pt
```

不知道历史候选 checkpoint 在哪里时，可以列出所有模型：

```bash
find experiments/TASK_ID/candidates -type f -name 'model_3000.pt'
```

## 结果目录

全局环境检查结果保存在 `artifacts/environment_manifest.json`。每个任务位于 `experiments/<task_id>/`，包含 `state.json`、`task_spec.json`、实验谱系、候选配置、提示词与回复、指标、checkpoint、同步 rollout、诊断、报告及 `final/` 最终产物。详细格式见 `docs/agent/EXPERIMENT_FORMAT.md`。

## 测试

本机 ROS 安装会向 Python 3.8 暴露不兼容的全局 pytest 插件，因此需要关闭第三方插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

## 常见问题与当前限制

- 页面出现“登录”或验证码：在 Chrome 中手动恢复，重新执行 `opencli doctor`，然后运行 `resume`。
- 找不到输入框或上传控件：更新 OpenCLI，绑定模式下将 ChatGPT 标签页切换到前台，再运行 `opencli-test`。
- 日志出现 `set-file-input: Not allowed`：当前 Provider 会自动使用通用 `#upload-files` 输入框和分块 DataTransfer 兜底；若仍失败，查看 rollout 目录中的 `visual_provider_error.txt`，其中会保留原始错误。
- GPU OOM：降低分阶段控制器使用的环境数量。
- 离屏摄像机没有图像：确认 CUDA 图形设备可见；仓库补丁仅在 `env.record_video` 启用时保留图形设备。
- 历史 Go2“后腿站立行走”示例虽然保存了 checkpoint，但增强评估已确认其学习成低姿态爬行，不能视为成功策略。奖励符号、课程执行、终止门槛和候选筛选修复后必须创建新作业重新训练；旧 checkpoint 仅用于回归分析。日常自动测试不会重复执行耗时训练，也不会部署到实体机器人。实机部署明确位于本 Agent 范围之外，必须经过独立人工批准。

更多说明见 `docs/agent/TROUBLESHOOTING.md` 和 `docs/agent/IMPLEMENTATION_STATUS.md`。
