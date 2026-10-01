# 实现状态

2026-09-30：新增独立训练准入策略，默认 `budgeted_exploration`。候选物理失败不再等同于目标不可训练；能力/指标和真实环境运行证据完整时，允许最多 3000 累计迭代、1 次修订的仿真探索。缺失求解器时可隔离执行默认姿态健康检查；Mock、基础设施故障或必要能力缺失仍不放行。冻结验收合同，恢复校验 run_id 并保留预算消耗，上位机分别展示物理结果与准入决定。通用事件触发全身控制仍未实现，探索准入不是动作验证通过。

> 下方“实现列表”概述项目能力；带日期的验证条目是历史快照。当前 Task Feasibility 的真实物理状态以最新的 2026-09-27 Isaac Gym 记录和 [`TASK_FEASIBILITY.md`](TASK_FEASIBILITY.md) 为准。

## 已实现并完成本地验证

- Pydantic 任务、奖励、实验、指标、视觉和诊断 Schema。
- 真实 OpenCLI browser Provider：健康检查、bind/owned 会话、稳定 CSS 与语义定位、输入验证、显式发送按钮点击、用户消息提交确认、上传、最新回复轮询、JSON 提取与修复、安全重试、超时、登录/验证码恢复，以及请求/回复记录。
- 百炼 GLM Provider：OpenAI 兼容 Chat Completions、环境变量密钥、非思考 JSON 输出、Pydantic 校验与修复、有限重试、超时、错误解析和无鉴权头审计记录。
- 多 Agent 角色路由：GPT/OpenCLI 任务理解与视觉评价、百炼 GLM 奖励设计与训练诊断，以及兼容 OpenCLI/Mock 模式。
- 固定提示词流水线：`TaskIntentSpec`、环境/RAG/长期记忆上下文快照、版本化 PromptCompiler、SHA-256 和确定性 RewardReviewAgent。
- 四层记忆模块：任务级工作记忆、严格门控的情景记忆、独立证据升级的语义记忆、Prompt/Schema/规则程序记忆，以及只归档不删除的遗忘机制。
- 仅在显式选择时使用的确定性 Mock Provider。
- 基于实际仓库的环境能力和奖励检查，以及机器可读 manifest。
- 仅注册表奖励编译、校验、diff/hash/版本元数据，以及隔离生成代码的 AST/张量检查。
- 受限真实训练与 rollout 包装器、进程组、超时/停止/状态/日志/PID/退出码、checkpoint 发现和恢复。
- 与现有 episode/TensorBoard 日志集成的原始值和加权奖励统计。
- TensorBoard、PPO、奖励、轨迹指标及报告构建器。
- 同步三相机 MP4 与 Parquet、元数据、事件、抽帧/裁剪/接触图、叠加标注、rollout 选择、视觉评论和确定性融合。
- 带文件锁的实验树、实验谱系、预算、原子可恢复状态机、CLI 和完整 dry-run。
- 自动评估闭环：指标别名规范化、紧凑诊断能力清单、结构化奖励/课程修订、版本安全编译、checkpoint 续训/父版本回退/重训、多轮复评、预算终止和完整修订审计。
- 本地 RAG 训练经验模块：中文单字/二元词与英文变量混合分词、BM25 持久化索引、文档与历史实验白名单、机器人及完成状态加权、设计/诊断阶段限长注入、来源审计、当前任务排除、提示注入边界和失败开放降级。
- Task Feasibility Pipeline：确定性能力检查、`LEVEL_0`—`LEVEL_6` 证据层级、统一子阶段报告、静态 Pinocchio IK/CoM/支撑几何/重力 RNEA、torque/friction/关节轨迹 validators，以及 locomotion 的 gait/FootTrajectory 原型。动态链每个 Unitree policy step 采样足端笛卡尔目标，以 Pinocchio 多脚 IK 热启动生成 joint reference，经独立 `q/qdot/qddot` 轨迹检查后进入 Isaac Gym position-action rollout；只有真实完整 rollout 才能报告 `DYNAMIC_PHYSICS_VALIDATED`。真实 Isaac Gym 调用已隔离到 worker，原生扩展崩溃/超时不会杀死上位机，而会保守返回 `CAPABILITY_ONLY`。单腿、前/后腿支撑会运行有限候选 IK/静态必要条件搜索；倒立、跳跃、特技和缺专用后端的操作任务保持人工复核。真实 rollout 失败/模型缺失转 `HUMAN_REVIEW`，仅明确不支持的能力转 `FAILED`；Mock 永不冒充真实物理证据。
- 本地强化学习上位机：动作任务下发、机型与模式选择、实时状态和进度、增量日志、历史实验、安全停止、响应式操作台，以及受限 JSON API。
- 上位机闭环监控：显示当前轮次、奖励版本、诊断决策和剩余预算，并严格区分 PPO 进程结束、等待人工复核与联合验收完成。
- 上位机人工复核恢复：显示具体阻塞原因，并可复用当前奖励版本、多种子 checkpoint、剩余预算和已采集 rollout 继续自动闭环。
- 桌面上位机软件：无地址栏的隔离 Chrome 应用窗口、现代中文字体和缩放、响应式操作台、作业互斥、窗口关闭后安全停止和相对路径启动脚本；Tk 作为无 Chromium 时的兼容后备。
- 自动测试：覆盖静态/动态 MotionType 分流、TROT 步态参数生成与别名规范化、相位/频率/占空比动作映射、速度范围和单位约束、动态指标采集、缺失步态拒绝、真实失败状态映射、Mock 隔离、训练就绪判定、状态机事件、Isaac Gym action 映射及“不创建 PPO runner/不训练”。
- Agent 生产代码和测试代码中的所有函数均具有中文 docstring；AST 审计缺失数为零。
- 固定真实动作基准协议、多种子最差/中央/最好视觉保守聚合、视觉一致性置信度校准、零命令奖励投机探针、严格状态转换、幂等证据复用和恢复前置校验。
- Provider 角色注册表和能力声明、BM25/本地 TF-IDF 混合 RAG、语义记忆反例与 `superseded` 治理、JSONL 状态/Provider/GPU 事件、多 GPU 持久等待队列、CI、Conda 环境入口和产物迁移器。

## 外部或运行时验证状态

- 2026-09-29（双腿支撑六项数值规划）：新增 `feasibility/balance`，实现质心迁移、接触时序、基座姿态、足端轨迹、自由浮动 Pinocchio IK、摩擦锥接触力优化和 RNEA 逆动力学；数值就绪后复用 Unitree Isaac Gym PPO 环境短时验证。静态双前腿支撑可进入物理 rollout，双前腿交替前走因单足阶段动力学残差保持 `INCONCLUSIVE`；Mock 不升级物理等级。全量 223 项测试通过。

- 2026-09-28（当前通用多层级 Feasibility）：全量收集 217 项测试并全部通过。新增 `MotionConstraintSpec`、五类确定性规划器注册表与全身求解能力门；LLM 不能注入 joint/torque/policy，缺少浮动基座 IK、逆动力学、接触力或轨迹优化时保持 `INCONCLUSIVE`。单腿四支撑侧、前/后腿支撑三档抬脚高度候选继续逐候选运行真实 Pinocchio IK 与静态必要条件检查。当前 IK-reference 版本的真实 Go2 倒退 `0.3 m/s` 短时验证在隔离 worker 内完成 `0.2 秒 / 10 步`，无跌倒和关节越限，但速度误差约 `0.304 m/s` 且部分足端在 locomotion 采样段未触地，故真实返回 `PHYSICS_FAILED`。这只否定本次短时原型，不代表 PPO 无法学习。

- 2026-09-27（静态 Isaac Gym 可行性预检）：Go2 默认姿态 Pinocchio IK 收敛；真实 1 秒 rollout（50 策略步）检查通过，报告等级为 `STATIC_PHYSICS_VALIDATED`。末段稳态窗口最大关节跟踪误差 `0.545 rad`（阈值 `0.75 rad`），最低 base 高度 `0.214 m`；该结果只覆盖静态默认站姿，不代表策略已训练或动态步态可行。
- 2026-09-27（动态 Isaac Gym 探针接入）：新增 `DynamicMotionPrototypeGenerator` 和 `IsaacGymDynamicValidator`，默认预算为 5 秒/250 策略步；必须完整 rollout，并通过速度误差、稳定性、足端接触/滑移、关节位置/速度及力矩饱和检查才会报告 `DYNAMIC_PHYSICS_VALIDATED`。GPU PhysX 实测 0.2 秒倒退 0.3 m/s 探针完成 10 步，`backend=isaacgym`，没有摔倒/关节越限，但因最大接触期足端滑移 `0.324 m/s` 超过 `0.25 m/s` 阈值，正确返回 `PHYSICS_FAILED`；这是短探针失败，不是 5 秒完整动作验证，也不证明机器人无法通过 PPO 学会行走。
- 2026-09-28（参数化动态步态原型）：`DynamicMotionPrototype` 新增 `VelocityTrajectory` 和 `GaitPattern`，默认 Go2 trot 为 `2 Hz / duty_factor=0.5`、FR/RL 与 FL/RR 对角相位；频率、占空比、相位会实际驱动 Isaac Gym 的逐步开环探针。自动测试验证模型约束和张量映射；此代码更新后的完整 5 秒 GPU rollout 尚未作为通过证据，Mock 测试不替代真实 PhysX 验收。
- 2026-09-28（参数化 TROT 历史 GPU 预检，旧固定波形版本）：真实 Isaac Gym `Go2 backward 0.3 m/s` 完整执行 `5 秒 / 250 步`；无跌倒/关节越限，但速度误差 `0.267 m/s` 高于低速自适应阈值 `0.075 m/s`，足端最大滑移 `2.102 m/s` 高于 `0.25 m/s`，因此保守返回 `PHYSICS_FAILED`。该数据来自当前 Pinocchio IK 参考接入之前的固定开环波形版本，仅作历史基线，不代表新版本已通过。
- 2026-09-28（动态 FootTrajectory→IK 编译闭环）：真实 Go2 URDF 上逐采样调用 Pinocchio IK，按 PPO default pose/action scale 编译每步 position reference，并检查 joint position/qdot/action clip；生成的参考还进入 `TrajectoryValidator` 做 `q/qdot/qddot` 与可用限位检查。Isaac Gym 在独立 worker 中执行，测试覆盖 `SIGSEGV/-11` 降级路径；本次 worker 已成功完成真实 10 步 rollout 并如实返回 `PHYSICS_FAILED`，未达到动态通过。
- 2026-09-27：`python -m compileall -q rl_training_agent` 通过；全量 pytest 为 189 passed，只有既有 PyTorch `torch.meshgrid` indexing warning。
- 2026-09-27（旧 MuJoCo 转换实验，已停用作 Go2 真实验收）：此前试验曾以 URDF 派生 MJCF 做 MuJoCo rollout，并测到关节误差和 base 下沉；其动力学标定存在疑问，因此当前生产链改用 Unitree Isaac Gym，不再依赖该转换模型。
- 2026-09-27（较早的历史快照，已由上方物理预检更新取代）：全量测试曾为 165 项；当时环境无 Pinocchio 且 Go2 无原生 MJCF，故真实预检只能返回 `CONDITIONAL`。本次更新已安装/锁定 Pinocchio，并实现 URDF 派生 MJCF。
- 已在 `rl_agent` 环境验证 Isaac Gym 可导入且 CUDA 可用。
- 2026-09-22 完整复检确认 Python 3.8.20、PyTorch 2.3.1、Isaac Gym、CUDA、训练/播放入口、相对路径配置、Go2 环境清单、上位机静态资源和 114 项自动测试均正常；多 Agent 离线端到端任务已走完全部角色状态并完成模拟联合验收，工作/程序记忆正确落盘，情景记忆门控正确拒绝把 dry-run 结果晋升为生产经验。
- 当前工作站未设置 `DASHSCOPE_API_KEY`；百炼请求协议已通过模拟 Chat Completions 服务测试，但尚未执行真实云端 GLM 往返。配置密钥和业务空间端点后应先运行 `doctor`，再启动真实训练。
- 2026-09-22 检查时 OpenCLI 可执行文件存在，但浏览器桥接健康检查超时，因此当前 `production_ready=false`。该状态不影响离线演练，但开始真实训练前必须恢复 Browser Bridge 并重新运行 `doctor` 与 `opencli-test`。
- 2026-07-17 已使用 Go2、GPU PhysX 和 64 个并行环境完成 1 次真实 PPO 启动验证：Actor/Critic 网络成功创建，第 0 次学习迭代完成，采样速度约 3732 steps/s，并生成 TensorBoard 事件以及 `model_0.pt`、`model_1.pt`。该验证只确认训练链路可运行，不代表动作已经收敛。
- 尚未执行耗时的 3000 iteration 完整收敛训练；完整的启动、监控、停止和恢复路径已经具备。
- 已实现真实离屏 rollout，但仍需要真实 checkpoint 和 GPU 图形上下文进行完整实测。
- 2026-07-18 已通过真实 OpenCLI 网页文本往返、三张训练接触图上传、视觉工具分析和严格 Pydantic 校验。OpenCLI 1.8.6 / 扩展 1.0.22 的 `set-file-input` 仍会返回 CDP `-32000 Not allowed`，但 Provider 会自动切换到分块 DataTransfer，不再构成图片上传阻塞。真实验证记录位于 `artifacts/opencli_visual_fix_test/`。
- 2026-07-18 已使用现有 `model_1500.pt` 重新录制跟随 base/yaw 的三视角 rollout，并完成增强视觉复评。最终报告对 250 帧执行完整同步扫描，五类 `evidence_findings` 均有明确结论，`uncertain_items` 为空；全非足刚体扫描纠正了旧 contact 列表造成的碰撞假阴性。
- 未实现或授权任何实体机器人部署。

Agent 模块没有关键占位实现。真实长训练的学习效果属于实验结果，不属于代码实现的替代品。
