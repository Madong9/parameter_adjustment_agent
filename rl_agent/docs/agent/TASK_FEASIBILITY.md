# 训练前动作可行性预检

该流水线位于任务理解之后、RAG/奖励设计/PPO 之前。目标是使用本项目 Unitree PPO 的真实 Go2 资产和仿真配置，做低预算物理合理性检查；它不训练策略，也不证明机器人已经学会用户动作。

## 通用动作训练准入（2026-09-30）

物理探针结论与训练准入现在独立存储。`PHYSICS_FAILED` 保留为本次候选执行失败；缺少求解器或优化未收敛仍为 `INCONCLUSIVE`。两者都不能据此证明目标整体不可学习。

`feasibility/admission.py` 中的 `TrainingAdmissionPolicy` 核对真实机器人模型、关节/执行器、策略观测、命令、成功/失败指标，以及当前 Isaac Gym 运行证据。已有真实失败 rollout 可以证明仿真确实运行；没有目标 rollout 时，在独立 worker 中执行一次默认姿态健康检查，其通过只证明基础设施可用。原生崩溃、超时、Mock、缺模型或必要指标均不能放行。

| 训练准入 | 含义 |
| --- | --- |
| `ALLOW_TRAINING` | 能力与目标物理探针通过，继续奖励设计/审查/编译 |
| `ALLOW_BUDGETED_EXPLORATION` | 能力和仿真环境就绪，目标探针失败或未知，仅允许预算内仿真探索 |
| `NEEDS_REVIEW` | 缺能力证据、模型、指标、环境运行证据、目标澄清，或 strict 模式未通过 |
| `REJECT` | 当前机器人/环境明确缺少任务所需能力 |
| `DRY_RUN_ONLY` | 仅演练，不能恢复成真实训练 |

在 `config/agent.yaml` 配置：

```yaml
feasibility_admission_mode: budgeted_exploration  # strict 恢复必须目标物理通过的门控。
exploration_max_iterations: 3000  # 候选、种子和闭环修订合计的 PPO 迭代数。
exploration_max_revisions: 1  # 自动修订次数。
```

实际预算取准入上限、本地总上限与模型提出的任务预算的最小值，恢复不会重置或扩大原准入额度。探索模式按“首轮训练 + 尚未使用的修订轮次”平均预留多种子训练额度，避免首轮耗尽预算后产生无法执行的修订许可。奖励编译后再检查成功/安全指标注册情况以及候选筛选、多种子训练的最低预算。探索任务可以 `training_readiness.ready=true`，同时保持 `validation_level=PHYSICS_FAILED`；不会自动标记物理通过或策略完成。最终数值/视觉验收与记忆晋升规则保持独立。

产物包括 `training_admission.json`、`training_readiness.json`、可选的 `environment_health.json` 和 `acceptance_contract.json`。目标探针出现 NaN/Inf 等非有限状态时，必须由隔离默认姿态检查独立证明环境健康。恢复核对准入 run_id；奖励修订和恢复校验验收合同绑定的任务语义、行为要求、成功/安全阈值，禁止更换目标或降低阈值后复用旧 checkpoint。缺少完整验收合同的历史任务不能恢复，需要重新下发。上位机同时展示探针结果、准入理由和探索预算。仅修改配置不会把已停止的旧任务自动恢复；训练前停止且没有 checkpoint 的任务需重新下发。

通用约束新增 `hard_constraints`、`soft_preferences`、`task_requirements` 和阶段级 `forbidden_contacts`、`optional_contacts`、`transition_conditions`。双腿平衡明确区分准备、质心转移、卸载和目标保持/行走，准备阶段允许四足着地。切换条件是规划协议，当前并未实现通用事件触发 WBC；跳跃、转向、操作等缺失的专用求解器仍如实报告。机器人能力来自已有 RobotModel 与环境清单，奖励设计上下文同时收到这些来源、原型风险与准入预算。

## 通用训练前检测架构

```text
用户自然语言
    ↓
TaskIntentSpec（LLM 只提取动作语义）
    ↓
MotionConstraintCompiler（本地 Schema 校验）
    ↓
MotionConstraintSpec
    ├─ 动作阶段与持续时间
    ├─ 机身/质心/速度/末端目标
    ├─ 接触语义与安全约束
    └─ 允许假设；禁止 joint/torque/policy
    ↓
DeterministicMotionPlannerRegistry
    ├─ gait
    ├─ com
    ├─ contact_schedule
    ├─ base_pose
    └─ end_effector
    ↓
WholeBodyMotionSolver
    ├─ Pinocchio IK
    ├─ floating-base IK
    ├─ inverse/centroidal dynamics
    ├─ contact-force optimization
    └─ trajectory optimization
    ↓
动作专用 Isaac Gym Validator
    ↓
稳定性、接触、力矩、速度、能量和轨迹完整性
```

`MotionConstraintSpec` 和规划器注册表现已接入主 Pipeline。直线 locomotion、普通静态姿态，以及双前腿或双后腿静态支撑已有确定性数值实现；平衡路径生成质心迁移、接触时序、基座姿态、足端轨迹，并执行浮动基座 IK、接触力优化和逆动力学。单腿、转向、跳跃、特技、操作及需要角动量控制的双腿支撑行走仍会在缺少专用后端时返回 `INCONCLUSIVE`。规划器输出成功只表示“可进入下一求解阶段”，绝不等价于 `PHYSICS_VALIDATED`。

## 当前真实验证路径

```text
TaskIntentSpec + compact_manifest
    -> 本地机器人/环境能力检查
    -> MotionType 分类 + MotionPrototype
       ├─ STATIC_POSE
       │   -> RobotMotionTarget -> Pinocchio IK -> 静态几何/关节检查 -> IsaacGym static rollout
       ├─ LOCOMOTION
       │   -> GaitPattern + FootTrajectory + VelocityTrajectory
       │   -> 每策略步足端采样 -> Pinocchio IK -> joint reference -> IsaacGym rollout
       ├─ BALANCE 双前腿/双后腿静态支撑
       │   -> CoM/接触/基座/足端轨迹 -> 浮动基座 IK -> 接触力优化 -> 逆动力学
       │   -> Unitree Isaac Gym 短时关节参考 rollout；真实通过才报告 STATIC_PHYSICS_VALIDATED
       ├─ BALANCE 单腿/动态支撑行走/倒立
       │   -> 缺少专用角动量或接触恢复后端时 INCONCLUSIVE / HUMAN_REVIEW
       ├─ JUMP / ACROBATIC
       │   -> PRELOAD/TAKEOFF/FLIGHT/ROTATION/LANDING/RECOVERY 阶段骨架
       │   -> INCONCLUSIVE / HUMAN_REVIEW（不调用 locomotion 验证器）
       └─ MANIPULATION / UNKNOWN / 未覆盖类型
           -> 操作类能力清单 + 笛卡尔目标 Schema；缺少专用验证器时 CONDITIONAL
       -> 同一 Unitree task_registry Go2 PPO 环境 / asset / PD / action_scale / PhysX
    -> CompleteFeasibilityReport + events.jsonl
```

Go2 的真实物理检查不加载从 URDF 派生的 MJCF。`RobotModelLoader.load_robot_model(..., require_mjcf=False)` 只读取真实 URDF 和机器人元数据；仿真环境通过 `legged_gym.envs.task_registry` 构造。Validator 绝不调用 `task_registry.make_alg_runner()`，不会创建 PPO 算法、加载/训练策略或写训练 checkpoint。

为降低预检的随机性，环境保持 Unitree 机器人、控制器、PhysX 和 asset 参数，只将并行环境数设为 1、关闭观察噪声、摩擦随机化、随机推力、命令课程和地形课程。初始 DOF/base 状态设为 PPO 配置的默认姿态。正式 PPO 训练配置不会被改写。

## 证据等级和安全门

| 状态/等级 | 含义 | 协调器处理 |
| --- | --- | --- |
| `STATIC_PHYSICS_VALIDATED` | 真实 Pinocchio IK 与 Isaac Gym 静态姿态 rollout 通过 | 只放行报告覆盖的静态姿态；不表示动态任务可行 |
| `DYNAMIC_PHYSICS_VALIDATED` | Unitree Isaac Gym 动态 rollout 完整，速度跟踪、稳定性、足端接触/滑移、关节与力矩检查全部通过 | 只放行报告覆盖的动作类别、速度和时长；不是已训练策略的成功保证 |
| `TRAINING_READY` | 动态物理预检通过，并且存在奖励配置与确定性评估指标 | 训练配置就绪标记；不是训练结果或策略验收 |
| `PHYSICS_FAILED` | 本次真实候选检查检测到不可达、控制跟踪或约束违规 | 交独立准入策略判断；不证明 PPO 无法学习 |
| `CAPABILITY_SUPPORTED` | 机器人/接口具备尝试能力，但没有完整物理验证证据 | 不放行 |
| `CONDITIONAL` | 后端不可用、原型回退、覆盖不足或任务有歧义 | 根据能力、环境健康与目标清晰度决定复核或预算探索 |
| `MODEL_UNAVAILABLE` | 当前缺模型或后端，证据不足以判断动作是否可行 | 进入 `HUMAN_REVIEW` |
| `UNSUPPORTED` | 能力清单有明确证据表明机器人缺少必需硬件/命令能力 | 进入 `FAILED` |
| `MOCK_VALIDATED` | 仅测试替身的流程结果 | 永不作为真实物理通过 |

真实 Go2 报告的 `backend` 为 `isaacgym`。只有完整真实 rollout 才能给出 `STATIC_PHYSICS_VALIDATED`、`DYNAMIC_PHYSICS_VALIDATED` 或 `PHYSICS_FAILED`。Isaac Gym 未安装、资产不一致、环境初始化异常、action 无法映射等属于证据不足，报告 `CAPABILITY_ONLY`/`UNAVAILABLE`；Mock 后端固定为 `MOCK_VALIDATED`，不能冒称物理结论。

生产协调器由 `TrainingAdmissionPolicy` 决定能否进入 Reward Designer。strict 模式要求目标真实物理通过；默认预算探索模式允许能力及环境就绪的失败/未知候选在受限预算下尝试学习。Mock 不能作为生产证据。Go2 动态后端当前仅覆盖前进/后退 LOCOMOTION；转向、跳跃、特技和操控任务不复用 trot validator，也不因探索准入获得物理通过等级。

统一报告新增 `action_type`、`feasibility_level`（`LEVEL_0_LANGUAGE` 至 `LEVEL_6_OPTIONAL_RL_PROBE`）、`confidence`、逐阶段 `stage_reports`、`kinematic_report`、`static_dynamics_report`、`trajectory_report`、`physics_report`、证据、限制和 `recommended_next_step`。`confidence` 表示证据覆盖度，不是 PPO 成功概率；Mock 最多到能力层，不能升为真实物理层。

静态计算器位于 `feasibility/validation/`：支撑多边形使用质心投影与足端目标点做确定性凸包检查；CoM 只在 Pinocchio URDF 惯性和 IK 关节解可用时计算。固定根 `rnea(q, 0, 0)` 重力补偿与 URDF effort 比较只是必要条件诊断，因缺接触力分配仍为 `CONDITIONAL`；摩擦/自碰撞缺可信接触数据时保持 `UNKNOWN`。动态验证器已把每步 Pinocchio IK 参考交给 `TrajectoryValidator` 计算 `q/qdot/qddot` 并检查模型中可用的限位；缺失的加速度/扭矩参考明确作为未知，不会伪造完整轨迹动力学通过。

`FootTrajectory -> Pinocchio IK -> joint reference -> Isaac Gym` 执行链路已接通。完整目标 rollout 通过才增加物理通过证据；奖励配置、确定性指标和准入预算另外写入 `training_readiness.json`。预算探索可训练，但不会冒称动态物理通过或已学会目标动作。

## 与 PPO 环境的一致性

- 训练工程路径由设置中的 `training_root` 提供；Go2 PPO 注册任务从 `unitree_rl_gym/legged_gym/envs/go2` 加载，不硬编码开发机绝对路径。
- Go2 asset 是 `resources/robots/go2/urdf/go2.urdf`。运行时验证注册配置解析出的 URDF 与 `RobotModelLoader` 的 URDF 相同。
- 运行时验证 Unitree `default_joint_angles` 与 RobotModel 配置相同，并按 `target_q = default_dof_pos + action_scale * action` 反算规范化 action。缺 joint 或 action 超出 PPO 的 clip 区间时拒绝 rollout，不静默裁剪动作。
- rollout 使用 `env.step(actions)`，由 Unitree 环境自己的 torque computation、decimation、PhysX simulate 和终止检查执行。检查 base 高度/倾斜、关节限位、PD 稳态跟踪误差、力矩限值及足端/禁止接触；同时记录全程最大跟踪误差作为瞬态诊断指标，并保存控制器、action scale、decimation、步数和随机化设置。
- 动态 rollout 将 `env.commands` 设为当前阶段的机身速度目标；每个 Unitree policy step 由 `FootTrajectoryGenerator` 采样四足笛卡尔目标，以真实 Pinocchio 多脚端 IK 求解关节参考，上一时刻解作为 warm start，再按 PPO 默认姿态与 `action_scale` 编译为位置动作。Mock/test double 不走该真实参考路径，且不能产生物理通过。
- 动态指标包括 rollout 完整性/步数、base 最低/最高高度及高度变化、roll/pitch/yaw 和姿态误差、速度误差、四足接触率/接触均衡度/累计离地时间/最大接触滑移、关节位置/速度越限、最大/RMS torque 和 torque 饱和占比；参考轨迹在仿真前还会检查 PPO joint position、有限差分速度和 action clip。
- IK-derived 位置参考仍是开环运动原语，不是 policy，也不是根据 PPO 权重运行出来的动作；不能作为训练数据或部署控制器。通过只说明该参考在本次短时配置下通过阈值，不代表机器人已学会任务。
- 当前 Unitree Go2 配置把 self collision 设为禁用，因此报告明确记录 `self_collision_enabled_by_ppo_config=false`，不能宣称已验收自碰撞。

## 动作覆盖边界

静态动作使用 Pinocchio 从 Go2 URDF 对脚端做 IK；LLM 不生成关节角。动态 locomotion 由 `DynamicMotionPrototypeGenerator` 生成准备/移动/停止速度阶段和参数化步态，默认原型为 `TROT / 2 Hz / duty_factor=0.5`，相位为 `FR+RL` 与 `FL+RR` 对角交替。意图约束可覆盖 `gait_type`、`gait_frequency`、`duty_factor`、`gait_phase_offsets`、`step_length` 和 `swing_height`；倒退任务会反转局部足端 x 扫掠方向。Isaac Gym 验证器逐 policy step 采样足端目标、调用多脚端 Pinocchio IK、热启动求解并映射到 PPO position action。原型 Schema 不包含 joint trajectory、torque 或 policy 字段。

1 m/s 等目标是否可行不由 LLM 判断：验证器先核对 Unitree `lin_vel_x` 命令范围，再根据完整真实 rollout 的平均速度误差与安全指标判定。默认 rollout 上限 5 秒/250 个策略步；超过预算拒绝截断后冒充完整验证。生成参考前检查关节位置限位、有限差分速度限位和 action clip；仿真中再检查跟踪误差、跌倒、接触/滑移、关节限位/速度及力矩饱和。动态 rollout 完整且全部通过时才标为 `DYNAMIC_PHYSICS_VALIDATED`；真实已完成但违规则为 `PHYSICS_FAILED`。Pinocchio/IK/reference 编译失败属于 `CAPABILITY_ONLY/CONDITIONAL`，不会伪称物理失败或通过。没有明确 gait、速度或足端轨迹的 locomotion 会在创建仿真前拒绝。转向当前尚无对应动态探针，跳跃需要单独验证离地/落地阶段，不能借用 locomotion 结论。

此动态验证仍是短时开环的 IK-derived joint position reference：它证明的是该运动原语在当前 Unitree Isaac Gym 模型、PD 控制和 rollout 时长下的物理响应，不证明闭环稳定控制能力，也不等于 PPO 已学会目标动作。真实物理等级要求同一批 `FootTrajectory -> Pinocchio IK -> JointReference -> Isaac Gym` 样本完整执行且全部确定性阈值通过；Mock 只能回报 `MOCK_VALIDATED`。

跳跃与特技目前仅有阶段骨架，没有起跳/落地数值目标、旋转轨迹或专用物理验证器。双前腿或双后腿静态支撑已使用自由浮动 Pinocchio 模型、质心目标、足端目标、接触力摩擦锥优化与 RNEA 逆动力学；数值就绪后才交给 Unitree Isaac Gym。单腿仍使用多候选静态必要条件检查；双腿支撑行走若单足阶段的接触力或基座动力学残差超限，会明确保持 `INCONCLUSIVE`。倒立仍需专用接触恢复与轨迹优化。操作类提供封闭的末端笛卡尔目标 Schema，但尚未接入末端 IK、碰撞、工作空间和物体动力学验证。

## 可视化查看验证过程

动态前进/后退预检可以使用与正式验证相同的 Go2 URDF、Unitree 环境、足端轨迹、Pinocchio IK 和位置控制链路打开 Isaac Gym Viewer：

```bash
conda activate rl_agent
cd /path/to/parameter_adjustment_agent/rl_agent
python -m rl_training_agent feasibility-view \
  --task "Go2 倒退走 0.3m/s 5秒" \
  --robot go2 \
  --max-seconds 5
```

Viewer 按仿真时间同步播放，按 `Esc` 或关闭窗口可提前结束。终端会先打印运动原型，窗口结束后打印速度跟踪、姿态、足端接触/滑移、关节和力矩报告。该模式只是把同一次开环物理预检显示出来，不加载 PPO policy、不训练模型，也不改变通过阈值。当前 Viewer 入口仅覆盖 Go2 前进/后退 `LOCOMOTION`；站立、单腿、跳跃、翻转和机械臂任务不会被错误转交给该动态 Viewer。

## 配置和报告

- 机器人元数据：[`config/robot_models.yaml`](../../config/robot_models.yaml)，资产路径相对训练工程。
- Unitree 仿真参数和 Go2 控制器：`unitree_rl_gym/legged_gym/envs/go2/go2_config.py` 及其基类配置。
- 实现：静态 `rl_training_agent/feasibility/simulation/isaacgym_validator.py`；动态 `rl_training_agent/feasibility/simulation/isaacgym_dynamic_validator.py`；轨迹生成 `rl_training_agent/feasibility/motion_prototype/dynamic_generator.py`。
- 新增的确定性检查：`rl_training_agent/feasibility/validation/static_validator.py`、`torque_validator.py`、`friction_validator.py`、`trajectory_validator.py`；它们不会在缺失模型输入时合成数据。
- 每次任务报告：`experiments/<task_id>/feasibility_report.json`。
- 阶段事件：`experiments/<task_id>/events.jsonl`。

MuJoCo 转换器仍是可单独调用的工具/兼容测试适配器，但 Go2 生产 Feasibility Pipeline 不使用转换产物作为真实动力学证据。

## 本工作区验证记录（2026-09-27）

- 已通过当前 Conda `rl_agent` 环境读取 Go2 URDF，Pinocchio 默认姿态 IK 收敛，残差为 0。
- 已实际创建 `task_registry` 中的 Go2 Unitree Isaac Gym 环境（GPU PhysX，单 env、headless），没有创建 PPO runner。
- 默认站姿完整 rollout `1.0 秒`、50 个策略步（每步 `0.02 秒`）；此前记录的结论现命名为 `backend=isaacgym`、`validation_level=STATIC_PHYSICS_VALIDATED`。最低 base 高度约 `0.214 m`，高于当前跌倒阈值 `0.21 m`。
- 动态接口使用同一 GPU PhysX 环境实测 `0.2 秒 / 10 步 / -0.3 m/s`：rollout 完整，无跌倒、关节位置/速度越限，但最大足端滑移 `0.324 m/s` 超过 `0.25 m/s` 阈值，因此真实结果是 `PHYSICS_FAILED`。尚未执行 5 秒完整动态样例；不能把该短样例写成 `DYNAMIC_PHYSICS_VALIDATED`。
- 该结果只验证了默认静态站姿在真实 Unitree Isaac Gym 环境中的短时物理 sanity check；没有训练或运行策略，不能据此称已学会站立，更不代表倒退/行走/跳跃通过。
- 动态探针的自动化测试使用 Mock environment 只验证张量接口、指标与隔离等级；除非另有注明，不把测试替身结果描述成 GPU PhysX 动态通过。

## 参数化步态历史运行记录（固定开环波形版，2026-09-28）

- `DynamicMotionPrototype` 已加入 `VelocityTrajectory` 与 `GaitPattern`；默认 Go2 trot 为 `2 Hz / duty_factor=0.5`，FR/RL 与 FL/RR 交替。`WALK`、`PACE`、`BOUND` 使用不同的默认归一化相位预设；constraints 可以覆盖 gait 类型、频率、占空比和相位。Isaac Gym 开环动作根据频率、占空比、相位及速度生成，不改变 PPO 算法。
- 在真实 GPU PhysX 上完成了一次 5 秒、250 策略步的 `Go2 倒退 0.3 m/s` 预检。rollout 完整且没有触发跌倒/关节越限，但真实机身平均速度误差为 `0.267 m/s`（低速任务自适应阈值 `0.075 m/s`），最大足端滑移为 `2.102 m/s`（阈值 `0.25 m/s`），故返回 `PHYSICS_FAILED`；并且部分足端仍未出现足够离地，不能宣称这套开环动作是可行步态。
- 此次真实失败表明参数化原型和验收路径已接通，**不表示默认探针已调到通过，更不表示目标动作无法由 PPO 学成**。默认静态站姿流程未改；训练策略仍需经 PPO 和独立评估验证。

## 多层级更新验证记录（2026-09-28）

- 全量自动测试收集 217 项并全部通过；其中覆盖通用约束注入防护、规划器分派、全身求解缺口门控、分类、单腿/前后腿候选搜索、支撑凸包、joint limits、torque/friction 输入校验、真实 IK 参考轨迹差分、原生 worker 崩溃隔离、Mock 隔离，以及真实 Go2 URDF/Pinocchio 静态重力估算。
- 真实 Go2 静态站姿：Pinocchio IK 成功，Pinocchio CoM 投影位于四足支撑凸包内（几何边界裕度约 `0.172 m`）；固定根 `rnea(q,0,0)` 重力力矩与 URDF effort 读取成功，但子项仍标为 `CONDITIONAL`，因为没有脚底接触约束的逆动力学/力分配。真实 Isaac Gym `1 秒 / 50 步` rollout 完成且无跌倒、关节限位、执行器力矩违规；由于使用本地规则 fallback 原型，外层 Feasibility 保持 `CONDITIONAL`，不自动放行奖励设计。
- 历史基线（当前 FootTrajectory→IK 接入之前）：真实 Go2 倒退 `0.3 m/s` 的 `0.2 秒 / 10 步` 固定关节波形探针返回 `PHYSICS_FAILED`，违规为 `foot_slip_exceeded`、`one_or_more_feet_never_contacted`、`velocity_tracking_error_exceeded`。状态机将真实失败送入 `HUMAN_REVIEW`；这不是当前 IK-reference 版本的运行结果，也不能据此判定 PPO 不可能学会倒退走。
- 单腿与前/后腿支撑已运行离线多候选 Pinocchio IK/静态必要条件检查，但因接触力、摩擦、浮动基座及动态平衡证据不完整仍返回 `CONDITIONAL/INCONCLUSIVE`；倒立不执行不适用的固定基座 IK，跳跃/空翻只报告阶段骨架，均未启动 PPO。

测试命令：

```bash
conda activate rl_agent
cd /path/to/parameter_adjustment_agent/rl_agent
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q rl_training_agent/feasibility/tests
```

## 双腿支撑平衡规划更新（2026-09-29）

新增 `feasibility/balance/`，用于双前腿或双后腿支撑动作的训练前低成本规划：

1. `MotionConstraintSpec` 确定支撑腿、目标时长与是否要求行走；
2. 本地规划器按 `prepare -> com_transfer -> unload_and_lift -> support_walk/balance_hold` 生成质心、接触、基座和足端时间序列；
3. Pinocchio 自由浮动模型联合求解支撑足、抬起足高度、质心和基座姿态，LLM 不提供关节角；
4. SciPy SLSQP 按点足摩擦锥分配接触力；
5. Pinocchio RNEA 减去 `Jᵀf` 后检查浮动基座动力学残差和 URDF effort；
6. 只有数值计划 `READY_FOR_PHYSICS` 才会转换成 PPO position-control 关节参考并送入现有 Unitree Isaac Gym 环境。

当前真实 Go2 URDF 离线测试中，“两只前腿静态站立 3 秒”的六项数值规划达到 `READY_FOR_PHYSICS`；这仍需实际 Isaac Gym rollout 才能升级为 `STATIC_PHYSICS_VALIDATED`。“两只前腿站立前走”在交替单前足阶段触发接触力 QP 和浮动基座动力学残差违规，因此保持 `INCONCLUSIVE`。该结论说明当前原语缺少角动量/动态单足支撑控制，不表示 PPO 永远无法学习此动作。

Mock adapter 即使返回成功也只保留 `MOCK_VALIDATED`。能力检查已同时修正：`forbidden_behaviors` 只作为安全约束，不再参与前腿/后腿目标分类。

当前全量自动测试共收集 223 项并全部通过；`compileall` 与 `git diff --check` 通过。

真实 GPU PhysX 端到端复核（2026-09-29）：双前腿静态支撑的数值计划为 `READY_FOR_PHYSICS`，随后通过 `backend=isaacgym` 进入 Unitree PPO 环境；rollout 在约 `0.38 秒 / 19 步` 时检测到 `actuator_settled_tracking_error_exceeded`，最大稳态关节跟踪误差约 `0.800 rad`，力矩限值比约 `0.484`，因此真实结论为 `PHYSICS_FAILED`。该结果证明接线和失败门控生效，同时说明当前轨迹/PD 组合不能稳定执行目标；不能把离线数值规划就绪写成动作可行。
