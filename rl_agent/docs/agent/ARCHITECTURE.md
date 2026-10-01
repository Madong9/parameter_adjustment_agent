# 系统架构

最新架构以可编辑源图 [`images/rl-agent-architecture-v2.mmd`](images/rl-agent-architecture-v2.mmd) 为准；[`images/rl-agent-architecture-v2.png`](images/rl-agent-architecture-v2.png) 是尚未包含训练准入分流的历史导出。图中以自研确定性状态机编排按职责分工的模型 Agent；没有依赖 LangGraph、CrewAI 等编排框架。更完整的角色、Schema、状态机、运行命令和路线图见 [`AGENT_COMPLETE_GUIDE.md`](AGENT_COMPLETE_GUIDE.md)。

命令层创建 `Settings`、`MultiAgentProvider`、`TrainingKnowledgeBase`、`LongTermMemoryStore` 和 `TrainingOrchestrator`。真实模式下，角色路由将任务理解和视觉评价交给 `FallbackProvider(ChatGPT 聊天模式 -> 豆包对话模式)`，将奖励设计和训练诊断交给百炼 GLM；离线演练使用确定性 Mock，但经过相同的上下文、提示词、审查、训练和记忆产物链。

编排器首先运行 `EnvironmentInspector`，然后要求任务理解 Agent 把上位机动作输入转换为 `TaskIntentSpec`。`MotionConstraintCompiler` 将语义编译为禁止 joint/torque/policy 的 `MotionConstraintSpec`，确定性注册表分别运行步态、质心、接触时序、基座姿态和末端规划器；`WholeBodyMotionSolver` 再核对 Pinocchio IK、浮动基座 IK、逆动力学、接触力和轨迹优化后端。只有动作专用求解链完整时才进入 Isaac Gym：静态姿态使用 IK、CoM/支撑几何、重力 RNEA 诊断和静态 rollout；Go2 直线行走逐步执行 `FootTrajectory → Pinocchio IK → JointReference → IsaacGym`。单腿/前后腿支撑会搜索有限候选但因接触动力学不足保持 `INCONCLUSIVE`；跳跃、特技、转向和操作会列出缺失求解器，绝不复用错误的 trot validator。完整真实 rollout 通过才报告物理验证，Mock 不得升级。之后 `ContextBuilder` 合并环境能力、机器人、RAG 和长期记忆，奖励设计仍必须通过审查与安全编译。

RAG 使用无需外部服务的中文 BM25 索引。索引范围由白名单控制，默认只包含奖励设计、视觉评估、安全文档和历史实验的任务、奖励、诊断、修订与最终摘要。当前任务会从历史实验命中中排除，机器人一致和已完成实验会获得有限加权。检索内容不能覆盖当前环境能力清单、安全约束或确定性验收结果。

`UnitreeProjectAdapter` 是唯一的命令构造器，只为受限真实训练包装器和 rollout 包装器生成参数数组。`ProcessManager` 使用 `shell=False`、独立进程组、有限等待、PID/日志/退出码文件和进程组终止机制。系统不提供通用 shell 执行接口。

物理验证和训练准入相互独立：`TrainingAdmissionPolicy` 保留 `PHYSICS_FAILED/INCONCLUSIVE`，默认在能力、真实模型和 Isaac Gym 运行证据完整时允许有限预算探索。没有目标 rollout 时，隔离 worker 执行默认姿态健康检查，只证明基础设施可用。`strict` 模式仍要求目标探针通过；Mock 不能放行真实训练。准入、环境健康、编译后就绪和冻结验收合同分别留档，恢复校验 run_id、验收指标及预算。阶段约束区分准备期四足接触、质心转移、卸载和目标支撑，避免把最终姿态条件错误地应用于整个运动过程。详见 [`TASK_FEASIBILITY.md`](TASK_FEASIBILITY.md)。

证据处理链如下：

```text
自然语言目标 + 环境能力清单
  -> GPT/OpenCLI 聊天模式输出 TaskIntentSpec（故障时由豆包对话接管）
  -> 动作约束、确定性规划和候选物理探针
  -> 独立训练准入：正常预算 / 有限探索 / 复核 / 能力拒绝
  -> RAG + 已晋升长期记忆检索
  -> 本地 ContextBuilder 与固定 PromptCompiler
  -> 百炼 GLM 输出 TaskRewardBundle
  -> 独立 RewardReviewAgent
  -> 本地奖励校验、编译与训练
  -> 轨迹 + 奖励 + 仿真器 MP4
  -> 基于轨迹的 EventDetector
  -> 干净接触图与标注接触图
  -> 纯视觉报告
  -> 确定性安全/任务评估
  -> RAG 检索相似失败与修订经验
  -> 百炼 GLM 融合后的 TrainingDiagnosis
  -> 受限 RewardPlanReviser
  -> 新奖励版本编译与多种子续训/重训
  -> 下一轮 rollout 与联合验收
  -> RewardExperienceEvidenceBuilder 从真实产物构建带哈希证据包
  -> ExperienceEligibilityChecker 判定 SUCCESS / VERIFIED_FAILURE / INCONCLUSIVE
  -> 合格任务由 Reward Experience Agent 归纳文字结论，RewardExperienceValidator 复核来源与差异
  -> MemoryCuratorAgent 再执行多随机种子情景记忆门控
  -> SemanticMemoryConsolidatorAgent 用独立证据升级通用规律
  -> ProceduralMemoryAgent 固化提示词、Schema 和校验策略
  -> 完成、预算耗尽或真实阻塞
```

上述链路由 `TrainingOrchestrator` 循环执行。诊断输入不仅包含视觉和 PPO 结果，还包含当前任务、奖励计划、注册奖励白名单、可计算指标、逐项奖励均值、历史轮次和剩余预算。`RewardPlanReviser` 只接受结构化的 add/remove/update 和课程修改；修改后重新触发 Pydantic、奖励符号、权重上限、任务指标覆盖和编译校验。`continue` 可以保持奖励版本不变直接追加训练，`restart` 从随机初始化开始，`continue_from_parent` 可回退到父 checkpoint。

确定性评估器是完成状态的最终裁决者。只有任务指标、硬安全约束和视觉对齐在同一轮全部通过时才进入 `COMPLETED`；普通未达标继续自动闭环，Provider 不可用、不可计算需求、安全修订失败、真实歧义或预算耗尽才进入 `HUMAN_REVIEW`。

`PersistentStateMachine` 会原子写入每次状态变化。`ExperimentStore` 校验路径，并使用 `fcntl` 文件锁串行化写入。重启、回滚或奖励修订不会删除父实验和原 checkpoint。

训练终态仍由本地数值、硬安全约束和视觉结果决定。终态之后的经验归纳只是记忆整理：LLM 只能写带 evidence ID 的事实、假设和局部模式；初始/最终奖励、reward diff、指标和结局均由 Python 从产物计算。Validator 会复核源文件 hash、真实 reward_plan、数值/视觉评价和任务身份；Provider 故障、训练中断或证据不足记为 `INCONCLUSIVE`，不进入长期 Memory。详细 Schema 和文件布局见 [`REWARD_EXPERIENCE.md`](REWARD_EXPERIENCE.md)。

生产角色由 `MultiAgentProvider` 路由：`opencli-doubao` 用 `FallbackProvider` 包装 `OpenCLIChatGPTWebProvider` 与 `OpenCLIDoubaoWebProvider`，承担任务理解和多模态视觉评价；ChatGPT 每次新会话都强制使用“聊天”，豆包强制使用“对话”。`BailianGLMProvider` 承担奖励设计和训练诊断。`MockLLMReasoningProvider` 只有在显式选择时才可使用，主要服务于测试和 dry-run。兼容命令仍可显式选择单一 `opencli` 或 `doubao` Provider。

独立视觉评论不会接收 RAG 奖励经验，只读取任务规格和同步视觉证据。这样可以避免视觉裁判因为知道奖励设计而产生确认偏差。RAG 失效时采用失败开放策略：记录错误后继续使用当前任务证据，不会因为索引损坏阻断训练主链路。
