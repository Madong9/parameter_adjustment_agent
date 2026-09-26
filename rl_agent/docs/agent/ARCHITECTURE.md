# 系统架构

总览图：[`images/rl-agent-architecture-v2.png`](images/rl-agent-architecture-v2.png)。可编辑源图：[`images/rl-agent-architecture-v2.mmd`](images/rl-agent-architecture-v2.mmd)。图中以自研确定性状态机编排按职责分工的模型 Agent；没有依赖 LangGraph、CrewAI 等编排框架。更完整的角色、Schema、状态机、运行命令和路线图见 [`AGENT_COMPLETE_GUIDE.md`](AGENT_COMPLETE_GUIDE.md)。

命令层创建 `Settings`、`MultiAgentProvider`、`TrainingKnowledgeBase`、`LongTermMemoryStore` 和 `TrainingOrchestrator`。真实模式下，角色路由将任务理解和视觉评价交给 `FallbackProvider(ChatGPT 聊天模式 -> 豆包对话模式)`，将奖励设计和训练诊断交给百炼 GLM；离线演练使用确定性 Mock，但经过相同的上下文、提示词、审查、训练和记忆产物链。

编排器首先运行 `EnvironmentInspector`，然后要求任务理解 Agent 把上位机动作输入转换为 `TaskIntentSpec`。`ContextBuilder` 合并精简环境能力、当前机器人、RAG 和长期记忆，`RewardPromptCompiler` 使用固定版本模板产生带哈希的提示词。奖励设计结果必须通过 `RewardReviewAgent` 和 `RewardCompiler`，模型无权直接写训练配置。

RAG 使用无需外部服务的中文 BM25 索引。索引范围由白名单控制，默认只包含奖励设计、视觉评估、安全文档和历史实验的任务、奖励、诊断、修订与最终摘要。当前任务会从历史实验命中中排除，机器人一致和已完成实验会获得有限加权。检索内容不能覆盖当前环境能力清单、安全约束或确定性验收结果。

`UnitreeProjectAdapter` 是唯一的命令构造器，只为受限真实训练包装器和 rollout 包装器生成参数数组。`ProcessManager` 使用 `shell=False`、独立进程组、有限等待、PID/日志/退出码文件和进程组终止机制。系统不提供通用 shell 执行接口。

证据处理链如下：

```text
自然语言目标 + 环境能力清单
  -> GPT/OpenCLI 聊天模式输出 TaskIntentSpec（故障时由豆包对话接管）
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
  -> MemoryCuratorAgent 晋升联合验收成功或多种子明确失败模式
  -> SemanticMemoryConsolidatorAgent 用独立证据升级通用规律
  -> ProceduralMemoryAgent 固化提示词、Schema 和校验策略
  -> 完成、预算耗尽或真实阻塞
```

上述链路由 `TrainingOrchestrator` 循环执行。诊断输入不仅包含视觉和 PPO 结果，还包含当前任务、奖励计划、注册奖励白名单、可计算指标、逐项奖励均值、历史轮次和剩余预算。`RewardPlanReviser` 只接受结构化的 add/remove/update 和课程修改；修改后重新触发 Pydantic、奖励符号、权重上限、任务指标覆盖和编译校验。`continue` 可以保持奖励版本不变直接追加训练，`restart` 从随机初始化开始，`continue_from_parent` 可回退到父 checkpoint。

确定性评估器是完成状态的最终裁决者。只有任务指标、硬安全约束和视觉对齐在同一轮全部通过时才进入 `COMPLETED`；普通未达标继续自动闭环，Provider 不可用、不可计算需求、安全修订失败、真实歧义或预算耗尽才进入 `HUMAN_REVIEW`。

`PersistentStateMachine` 会原子写入每次状态变化。`ExperimentStore` 校验路径，并使用 `fcntl` 文件锁串行化写入。重启、回滚或奖励修订不会删除父实验和原 checkpoint。

生产角色由 `MultiAgentProvider` 路由：`opencli-doubao` 用 `FallbackProvider` 包装 `OpenCLIChatGPTWebProvider` 与 `OpenCLIDoubaoWebProvider`，承担任务理解和多模态视觉评价；ChatGPT 每次新会话都强制使用“聊天”，豆包强制使用“对话”。`BailianGLMProvider` 承担奖励设计和训练诊断。`MockLLMReasoningProvider` 只有在显式选择时才可使用，主要服务于测试和 dry-run。兼容命令仍可显式选择单一 `opencli` 或 `doubao` Provider。

独立视觉评论不会接收 RAG 奖励经验，只读取任务规格和同步视觉证据。这样可以避免视觉裁判因为知道奖励设计而产生确认偏差。RAG 失效时采用失败开放策略：记录错误后继续使用当前任务证据，不会因为索引损坏阻断训练主链路。
