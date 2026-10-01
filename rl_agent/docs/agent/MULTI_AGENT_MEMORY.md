# 多 Agent 协作与长期记忆

## 角色分工

生产模式使用 `MultiAgentProvider`，但最终控制权始终属于本地 `TrainingOrchestrator`：

| 角色 | 实现 | 输入 | 输出 |
| --- | --- | --- | --- |
| 协调器 | 本地确定性状态机 | 上位机任务和当前实验状态 | 状态转换、预算、产物和恢复点 |
| 任务理解 Agent | GPT / OpenCLI | 用户动作、机器人 | `TaskIntentSpec` |
| 动作可行性预检 | 本地规则 + GPT/Mock 语义阶段 | TaskIntent + 紧凑环境清单 + 机器人模型 | `CompleteFeasibilityReport` |
| 上下文构建器 | 本地代码 | 环境能力、机器人、RAG、长期记忆 | `context_snapshot.json` |
| 提示词编译器 | 本地固定模板 | TaskIntent 与上下文 | 带版本和哈希的提示词 |
| 奖励设计 Agent | 百炼 GLM | 编译后的提示词 | `TaskRewardBundle` |
| 奖励审查 Agent | 本地确定性代码 | TaskIntent、TaskSpec、候选与注册表 | `RewardReviewReport` |
| 数值评估器 | 本地确定性代码 | 轨迹、接触力、PPO 和安全指标 | `EvaluationResult` 数值部分 |
| 视觉评估 Agent | GPT / OpenCLI | TaskSpec 和同步图像证据 | `VisualBehaviorReport` |
| 诊断与修订 Agent | 百炼 GLM | 视觉、数值、预算、RAG 和记忆 | `TrainingDiagnosis` |
| Reward Experience Agent | 诊断 Provider（当前默认为百炼 GLM）+ 本地代码 | 证据包、奖励差异、数值和视觉评价 | 带证据引用的经验叙述 |
| Memory Validator | 本地确定性代码 | 经验、原始配置和证据文件 | 通过/拒绝及可追溯原因 |
| 情景记忆整理器 | 本地确定性代码 | 通过 Validator 的 Reward Experience | 情景记忆或多种子门控拒绝 |
| 语义整理 Agent | 本地确定性代码 | 多条独立情景证据 | 候选或活跃语义规律 |
| 程序记忆 Agent | 本地确定性代码 | 提示词、Schema、状态和校验策略 | 可复现程序快照 |

Agent 之间不共享自由聊天记录，而是通过版本化 JSON 和文件产物交接。这样任何阶段失败后都能确定失败位置，也能从既有 checkpoint 和证据恢复。

## 动作输入链路

上位机只提交用户原文、机器人和运行模式。任务理解 Agent 不允许生成奖励或命令，只负责提取动作标准名、目标速度、地形、必需行为、禁止行为、假设、歧义和检索关键词。

`Task Feasibility Pipeline` 紧接在意图保存后运行，早于 RAG、奖励模型和 PPO。它复用 `EnvironmentInspector` 与 `ContextBuilder.compact_manifest()`，由本地检查机器人/关节/执行器、观测、命令和成功/失败指标；动作阶段 Provider 只能产出语义 `MotionPrototype`。Go2 生产路径用 Pinocchio 求解 URDF IK，再加载 Unitree task_registry 中与 PPO 相同的 Isaac Gym 环境做不训练策略的短时物理 sanity check，不依赖转换 MJCF。完整门控和动作覆盖边界见 [`TASK_FEASIBILITY.md`](TASK_FEASIBILITY.md)。

本地协调器会强制把 `TaskIntentSpec.original_instruction` 和 `robot` 恢复为上位机原始值，防止模型改写任务身份。随后 `ContextBuilder` 移除源码绝对路径和重复字段，将以下内容分区保存：

- 当前环境注册奖励、观测量、终止条件和验收指标；
- 当前机器人和命令空间；
- RAG 项目文档与原始实验命中；
- 只包含已验证实验的长期记忆命中；
- 明确的只读证据和提示注入边界。

`RewardPromptCompiler` 使用仓库内 `reward_design_agent.md`，生成 `reward-design-v2` 提示词和 SHA-256。GLM 无法改变模板版本，也不能把历史检索文字当成系统指令。

## 百炼 GLM 配置

模型和端点配置位于 `config/bailian.yaml`：

```yaml
model: glm-4.7
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
api_key_env: DASHSCOPE_API_KEY
timeout_seconds: 300
max_retries: 2
enable_thinking: false
response_format_json: true
```

API Key 只能通过环境变量传入：

```bash
export DASHSCOPE_API_KEY="你的百炼 API Key"
export DASHSCOPE_BASE_URL="https://<业务空间ID>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
```

如果控制台实际开通的模型不是 `glm-4.7`，使用 `BAILIAN_MODEL` 覆盖，不要改代码：

```bash
export BAILIAN_MODEL="控制台返回的模型ID"
```

Provider 使用 `/chat/completions`、`response_format=json_object` 和 `enable_thinking=false` 请求严格 JSON。首次 Schema 校验失败时会携带校验错误和原回复执行一次结构化修复。请求审计只保存模型、消息和参数，不保存 Authorization 请求头。

## 奖励审查门控

奖励审查发生在 GLM 生成之后、配置编译和 PPO 训练之前。它检查：

- 每个候选是否只使用注册奖励；
- 是否覆盖全部必选成功指标；
- 后腿或前腿支撑动作是否使用对应姿态门控奖励；
- 水平姿态奖励是否与目标俯仰角冲突；
- 是否明确记录奖励投机风险。

审查按候选独立给出通过/拒绝及原因，并写入 `reward_review.json`。不合格候选会被隔离，不创建训练目录、不消耗训练预算；同批次通过审查的候选仍会继续编译和训练。只有当所有候选都被拒绝（或没有记录任何奖励投机风险）时，才写入 `blocking_report.json` 并进入 `HUMAN_REVIEW`。审查 Agent 不直接修改奖励方案。

## 四层记忆生命周期

RAG 和记忆不是同一个数据层。RAG 可以读取项目文档和白名单原始实验；记忆模块负责筛选、写入、升级、检索和遗忘。四层职责如下：

| 层级 | 保存位置 | 内容 | 能否直接指导模型 |
| --- | --- | --- | --- |
| 工作记忆 | `experiments/<task-id>/memory/working_memory.json` | 当前状态、轮次、奖励版本、预算、最新评估和诊断 | 仅供当前任务恢复和上位机展示 |
| 情景记忆 | `artifacts/memory/records/` | 某机器人执行某动作时的奖励、差异、checkpoint、指标、视觉结论和证据 | 仅 `active` 记录可以检索 |
| 语义记忆 | `artifacts/memory/semantic/` | 跨独立任务验证的通用规律或明确失败模式 | 只有达到证据阈值的 `active` 规律可以检索 |
| 程序记忆 | `artifacts/memory/procedural/` | 提示词版本、Schema 哈希、校验规则、状态机和诊断策略 | 不作为动作经验，只用于复现和审计 |

情景记忆晋升流程为：

```text
真实实验结束
  -> 排除 dry_run、HUMAN_REVIEW、Provider 故障和格式错误
  -> 至少两个不同随机种子的结果一致
  -> 成功：任务指标 + 硬安全约束 + 视觉对齐全部通过
     或失败：FAILED 且存在确定性 violations
  -> RewardExperienceEvidenceBuilder 只从任务文件、奖励配置、manifest、评估文件建证据包
  -> ExperienceEligibilityChecker：SUCCESS / VERIFIED_FAILURE / INCONCLUSIVE
  -> 仅 SUCCESS 或 VERIFIED_FAILURE 调用 Reward Experience Agent 总结文字结论
  -> RewardExperienceValidator 复核文件哈希、配置差异、指标来源、任务身份、证据 ID 和因果措辞
  -> MemoryCuratorAgent 再执行多随机种子门控并写入情景记忆
  -> SemanticMemoryConsolidatorAgent 聚合同类独立任务证据
  -> 证据数达到 memory_min_semantic_support 后从 candidate 升级为 active
```

Reward Experience Agent 不接收整个训练目录或原始日志。奖励名称、初始/最终配置、reward diff 和评估指标由 Python 从真实 JSON 文件计算；模型只能返回 `observed_facts`、`hypotheses`、成功/失败模式和限制说明。观察事实必须带 evidence ID，假设必须使用“可能/推测”等不确定措辞。奖励改变和行为改变按奖励项显式配对，只能写“同期观察到”，不能宣称单次实验证明因果。

证据包、门控和归纳产物按所选 experiment ID 保存在 `experiments/<task-id>/memory/reward_experience/<experiment-id>/`：`evidence.json`、`training_result_snapshot.json`、`eligibility.json`、`experience.json` 和 `validation.json`。不合格任务也会保存证据与 `INCONCLUSIVE` 原因，但不会调用经验总结模型或进入长期记忆。Provider 故障、dry-run、PPO 中断、缺失文件和无明确失败行为证据均属于 `INCONCLUSIVE`。

通过验证的完整 `RewardExperience` 会嵌入 `LongTermMemoryRecord.reward_experience`，与奖励项、指标、视觉结论及证据来源一起写入 `artifacts/memory/records/`。长期记忆检索的 BM25 文本和上下文压缩结果包含奖励差异、观察事实、假设、失败模式及适用范围，所以后续奖励设计 Agent 可检索；RAG 原始实验索引仍是独立的数据源。

每条情景记忆至少保存机器人型号、动作标准名称、仿真平台、环境配置哈希、Git commit、奖励方案、修改差异、checkpoint、确定性指标、视觉结论、最终状态、证据来源、随机种子数、可信度和创建时间。经验证的 `RewardExperience` 进一步保存初始/最终奖励设计、每项改动、同期行为观察、假设、适用范围和失败风险。初始奖励版本从候选父子关系追溯，修订版本从真实 `reward_plan.json` 和 `revision_audit.json` 计算；模型不能覆盖这些字段。

`HUMAN_REVIEW`、Provider 超时、格式错误、预算耗尽但未得到明确确定性失败证据、单种子结果和 `dry_run` 只保留在实验目录，不会晋升为正确经验。明确失败模式使用 `verified_failure` 标识，不会伪装成成功经验。

遗忘采用“归档而非删除”：低于 `memory_min_confidence`、超过 `memory_max_age_days` 或超出 `memory_max_records` 容量的记录被标记为 `archived`，写明 `archive_reason`，原始 JSON 和证据仍保留。检索只读取 `active` 情景记忆与 `active` 语义规律，并排除当前任务，避免循环自证。

相关阈值全部位于 `config/agent.yaml`，路径仍使用相对路径：

```yaml
memory_enabled: true
memory_root: memory
memory_top_k: 4
memory_max_context_chars: 4000
memory_require_multi_seed: true
memory_min_semantic_support: 2
memory_max_records: 500
memory_max_age_days: 365
memory_min_confidence: 0.55
```

可以在 Agent 目录检查记忆状态和检索结果：

```bash
python -m rl_training_agent memory-stats
python -m rl_training_agent memory-query --query "Go2 后腿站立行走" --robot go2
```

## 运行前检查

```bash
conda activate rl_agent
python -m rl_training_agent doctor
python -m rl_training_agent opencli-test
```

`doctor` 只有在训练环境、CUDA、OpenCLI/ChatGPT 和百炼密钥都满足当前多 Agent 模式时才报告 `production_ready=true`。没有百炼密钥仍可使用上位机“离线演练”，但真实训练会在奖励设计阶段明确失败，不会自动回退到 Mock。
