# 奖励经验总结 Agent

Reward Experience Agent 在一次 RL 闭环结束后，把可验证的奖励与行为证据整理成可检索的情景经验。它不替代数值/视觉验收，也不训练策略。

## 所在流程

```text
Diagnosis / 终态
  -> RewardExperienceEvidenceBuilder：读取本次实验产物
  -> ExperienceEligibilityChecker：判定 SUCCESS / VERIFIED_FAILURE / INCONCLUSIVE
  -> Reward Experience Agent：仅归纳文字观察、假设和局部模式
  -> RewardExperienceValidator：复核来源、差异、指标、任务身份和措辞
  -> MemoryCuratorAgent：检查既有多随机种子及结果门控
  -> LongTermMemoryStore：写入 episodic memory
  -> SemanticMemoryConsolidatorAgent：收集独立实验并决定 candidate / active
```

状态机仍使用现有 `MEMORY_CURATING`，避免在训练状态图中增加模型内部步骤；各子阶段另以 JSONL 事件写入 `experiments/<task-id>/events.jsonl`。最终状态仍由数值评估、视觉评估和安全约束决定，经验总结失败不会改变训练结果。

## 谁负责什么

Python 负责读取文件、生成哈希、计算奖励差异、门控结局、校验引用并决定是否晋升。LLM 不收到训练目录或原始日志，也不能返回或覆盖奖励权重、最终奖励配置和指标数值。

经验文字 Provider 复用 `diagnosis_provider` 配置（当前项目默认为百炼 GLM）。如果选择 `opencli` / `opencli-doubao`，模型调用仍走现有普通聊天模式和主备路由。Provider Registry 会在启动时检查它是否声明 `reward_experience_summarization` 能力；API Key 沿用已配置 Provider 的密钥，不新增另一套密钥。

模型输出 Schema 是 `RewardExperienceNarrative`，仅包含：

- `observed_facts`：直接观察到的事实，必须有 evidence ID。
- `hypotheses`：可能原因，必须使用“可能、推测、假设”等不确定表达。
- `reward_change_observations`：按奖励名显式说明和行为变化同期出现的现象，引用数值或视觉评估文件；不能写成因果。
- `successful_patterns` / `failure_patterns`：局部实验模式，不声称跨机器人普遍成立。
- `limitations`。可信度不是由 LLM 自评，而是由本地代码按真实随机种子数计算并限制在 0.65 以下。

奖励配置和差异不由模型生成。`RewardExperienceAgent` 将受限文字与本地证据组合为最终 `RewardExperience`。

## 证据和资格门控

`RewardExperienceEvidenceBuilder` 从当前 task / selected experiment 文件读取信息：

| 内容 | 主要来源 |
| --- | --- |
| 机器人、目标动作 | `task_spec.json` |
| 初始/最终 reward terms | 最初父候选及最终候选的 `reward_plan.json` |
| 逐轮奖励差异、版本和修订顺序 | 完整父子候选链、各版本 `reward_plan.json`、`revision_audit.json` |
| PPO 终态、训练迭代、checkpoint | `manifest.json`、终态快照、最终 checkpoint |
| 任务指标与安全结论 | rollout 的 `evaluation.json`、`numeric_summary.json` |
| 视觉结论和行为风险 | rollout 的 `visual_report.json` |
| 代码和配置版本 | manifest 的 Git commit 与 config hash |

每个文件都生成内容 SHA-256 和稳定 evidence ID。Validator 会检查文件仍存在、hash 未改变，重新从实际奖励计划计算 reward diff，并逐项复核数值、视觉摘要、任务身份及 evidence ID。

资格状态定义：

- `SUCCESS`：真实非 dry-run PPO 已完成并有 checkpoint；最终状态为 `COMPLETED`；数值任务指标、硬安全约束和视觉验收全部通过；来源文件齐全。
- `VERIFIED_FAILURE`：真实 PPO 已完成；最终状态为 `FAILED`；存在明确失败的确定性指标，以及对应视觉失败/非预期行为或被数值文件复核的行为指标。
- `INCONCLUSIVE`：训练中断、dry-run、Provider/JSON 错误、人工审核、预算耗尽但没有可证实失败行为、缺失或哈希不一致的证据。只保留任务证据，不调用经验总结模型，不写长期记忆。

证据包保留 `reward_history` 父子链；每轮新增、删除和变化都会单独记录 `{reward_name, before, after, iteration, reward_version, evidence_ids}`，不会把多轮权重修改折叠成一个最终差值。初始奖励与最终奖励也并列保存，便于比较全程结果。行为变化是另一个独立叙述字段，并要求引用评估来源；例如“orientation 权重从 0.5 调整为 3.0；该策略在最终评估中 pitch 均值为 X”是同期观察，“提高 orientation 导致姿态稳定”则是未经证实的因果断言，会被拒绝。

## 输出与 Memory / RAG

按所选 experiment ID 保存：

```text
experiments/<task-id>/memory/reward_experience/<experiment-id>/
├── training_result_snapshot.json
├── evidence.json
├── eligibility.json
├── experience.json       # 仅模型归纳及 Validator 通过时存在
└── validation.json
```

有效 `RewardExperience`（含 evidence ID 到相对源文件与 SHA-256 的 `evidence_index`）被嵌入 `LongTermMemoryRecord.reward_experience`，写入 `artifacts/memory/records/`。情景记忆仍须通过原有 `memory_require_multi_seed` 门槛；记忆可信度按有效种子数本地计算并最高限制为 0.65，以免一次实验被当作普遍定律。

后续奖励设计阶段的 `LongTermMemoryStore.retrieve()` 会对奖励差异、观察、假设、成功/失败模式做 BM25 检索，并将完整记录或压缩摘要交给 `ContextBuilder`。RAG 原始实验索引与长期记忆是两个不同来源：RAG 保留可读资料，Memory 只检索通过资格和 Validator 门控的情景经验。当前 `MemoryCuratorAgent` 与 `SemanticMemoryConsolidatorAgent` 保持原样协作：语义规律要由至少 `memory_min_semantic_support` 条不同 task 记忆支持后才升级为 `active`。

可以查看当前任务产物：

```bash
TASK_ID=task-xxxxxxxxxx
EXPERIMENT_ID=run-xxxxxxxx-candidate-01-v01
find "experiments/$TASK_ID/memory/reward_experience" -maxdepth 3 -type f -print
cat "experiments/$TASK_ID/memory/reward_experience/$EXPERIMENT_ID/eligibility.json"
cat "experiments/$TASK_ID/memory/reward_experience/$EXPERIMENT_ID/validation.json"
```

## 测试范围与局限

`tests/test_reward_experience.py` 覆盖：成功差异（velocity `3 -> 1`、orientation `0.5 -> 3`）、有行为证据的确定性失败、训练中断归为 `INCONCLUSIVE`、伪造 reward diff 被拒绝。离线测试用测试 Provider，不调用外部 API，也不代表 Go2 的真实训练验证。

Validator 可以确定性验证“证据文件和引用是否存在、内容是否变化、数值和 reward diff 是否与文件一致、任务身份是否匹配、是否出现明显因果/泛化措辞”。自然语言蕴含校验不是形式化证明；所以每条叙述保留来源、适用范围、假设类别和可信度，未来应以多任务反例、专家复核和语义冲突治理继续约束。
