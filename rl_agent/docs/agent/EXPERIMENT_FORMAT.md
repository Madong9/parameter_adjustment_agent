# 实验目录格式

```text
experiments/<task_id>/
├── task_request.txt             # 原始用户指令
├── task_spec.json               # 规范化任务规格
├── environment_manifest.json    # 本任务使用的环境能力清单
├── task_intent.json             # GPT 任务理解 Agent 输出的 TaskIntentSpec
├── feasibility_report.json      # PPO 前能力、动作原型、IK 和 MuJoCo 预检报告
├── context_snapshot.json        # 环境、机器人、RAG 与长期记忆上下文
├── design_request.json          # 任务设计请求
├── design_response.json         # 任务设计结构化回复
├── reward_review.json           # 训练前奖励完整性、冲突与投机风险审查
├── prompts/
│   ├── reward_design_prompt.md  # 固定模板编译后的百炼请求
│   └── reward_design_prompt.json # 模板版本与 SHA-256
├── rag/
│   └── design_retrieval.json    # 奖励设计阶段的 RAG 查询、命中与来源
├── memory/
│   ├── design_retrieval.json    # 奖励设计阶段的已验证长期记忆
│   ├── working_memory.json      # 当前状态、奖励版本、预算与最新评估
│   └── promotion.json           # 是否通过门控晋升长期记忆
├── state.json                   # 可恢复状态与历史
├── lineage.json                 # 实验谱系
├── task_normalization.json      # 自然语言语义修正及指标别名映射
├── loop_status.json             # 当前闭环轮次、奖励版本、决策和预算
├── loop_history.json            # 历次联合验收结果
├── events.jsonl                 # 状态转换与 Provider 延迟结构化事件
├── provider_records/
│   ├── opencli/                 # GPT/OpenCLI 请求、附件和原始回复
│   └── bailian/                 # 不含鉴权头的 GLM 请求和原始回复
├── summary.json                 # 任务摘要
├── report.md                    # 人类可读报告
├── candidates/<experiment_id>/
│   ├── manifest.json
│   ├── reward_plan.json
│   ├── config.yaml
│   ├── config.diff
│   ├── compile_metadata.json
│   ├── stdout.log
│   ├── stderr.log
│   ├── metrics/
│   ├── checkpoints/
│   ├── prompts/
│   ├── responses/
│   ├── revision_audit.json      # 修订候选才有：诊断、父实验和实际修改
│   └── rollouts/round_<轮次>/rollout_<编号>/
│       ├── front.mp4、side.mp4、overview.mp4
│       ├── trajectory.parquet、rewards.parquet
│       ├── metadata.json、events.json、numeric_summary.json、evaluation.json
│       ├── contact_sheet_clean.png、contact_sheet_annotated.png、contact_sheet_multiview.png
│       ├── behavior_evidence.json、visual_attachment_manifest.json
│       ├── visual_report_individual.json、visual_ensemble.json
│       ├── rag_diagnosis_context.json # 诊断阶段的 RAG 查询、命中与来源
│       ├── memory_diagnosis_context.json # 诊断阶段的长期记忆命中
│       ├── event_takeoff.png、event_landing.png
│       └── visual_prompt.txt、visual_raw_response.txt、visual_report.json、diagnosis.json、decision.json
└── final/                        # 最终 checkpoint、配置和代表性材料
    ├── checkpoint.pt
    ├── config.yaml
    ├── reward_plan.json
    └── contact_sheet_clean.png、contact_sheet_annotated.png
```

Manifest 会记录任务/实验/父实验 ID、Git commit、配置哈希、奖励版本、种子、机器人、完整参数数组、起止时间、iteration、checkpoint、Provider 状态、结果和失败原因。`lineage.json` 的边记录每一版候选的父子关系，最终选中节点会写入完成、失败或人工复核结果。状态与 JSON 均使用原子写入；同一自然语言任务再次运行会保留旧候选和 checkpoint，但清除会误导界面的旧终态摘要，并用新的 `run-<编号>` 候选目录隔离本次训练。

全局 RAG 索引位于 `artifacts/rag/index.json`，不放入单个任务目录。每次设计和诊断实际使用的片段则保存在上述任务产物中，因此报告审查时可以还原模型参考了哪些历史信息。索引内只保存文本片段和可移植相对来源名，不保存视频、checkpoint 或主机绝对路径。

四层记忆分别位于任务目录的 `memory/working_memory.json`、`artifacts/memory/records/`、`artifacts/memory/semantic/` 和 `artifacts/memory/procedural/`。联合验收通过的成功结果和具有多种子确定性违规证据的明确失败模式可以成为情景记忆；`HUMAN_REVIEW`、Provider 故障、格式错误、单种子结果和 `dry_run` 不会成为可信长期记忆。归档记录保留原文件和原因供审计，但不会被检索注入模型。
