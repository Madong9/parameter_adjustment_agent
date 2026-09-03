# 实验目录格式

```text
experiments/<task_id>/
├── task_request.txt             # 原始用户指令
├── task_spec.json               # 规范化任务规格
├── environment_manifest.json    # 本任务使用的环境能力清单
├── design_request.json          # 任务设计请求
├── design_response.json         # 任务设计结构化回复
├── state.json                   # 可恢复状态与历史
├── lineage.json                 # 实验谱系
├── task_normalization.json      # 自然语言语义修正及指标别名映射
├── loop_status.json             # 当前闭环轮次、奖励版本、决策和预算
├── loop_history.json            # 历次联合验收结果
├── provider_records/            # 每次 OpenCLI 请求和原始助手回复
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
│       ├── event_takeoff.png、event_landing.png
│       └── visual_prompt.txt、visual_raw_response.txt、visual_report.json、diagnosis.json、decision.json
└── final/                        # 最终 checkpoint、配置和代表性材料
    ├── checkpoint.pt
    ├── config.yaml
    ├── reward_plan.json
    └── contact_sheet_clean.png、contact_sheet_annotated.png
```

Manifest 会记录任务/实验/父实验 ID、Git commit、配置哈希、奖励版本、种子、机器人、完整参数数组、起止时间、iteration、checkpoint、Provider 状态、结果和失败原因。`lineage.json` 的边记录每一版候选的父子关系，最终选中节点会写入完成、失败或人工复核结果。状态与 JSON 均使用原子写入；同一自然语言任务再次运行会保留旧候选和 checkpoint，但清除会误导界面的旧终态摘要，并用新的 `run-<编号>` 候选目录隔离本次训练。
