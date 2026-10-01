# 平台强化实现说明

本轮升级把总文档中的 P0 和后续工程建议落实为可执行模块。真实长训练仍然必须在 OpenCLI、百炼密钥和 GPU 正常时运行；离线测试只验证控制链，不能替代收敛证据。

## 1. 真实动作基准

固定套件位于 `config/benchmarks.yaml`，当前包含 Go2 前进、原地转向、跳跃、后腿站立和后腿行走。套件固定：

- 套件版本和名称；
- 机器人；
- 训练工程 Git commit 要求；
- 随机种子；
- 重复次数；
- 动作原文；
- 必须出现的验收指标。

真实运行：

```bash
python -m rl_training_agent benchmark \
  --suite config/benchmarks.yaml \
  --provider multi-agent
```

只运行一个动作：

```bash
python -m rl_training_agent benchmark \
  --suite config/benchmarks.yaml \
  --case forward_walk \
  --provider multi-agent
```

离线控制链验证必须显式使用 Mock：

```bash
python -m rl_training_agent benchmark \
  --suite config/benchmarks.yaml \
  --case forward_walk \
  --provider mock \
  --dry-run
```

报告保存在 `artifacts/benchmarks/<benchmark-id>/report.json`，通过 `real_evidence` 区分真实证据与模拟演练，并记录环境 commit、每次任务摘要、指标覆盖和终态一致性。

## 2. 多种子视觉保守聚合

每轮真实评估按数值分数选择互不重复的最差、中央和最好 rollout，分别调用视觉 Agent。聚合规则为：

- 所有样本视觉成功才允许通过；
- `alignment_score` 使用最低值；
- 任一报告需要人工复核，聚合结果就需要人工复核；
- 样本结论不一致时自动进入人工复核；
- 聚合可信度为最低原始可信度乘以结论一致率。

单样本报告写入 `visual_report_individual.json`，聚合证据写入 `visual_ensemble.json`，最终兼容报告仍为 `visual_report.json`。

## 3. 状态机和恢复

状态机现在使用显式允许转换表。非法跳转会抛出错误；带相同 `operation_id` 的状态提交只执行一次。同一稳定 `task_id` 的新训练必须显式调用 `start_new_run`，旧历史保留但运行上下文和操作键重新初始化。

恢复只允许从 `HUMAN_REVIEW` 进入，并校验：

- 最终摘要和候选存在；
- 奖励版本与 manifest 一致；
- 编译配置哈希与 manifest 一致；
- 多种子 checkpoint 可以找到。

已成功保存的单 rollout 视觉报告和诊断报告会被复用，防止 Provider 成功后进程崩溃导致重复消息。

## 4. Provider 注册表

`ProviderRegistry` 为每个 Provider 声明能力：

- `task_understanding`；
- `reward_design`；
- `visual_critique`；
- `training_diagnosis`。

`MultiAgentProvider` 按 `agent.yaml` 的四个角色字段创建实例并在启动前验证。例如百炼可承担奖励设计和诊断，但不能被配置为需要图片能力的视觉 Agent。

## 5. 混合 RAG

RAG 分数由 BM25 与本地 TF-IDF 稀疏向量余弦相似度融合：

```yaml
rag_bm25_weight: 0.65
rag_vector_weight: 0.35
```

该实现不需要外部向量数据库或嵌入服务，保留原有白名单、来源审计、当前任务排除和字符预算。后续如果加入神经嵌入，必须先建立固定检索评测集。

## 6. 语义记忆冲突

语义记录新增反例和替代关系。明确失败模式达到独立证据阈值后，如果反驳已有姿态规律，旧规律会改为 `superseded`，保存：

- `counterevidence_memory_ids`；
- `superseded_by`；
- `archive_reason`。

被替代规律不会再进入模型上下文，但原文件不会删除。

## 7. 置信度与奖励投机

视觉置信度通过跨样本一致率校准，不再直接信任单次模型自报值。

命令跟踪类任务额外执行零命令反事实 rollout。若零命令下机体漂移速度超过阈值或发生摔倒，则写入 `reward_hacking_counterfactual` 硬违规并阻止完成。跳跃等非命令动作不会错误应用零命令探针。

## 8. 可观测性

任务目录新增 `events.jsonl`，记录运行开始、状态转换和 Provider 调用耗时。上位机作业目录也保存 `events.jsonl`，在作业开始和结束时记录 GPU 利用率及显存快照。事件不包含 API Key 或完整提示词。

## 9. 多 GPU 持久队列

GPU 池由相对配置文件指定：

```yaml
gpu_ids: [0]
```

配置多个编号后，不同作业可以并行占用不同 GPU。GPU 全部占用时，新作业以 `queued` 状态持久化；任一作业结束后按创建时间自动分配和启动。服务重启后，带安全命令的等待作业可以恢复调度；旧版缺少命令的队列记录会转为 `interrupted`，不会盲目执行。

真实作业使用 `CUDA_VISIBLE_DEVICES` 绑定分配设备。等待作业可以在启动前取消。

## 10. CI、环境和迁移

- `.github/workflows/rl-agent-ci.yml`：Python 3.8 CPU 测试和编译检查；
- `environment.yml` 与 `requirements.lock`：Conda 入口和当前已验证依赖精确版本；
- `migrate-artifacts`：为旧实验和记忆补充格式版本。

默认只预览迁移：

```bash
python -m rl_training_agent migrate-artifacts
```

确认后写盘：

```bash
python -m rl_training_agent migrate-artifacts --apply
```

## 11. 尚需外部执行的验证

代码、协议和离线测试完成不代表真实策略已经收敛。以下工作必须在服务和 GPU 条件满足后执行：

1. 设置 `DASHSCOPE_API_KEY`；
2. 恢复 OpenCLI Browser Bridge；
3. 运行 `doctor` 和 `opencli-test`；
4. 执行全部真实基准；
5. 比较两次重复运行的终态、指标、奖励版本、训练时间和失败模式；
6. 对未达到一致性的动作继续修正奖励、环境或验收阈值。

只有 `real_evidence=true` 的基准报告可以用于声明真实可重复性。
