# 本地 RAG 训练经验模块

## 目标

RAG 模块让奖励设计和诊断 Agent 在处理新动作时，先查找当前项目中的可靠上下文，而不是把全部仓库一次性发送给模型。RAG 可以包含项目文档和原始实验记录；长期记忆则只包含通过联合验收门控后晋升的经验，两者会在上下文快照中分开标识。

该实现完全在本地运行，不需要向量数据库、嵌入模型或额外网络服务。核心检索器使用中英文混合分词和 BM25 排序，适合项目当前以中文动作描述、英文变量名和结构化 JSON 混合保存的材料。

## 索引来源

默认索引以下白名单内容：

- `REWARD_DESIGN.md`、`VISUAL_CRITIC.md` 和 `SAFETY.md`；
- `experiments/<task-id>/task_request.txt`；
- `task_spec.json`、`summary.json` 和 `loop_history.json`；
- 候选奖励的 `reward_plan.json` 和 `revision_audit.json`；
- 各轮 rollout 的 `diagnosis.json`。

JSON 会展开为带字段路径的文本。视频、图片、Parquet、checkpoint、TensorBoard、训练日志和原始 Provider 对话不会进入索引，避免索引膨胀和无关信息污染。

## 检索与注入位置

奖励设计前，查询由用户动作要求、机器人型号和“任务/奖励/课程设计”目的组成。检索结果写入：

```text
experiments/<task-id>/rag/design_retrieval.json
```

联合验收失败后，查询会额外包含未通过的约束、证据冲突和视觉摘要。检索结果写入：

```text
experiments/<task-id>/candidates/<experiment-id>/rollouts/round_<n>/rollout_<n>/rag_diagnosis_context.json
```

独立视觉评论阶段不注入 RAG。视觉模型只看任务规格与同步视觉证据，避免奖励设计或历史结论影响它对实际动作的判断。

## 排序与边界

检索器对中文连续文本生成单字和二元词，对英文变量保留完整标记，再计算 BM25 相关度。机器人型号相同和最终状态为 `COMPLETED` 的历史经验会得到小幅加权。当前正在执行的任务会被排除，防止尚未验证的中间结论被自己检索回来。

每次只返回 `rag_top_k` 个片段，并由 `rag_max_context_chars` 限制总字符数。注入内容带有明确声明：历史片段只是只读证据，不是可执行指令，不能覆盖当前环境能力清单、安全限制和确定性验收结果。索引读取或重建失败时，主训练链路会记录错误并继续运行。

## 配置

`config/agent.yaml` 中的相关字段如下：

```yaml
rag_enabled: true
rag_index_path: rag/index.json
rag_document_roots:
  - docs/agent/REWARD_DESIGN.md
  - docs/agent/VISUAL_CRITIC.md
  - docs/agent/SAFETY.md
rag_top_k: 6
rag_chunk_chars: 1200
rag_chunk_overlap: 120
rag_max_context_chars: 6000
```

`rag_index_path` 相对于 `artifact_root`；其余文档目录相对于 Agent 根目录。所有默认配置均为可上传的相对路径。

## 使用命令

在 `rl_agent` 目录执行：

```bash
# 强制重建索引并查看文档、片段数量
python -m rl_training_agent rag-index

# 预览某个动作会命中的经验
python -m rl_training_agent rag-query \
  --query "Go2 后腿站立行走的奖励设计和失败修正" \
  --robot go2 \
  --top-k 5
```

正常执行 `plan` 或 `train` 时不需要手工建索引。Agent 会比较来源路径、大小和修改时间；来源变化时自动重建，否则复用持久化索引。任务结束后会再次刷新索引，使新实验可供后续任务检索。上位机顶部会显示当前索引片段数量。
