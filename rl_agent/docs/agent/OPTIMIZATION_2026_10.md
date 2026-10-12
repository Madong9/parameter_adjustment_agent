# 动作语义、接触证据与预算优化

日期：2026-10-06。本轮在现有确定性状态机、多 Agent、可行性预检、奖励审查、PPO 与记忆架构上优化，不更换 PPO 算法，也不自动增加训练预算。

## 1. 动作目标不能在流程中悄悄改变

之前的侧向动作可能存在语义漂移：用户只说“侧向翻滚”，任务理解描述地面侧滚，但关键词匹配会触发空翻奖励和起跳、腾空、落地验收。这不只是训练迭代不足，而是动作目标与合同不一致。

新增 `agents/task_semantics.py`，使用用户原始输入核对 `TaskIntentSpec` 和奖励设计后的 `TaskSpec`：

| 原始目标 | 路由与门控 |
| --- | --- |
| “贴地侧滚后站起”“侧向翻滚，不起跳” | `GROUND_ROLL`；不得增加必选腾空或全足离地要求 |
| “起跳腾空侧翻后落地”“侧空翻” | `AERIAL_SIDE_FLIP`；必须保留起跳/腾空动作要求 |
| “侧向翻滚”“侧翻一圈” | `NEEDS_CLARIFICATION`；先明确贴地还是腾空 |
| “保持站立，不做侧空翻” | 站立；被否定的空翻不能触发空翻模板 |

这是一组保守的确定性语义规则，不是通用自然语言正确性的证明。除侧滚/侧翻及身份校验外，其他复杂目标仍需任务指标和实际行为验收。

审核结果保存为任务目录内的 `intent_semantic_check.json`、`task_semantic_check.json`，并写入 `events.jsonl`。冲突或歧义进入 `HUMAN_REVIEW`，不通过奖励重新设计、阈值放宽或追加预算掩盖。

贴地侧滚新增独立动作类型与约束分支。目前尚无经过验证的机体接触阶段合同及专用全身求解、物理执行器，返回 `CAPABILITY_ONLY/CONDITIONAL`。这表示缺少验证实现，不表示机器人一定无法学会，更不能用空翻或普通行走探针冒充已经验证。

## 2. 接触奖励、终止与评估必须描述同一事件

编译器现在接受 `compile(plan, directory, task=task)`，将真实任务硬约束编入 `config.yaml` 的 `safety_constraints` 与 `contact_semantics`。旧两参数接口保留，但生产编排始终传入任务。

`training/contact_semantics.py` 统一定义禁止接触刚体、事件阈值、最大接触力上界及比较符：

- 禁止集合由真实刚体索引排除足端得到，不再只依赖 `thigh/calf` 名称，避免漏掉基座或髋部。
- 碰撞事件默认阈值为 `> 0.1 N`，最大接触力默认底线为 `> 1 N`。它们是不同指标，分别记录和检查；任务更严格的上界会收紧运行合同。
- 不能手改编译配置放宽硬约束，不能关闭终止项来取消固定非足端接触检查。
- 不根据“翻滚”名字自动开放机体触地。计划接触需要单独设计明确、可审核的阶段合同。

训练目录保存 `contact_audit.json`；rollout 的 `metadata.json` 保存实际生效合同、足端与非足端刚体索引和采集阈值。

奖励质量审计同时用于短训练 pilot 和最终 rollout：不仅检查平均 `raw_collision`，还逐帧检查碰撞事件与原始、加权奖励是否一致。生产缺少合同元数据、时间未对齐或非足端索引不完整时审计失败。未超过合同上界的小幅正接触力不再被错误地一律当作违规。

## 3. 预算耗尽不是没有训练，也不表示训练成功

`BudgetTracker` 提供已用、上限、剩余和耗尽状态；上位机从该运行持久化的冻结记录读取，不能用当前 YAML 的默认值推算旧运行剩余额度。

界面分别展示：

- PPO：奖励版本是否已编译、是否有 checkpoint、训练是否完成。
- 验收：确定性指标、视觉与硬约束是否联合通过；Mock 不显示为真实通过。
- 预算：迭代和奖励修订的已用/上限，以及剩余额度。

最后一次合法训练后的首次评估仍然执行，不会因刚好耗尽而跳过验收。但已经停在 `HUMAN_REVIEW` 的耗尽运行再次恢复，只保存清楚的停止提示，不再调用 Provider、重复评估或扣减额度。

耗尽或冻结额度未知时，界面和 API 都禁止自动恢复。已有额度的 Provider 暂停仍可恢复，前提是动作语义、配置哈希、环境兼容性、checkpoint 和验收合同均一致。旧运行缺 `task_intent.json` 时只从已冻结的原始指令构造最小意图，不重新请求模型或重设计任务。

“已完成 v3 奖励训练”与“下一轮未启动”可以同时成立。前者说明已有策略经过训练，后者说明预算不允许再运行 v4；两者都不能替代动作验收。

没有新增自动扩额功能，也没有重置历史消耗。预算耗尽后可以只读复核已有轨迹；已联合验收通过的策略仍能播放，不消耗训练额度。

## 4. 状态机与调用顺序

```text
用户原文 → TaskIntentSpec → 意图语义审核
  歧义/冲突 → HUMAN_REVIEW
  一致 → 可行性预检 → 上下文 → 奖励设计 → TaskSpec 语义审核
    一致 → 逐候选审核 → 带接触合同编译 → 短训练奖励审计 → PPO
      → rollout 合同/奖励审计 → 数值与视觉联合验收 → 诊断/记忆

恢复 → 核对冻结记录与已训练策略
  预算耗尽 → 保存停止提示，不请求模型
  有额度 → 核对原动作与合同 → 继续已有闭环
```

没有新增大量状态。补齐 `SMOKE_TRAINING`、`CANDIDATE_SCREENING` 到 `HUMAN_REVIEW` 的合法安全收束边，仍禁止这两个阶段直接转为 `COMPLETED`。

## 5. 修改入口与测试

| 部分 | 主要文件 |
| --- | --- |
| 语义合同 | `agents/task_semantics.py`、`utils/task_semantics.py`、任务理解/奖励提示词 |
| 预检分流 | `feasibility/agent.py`、`motion_constraints.py`、`planning.py`、动态原型 Schema/Generator |
| 接触合同 | `training/contact_semantics.py`、`config_runtime.py`、`real_train.py`、`real_rollout.py` |
| 奖励质量 | `rewards/compiler.py`、`validator.py`、`quality_gate.py`、`agents/reward_reviewer.py` |
| 闭环恢复 | `orchestration/budget.py`、`orchestrator.py`、`state_machine.py` |
| 上位机 | `web_ui.py`、`web/index.html`、`web/app.js`、`web/app.css` |

新增四组回归：`test_task_semantic_contract.py`、`test_contact_contract.py`、`test_budget_optimization.py`、`test_budget_ui.py`。包含原文歧义、否定识别、目标漂移、接触完整性、逐帧错配、生产缺证据、冻结额度、恢复幂等及已验收播放边界。

在项目 `rl_agent` 目录、已安装的 `rl_agent` Conda 环境执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -o addopts=''
python -m compileall -q rl_training_agent tests
node --check rl_training_agent/web/app.js
git diff --check
```

结果：**562 项测试通过**，编译、JavaScript 语法与差异检查通过。测试使用临时目录和离线证据，没有为本轮优化发送真实模型请求或启动真实 PPO；测试通过不是运动策略成功证明。

## 6. 使用和后续边界

重启上位机加载新代码。给侧向动作输入明确目标，例如“向左贴地侧滚一圈后站起”或“向左起跳腾空侧翻一圈后落地”。两者是不同任务，不应把旧错误合同直接修改成另一个动作后续训。

运行时接触语义升级会改变编译配置及兼容性指纹，旧 checkpoint 不会被自动迁移成新合同通过证据。不兼容时需要审核或建立新实验，保留旧策略及原始证据。

后续优先级：实现经过审核的计划机体接触阶段合同及专用侧滚执行器；建立真实多种子动作基准；提供显式审批、保留已用额度和完整审计的预算扩展流程。以上未实现部分不会被本轮测试结果宣称为已经完成。
