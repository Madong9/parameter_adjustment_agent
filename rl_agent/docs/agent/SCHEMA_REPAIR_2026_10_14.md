# 奖励设计结构校验与无损修复

## 本轮问题

作业 `445f0bb3d389` 在 2026-10-14 本地时间 20:33:59 进入奖励设计人工复核，未启动 PPO。用户要求安全停止后，已确认没有活动训练、排队作业或遗留训练子进程。

首次模型回复只有 3 个结构错误：TaskSpec.safety_constraints 输出字符串，而 Schema 要求 MetricThreshold 对象：

```text
forbidden_body_contact <= 1 N
joint_limit_violations == 0
torque_limit_violations == 0
```

两轮整份回复重写丢掉了已正确的字段，最后变为 35 个错误：5 个任务必填字段、24 个奖励 purpose、6 个课程起止边界。所有回复还复用了历史 G1 实验编号 `task-845494f509`，当前请求却是 Go2 的 `task-ccd8109bb2`。初次请求包含历史 RAG 编号，没有显式给出当前编号。

这属于模型回复结构不合格及修复退化，不能直接归因于网络服务不可用，也不能通过给必填字段设置空列表来放行。

## 修改

1. `providers/schema_repair.py` 仅把 TaskRewardBundle.task_spec.success_metrics / safety_constraints 中完整、单一的“指标 比较符 有限数值 [单位]”转换为对象。保留原数字、方向和单位；没有单位就保留空。模糊文本、多条件、缺数值、NaN/Infinity 和溢出值不转换。缺失的安全合同、奖励用途、课程边界继续由严格 Schema 拒绝，不补造阈值。
2. `_request_model` 保存最完整的来源及结构化错误，只让格式重试修改无效位置。已有合法字段、默认值、候选数量、课程、任务身份和验收数值受到保护。包括隐式 required=True，不能添加 required=False 来禁用安全指标，也不能追加未声明列表项。仅对 CurriculumStage 已知的起止顺序错误允许修正边界，其他字段继续冻结。退化回复不替代来源，下轮仍基于保留的来源修复；耗尽时报告真实剩余问题。
3. 成功转换记录为 Provider 会话目录的 `schema_normalization_*.json`，原始 request/response 不被修改。流式补读继续检查本轮用户/助手消息归属，换会话恢复继续携带完整原始上下文。
4. 提示词升级为 `reward-design-v10`。编译请求显式给出 CURRENT_TASK_IDENTITY，并从当前 Schema 自动生成 SCHEMA_REQUIRED_FIELDS，明确目的字段、课程迭代边界和安全对象结构。RAG 中的编号只属于历史证据。
5. 编排器将用户请求作为最终身份来源，记录 `task_identity_normalization.json` 及计划的 provided/effective task_id。独立审查和编译入口也拒绝任务编号或机器人不一致的合同。
6. 格式异常报告 `failure_kind=response_schema` 和“奖励设计回复结构不合格”，区别于 Provider 超时或连接故障。PPO 未启动时仍保持人工复核，不自动恢复或重启。

## 离线验证

原始 response_001 只转换上述 3 项便通过完整 TaskRewardBundle 校验，保留 3 个候选、各 8 项奖励。纠正本地任务身份后，本轮合同的独立前翻基线通过原有审查与配置编译；验收与安全约束继续生效。

记录位于 `artifacts/schema_repair_validation/replay_report.json` 和 `schema_normalization.json`。此次没有发送模型请求、启动 PPO 或改写旧作业记录。训练保持停止；如后续重新下发，新作业会读取修复后的代码。

全量 **741 项通过，31.34 秒**，最终格式异常恢复提示的相关 6 项检查通过。Python 编译、JavaScript 语法和 `git diff --check` 通过。新增回归覆盖无损转换、不补造阈值、无效/歧义拒绝、修复退化、隐式安全默认值保护、列表数量、课程边界、当前任务身份和独立审查编译的一致性。

## 后续：版本号与设计说明类型

作业 `f60d48e0a2f0` 在本地时间 21:42:39 结束于奖励设计复核。首份回复的三个候选均使用 `"candidate-01"` / `"candidate-02"` / `"candidate-03"` 版本标签和单段 `design_rationale`，而 Schema 要求整数和字符串数组，共六项类型错误。两轮模型格式修复仍未修正这些类型，还改动了奖励权重、课程名称、用途和映射列表，原有保护因此正确拒绝了这些回复。

统一的 `normalize_reward_response` 在 TaskRewardBundle 和独立 RewardPlan 中处理这两种明确表示：

- `version` / `parent_version` 仅接受完整匹配 `candidate-<1–9 位十进制正整数>` 的标签，保留数字含义并转为整数。零、负数、小数、歧义后缀、大小写变化及未知标签不猜测。
- 非空 `design_rationale` 字符串包装为单项数组，逐字保留空白和完整说明，不拆分、不重写。空说明仍拒绝。
- 输入对象保持不变；奖励参数、权重、课程、目标和行为映射、验收阈值继续受到原有保护。转换在初次解析和修复来源检查中一致应用，避免重复向模型请求这六项格式修复。

审计 `scope` 为 `lossless_reward_response_types_v2`，逐项记录字段路径、原值和规范值。提示词升级为 `reward-design-v11`，初次设计与共享重设计提示均明确版本号、父版本及设计说明类型。

最新首份回复离线重放通过 TaskRewardBundle，三份版本为 1、2、3，各保留 8 项奖励；六项转换的原说明、奖励权重和课程名称均保持不变。模型候选仍因映射不完整而未通过奖励审查；现有独立本地前翻基线通过同一审查与配置编译，原有训练准入流程继续生效。记录位于 `artifacts/metadata_repair_validation/{replay_report,normalizations,original_candidate_review}.json`。没有模型请求、PPO 或旧作业记录修改。

新增 33 项离线回归验证类型转换、一次请求即可解析、审计、父版本、严格标签范围、幂等及执行字段保护。完整测试集 **774 项通过，32.78 秒**；Python 编译、JavaScript 语法和 `git diff --check` 通过。当前作业列表没有运行、停止中或排队任务。

## 后续：终止建议与统一字段类型

作业 `09d9e1672221` 于本地时间 22:15:44 停在奖励设计复核。首份回复的版本和设计说明已符合类型，但 `termination_suggestions` 的三个条目是 `{name, condition}` 对象，而该说明字段要求字符串数组。实际执行条件属于 `RewardPlan.terminations: List[TerminationSpec]`，两者语义不同。两份格式重写共有 42 / 58 处有效字段回归，其中第二份回复删除了候选 2 / 3 的关节和力矩安全奖励；保留最完整来源及防退化检查仍然必要。

反复出错的原因是网页模型没有硬性结构输出约束，提示词要求和完整 JSON Schema 都不能保证每次服从。此前的简明摘要只列必填字段，没有统一列出所有字段类型；针对已出现格式的转换也不能提前覆盖每一种新表示。不同轮次会在不同字段出错，这不能归因于 PPO 崩溃。

本轮统一处理两个边界：

1. 仅 TaskRewardBundle 顶层的 `termination_suggestions` / `reward_hacking_risks` 说明列表支持对象转完整 JSON 文本。保留所有键、数值、未知及嵌套内容，要求 JSON 往返一致且没有非有限值；不抽取 condition、不删其他属性。列表数量、顺序、已有文字和输入对象保持不变。实际执行终止条件、硬约束、参数和其他同名字段不应用此转换。审计 `scope` 为 `lossless_reward_response_types_v3`。
2. `agents/schema_contract.py` 从当前 JSON Schema 和真实模型的可空信息生成 `SCHEMA_FIELD_TYPES`，包含全部字段、必填/可选、模型引用、字符串/对象数组、元组、字典及枚举，区分可省略与可为 null。摘要同时用于初次、兼容入口和 RewardPlan 失败重设计，Schema 新增字段会自动纳入。完整 Schema、确定性校验和防退化保护保留。提示词升级 `reward-design-v12`，并明确每候选继承验收、前翻落地/站稳采用正值必选下界。

首份回复离线重放只转换三条终止建议便通过 TaskRewardBundle；JSON 解码可还原原对象，三个候选各 8 项奖励及全部其他原始字段保持不变。原始方案的落地稳定与站稳时长下界为 0，且候选缺失验收和对应表，因此仍未通过动作合同及奖励审查，本地基线也不会为无效合同放行。未自行填造正值阈值或改写已有作业。

记录位于 `artifacts/advisory_repair_validation/{replay_report,normalizations,original_candidate_review,task_semantics}.json`。新增 22 项离线回归，完整测试集 **796 项通过，32.31 秒**；Python 编译、JavaScript 语法和 `git diff --check` 通过。未发送模型请求或启动 PPO。网页回复今后仍可能出错，缺少执行字段或无效业务合同继续阻止准入，不能把本次格式兼容等同于动作已学会。

## 2026-10-15：按错误字段合并修复，统一 Provider 行为

再次核对确认，用户提供的三条终止建议类型错误仍来自历史作业 `09d9e1672221`。当前网页入口可直接解析其首份回复，保留三个候选、每候选八项奖励及其他全部字段。旧作业的复核原因作为历史记录保留；新下发任务由独立 Python 子进程加载当前代码，不需要为 Provider 修改重启上位机。

此前还有一个反复停止的原因：模型即便修好了错误字段，只要顺手改动其他有效内容，程序就会丢弃整份回复。现在 OpenCLI 和百炼共用 `merge_schema_repairs`，从完整来源的副本出发，只取 `validation_errors` 明确指出的字段；风险、课程、权重、安全阈值和默认值继续使用原值。合并结果重新经过完整 Schema 和字段保留检查。网页入口可以保留已减少错误的局部进度，下一轮基于该来源继续修复，重试上限不增加。

数组内补丁要求长度一致、对应元素身份不变，且该身份在原数组和回复数组中均唯一。无法确定身份、重排造成索引错位、已有容器整体替换和根校验错误不做局部合并。缺失修复值或仍然不合格的结果继续阻止训练。字段修复记录 `schema_field_repair_*.json`（百炼为 `*_schema_field_repair.json`），包含采用的字段、忽略的其他改动和最终有效内容；网页合并审计同时保留来源与补丁的类型转换明细。

百炼入口补齐与网页入口相同的奖励表示规范化、转换审计、原始任务上下文及有效字段保护，避免切换 Provider 后重新出现说明对象的字符串类型错误。

离线重放报告为 `artifacts/reward_field_repair_validation/replay_report.json`：网页原文解析通过；百炼对网页已提取的同一 JSON 内容仅需一次离线响应读取，两个入口结果一致，三条说明可完整还原。该网页原文另含已有网页解析器处理的比较符多余引号，所以百炼重放使用已提取 JSON，不宣称百炼支持相同的网页语法兼容。没有外部模型请求、PPO 启动或旧作业改写；原任务的业务合同问题仍由原有审查处理。

同时更新已有文档分块上传实现的测试：长中文与 emoji 文档逐字节还原，每次 eval 不超过 64 KiB，并覆盖多文件、空文档、异常确认和清理。这针对后续另一个作业 `f50c82cf9520` 的命令参数过长问题，不与历史的三项类型错误混同。

验证：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robot/miniconda3/envs/rl_agent/bin/python -m pytest --tb=short`，完整测试集 **876 项通过，32.31 秒**；三个修改的 Provider 模块编译和 `git diff --check` 通过。禁用第三方 pytest 插件自动加载用于隔离宿主 ROS 的测试插件，项目自己的 `conftest.py` 照常加载。
