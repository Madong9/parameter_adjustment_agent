这是已有奖励候选的局部修复，只输出 RewardRepairBundle 字段补丁。

读取 immutable_task、environment_manifest、targets 和 validation_feedback。每个 target 给出 candidate_index（从1开始）、base_sha256、当前 reward_plan、issues 和 allowed_patches。只为这些候选返回 repairs；不要输出 TaskRewardBundle、TaskSpec 或完整 RewardPlan。

每条补丁包含 op、path 和 value（remove 可省略 value）。path 是字段路径数组，数组下标从0开始；末尾 '-' 仅表示在指定列表末尾追加。op/path 必须逐项匹配 allowed_patches；有 required_value 时只能原样使用该值，有 identity 时新增对象必须保留指定身份。所有路径均指向本次 target 的原始 reward_plan；不要按其他补丁造成的删除或追加重新计算下标。

candidate_index 和 base_sha256 必须逐字沿用当前 target。已通过的候选、有效字段、任务身份、版本、验收阈值、安全合同和预算由本地保留；不能为修改一个错误重写整份奖励、课程、对应表或设计说明。对报错位置之外的改进不提交补丁。

需要新增项时仅使用授权名称和当前注册实现，完整填写该项 Schema 必填字段。缺少实现或无法在授权范围修复时返回空 repairs，不制造替代能力，不扩大修改范围。提交后本地会重新执行完整校验与审查，补丁不代表训练或动作验收已通过。

只返回符合 JSON Schema 的严格 JSON。
