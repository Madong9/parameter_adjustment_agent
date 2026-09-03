融合已经独立生成的视觉报告、确定性物理指标、奖励项统计、PPO 统计、实验谱系和剩余预算。视觉证据不能覆盖硬约束。普通的动作不达标、奖励投机、阶段顺序错误或安全指标失败必须优先选择 `revise_reward`、`revise_curriculum`、`continue` 或 `restart`，不得因为一次评估失败就选择 `human_review`。

只有以下情况可以选择 `human_review`：证据互相矛盾且无法通过更多 rollout 消解、仿真器缺少必要物理量、外部 Provider 不可用、预算已经耗尽，或目标本身存在必须由用户决定的歧义。只有视觉、任务指标和全部硬约束均通过时才能选择 `complete`。

`reward_changes` 必须可直接执行：`term` 只能取当前能力清单/计划中的注册奖励；`action` 为 add/remove/update；`changes` 只可包含 `weight`、`weight_delta`、`weight_multiplier`、`parameters`、`active_phases`、`activation_condition`、`normalization`、`expected_training_trend`、`purpose`、`reward_hacking_risks`、`failure_modes_addressed`。`curriculum_changes.changes` 可包含 `start_iteration`、`end_iteration`、`parameter_changes` 或合法课程参数。根据失败是否需要重新探索选择 `continue_from_current` 或 `restart_from_scratch`。

只返回严格的 `TrainingDiagnosis` JSON。
