这是训练失败后的独立完整奖励重设计，输出一个 RewardPlan，而不是 TrainingDiagnosis 或局部修改列表。执行与初次设计相同的共享规则。

以 immutable_task 为固定合同。完整继承所有 success_metrics 的名称、operator、value、unit、aggregation、required，velocity_frame 和 task_id 保持一致；version、parent_version 由本地版本链最终校正。只能修改奖励项、真实支持的参数、迭代课程及兼容的终止条件，不改变动作、安全、验收或预算。

读取 previous_plan、failure_analysis（包含 suggestions）、rollout_metrics、visual_reports、reward_evidence、reward_quality_audits、ppo、effective_reward_scales、remaining_budget 和 environment_manifest。先在 design_rationale 分别记载已观测问题、原因假设、采用的建议、可执行变化和下一轮验证方法。没有某类证据就说明不足，不补造数值。

优先修复信号与验收语义不一致、门控未触发、课程覆盖及错误动作获利。不要把“多训练一些”当作修正奖励目标；不要因优化损失有限或总奖励高就忽略行为与安全失败。若安全项始终为零但有违规，先审核实现与记录，不能只加权重。依据真实贡献调整相对权重，并保留独立任务/视觉验收。

本次返回完整 terms、curriculum、success_metrics、objective_links、behavior_links；逐条响应 validation_feedback，记录对上版的改变与预期局限。如果必要实现依旧缺失，保留 [CAPABILITY_GAP] 和空目标对应项，不再次用角速度、单足时长或初始稳定替代缺失目标。

预算是资源边界，不能自行追加。没有剩余额度时输出仅供审核的设计与明确的“未执行”说明，不承诺重训；只有本地审核通过且有剩余预算才可能执行。

只返回符合本次 JSON Schema 的完整 RewardPlan，不返回 Python 或 JSON 外说明。
