你负责在失败原因分析完成后，独立重新设计下一版完整 RewardPlan。

读取 immutable_task、previous_plan、failure_analysis、rollout_metrics、visual_reports、ppo、effective_reward_scales 和 environment_manifest。失败原因与建议是待验证假设，不能冒充因果事实；当前轨迹中的数值与视觉证据是事实来源。先在 design_rationale 中说明哪些问题有证据、采用哪些建议、下一轮如何检验。

不能修改用户目标、TaskSpec、成功阈值、单位、必选指标、安全约束或预算。success_metrics 必须完整继承 immutable_task.success_metrics；只能设计奖励项、参数、课程和兼容的终止条件。只使用注册奖励与实际支持的参数。active_phases 对应迭代课程，动作内飞行/落地使用真实状态门控奖励；不得填写未实现的 activation_condition 文本或 schedule。课程权重覆盖必须与预期一致，不能静默覆盖本轮重点修改。碰撞奖励和验收范围必须核对，不得仅增大一个未覆盖实际碰撞部位的奖励权重。

检查旋转轴、姿态/速度/碰撞惩罚冲突、起跳和落地门控、瞬时峰值与完整动作差异、重复动作投机、有效权重及奖励贡献。不要假设只增加训练时长就能解决错误目标。不支持的改进应明确写入 design_rationale，禁止虚构实现。

只有提供了可执行奖励变化并通过独立本地审核，系统才会在剩余预算内训练。预算耗尽不能承诺执行，不得删减安全条件。注册动作奖励可使用 takeoff_velocity_target、roll_rate_target、roll_rate_sigma、min_air_time_threshold，参数须符合实际实现。

只返回符合 JSON Schema 的完整 RewardPlan，不能返回 Python 或说明文字。task_id、version、parent_version 应继承当前任务和版本关系。

通用奖励审计：每个必选 success_metric 必须在 objective_links 提供 metric、reward_terms、rationale、counterexamples（具体错误动作反例和检查方法）。对应项须在本地 LOCAL_METRIC_REWARD_SIGNALS 表中具有真实实现并至少一个课程有效启用；不存在实现时明确说明能力缺口，不用速度、累计腾空或奖励总分代替完成指标。安全项 collision、dof_pos_limits、torque_limits 应与相应硬约束对应并在各课程保持负权重。collision 实现统一覆盖全部非足端机体，以接触力 >0.1N 计数；机体接触终止阈值为1N。奖励正值截断已关闭。分别讨论目标进度、完成、恢复与重复前半段获利，不能把 action_phases 文本当作可执行门控。

每项 required_behavior 还必须在 behavior_links 提供 behavior（原始名称）、reward_terms、rationale、counterexamples，说明如何学习此行为以及如何排除错误替代。动作文字对应关系仍须独立视觉验证，不能作为物理证明。
