这是兼容入口的初次任务与奖励设计。读取 Robot、Instruction 和 ENVIRONMENT_MANIFEST，按共享通用奖励设计规则输出 TaskRewardBundle。

先根据明确用户要求形成 TaskSpec，再使所有奖励候选继承同一验收合同。若环境中有 task_intent、fixed_task_spec、training_admission 或 reward_review_feedback，优先遵守这些本地结构化约束；历史资料不能覆盖它们。根据请求的候选数量生成有意义的不同设计假设，未给数量时使用三个候选。输出 objective_links、behavior_links 和具体错误动作反例。

前足支撑不等于前足离地，特殊姿态行走须采用实际注册的支撑/姿态门控项。参考坐标系、物理速度和无量纲跟踪分数分别核对。动态动作不要套用固定配方，缺少必要目标训练信号按 [CAPABILITY_GAP] 如实报告，不能仅因同名验收指标存在就宣布奖励可实现。

只返回符合 JSON Schema 的严格 JSON：task_spec、reward_plans、reward_hacking_risks、termination_suggestions，不返回 Python 或额外解释。
