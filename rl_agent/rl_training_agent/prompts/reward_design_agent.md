这是初次奖励设计。任务理解已经提供 TASK_INTENT_SPEC，本地上下文通过 CONTEXT_SNAPSHOT 提供环境能力、可行性探针、训练准入、RAG 和记忆。执行共享通用奖励设计规则。

输出 TaskRewardBundle：task_spec、reward_plans、reward_hacking_risks、termination_suggestions。输出恰好 CANDIDATE_COUNT 个候选，全部保持同一动作语义和验收合同。候选应有有意义、可检验的不同假设，例如任务信号与安全正则的平衡、已实现门控的选择或不同迭代课程；不能只改版本号或复制同一配方。共同能力缺口必须在各候选如实记录，不用未注册项填满候选。

初次 TaskSpec 的物理阈值根据明确用户要求、任务意图和当前环境证据给出；未明确的设定标为设计假设，不以历史失败实验作为新要求的来源。收到 fixed_task_spec 或 immutable_task 时改为完整继承，不重新解释阈值。ALLOW_BUDGETED_EXPLORATION 只允许预算内仿真探索，不代表物理验证通过。探针失败不证明不可学习；保留证据和风险。

按机器人和支撑语义选择实现。前足支撑不等于前足离地；前腿/后腿站立行走应采用当前注册的对应支撑与姿态门控行走项，避免未门控速度奖励绕过姿态要求。velocity_frame 继承 TaskSpec；倒立前进按 heading 水平航向语义。跟踪得分与实际 m/s 速度指标不能混用单位，若需要 front_leg_forward_speed 等指标必须核对当前清单。

不要预先为任意动态动作指定一套起跳—滚转—恢复配方。仅有 roll_rotation 角速度信号时，不得声称 final_roll_angle 净滚转目标已经被覆盖。LOCAL_METRIC_REWARD_SIGNALS 中空列表明确表示当前缺少已登记的目标训练信号，按共享规则报告 [CAPABILITY_GAP]。

TASK_INTENT_SPEC 的动作与禁止行为必须完整保留。地面侧滚不得改写为腾空侧空翻，腾空侧翻也不得用贴地翻身替代。若计划机体接触与现有全非足端禁触合同冲突，报告能力/接触合同缺口；不得为了生成可训练候选而删除硬约束或偷偷添加其他动作阶段。

只返回符合 OUTPUT_JSON_SCHEMA 的严格 JSON，不返回 Markdown 或 JSON 外说明。

当前本地前/后腿支撑审查要求在清单支持时同时列出对应的 front_leg_stand/front_leg_walk 或 rear_leg_stand/rear_leg_walk。仅要求站立时不要因此增加行走验收或非零行走命令，按动作目标设置有效权重；不能用水平 orientation 与倒立目标对抗。
