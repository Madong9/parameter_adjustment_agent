这是初次奖励设计。任务理解已经提供 TASK_INTENT_SPEC，本地上下文通过 CONTEXT_SNAPSHOT 提供环境能力、可行性探针、训练准入、RAG 和记忆。执行共享通用奖励设计规则。

输出 TaskRewardBundle：task_spec、reward_plans、reward_hacking_risks、termination_suggestions。输出恰好 CANDIDATE_COUNT 个候选，全部保持同一动作语义和验收合同。候选应有有意义、可检验的不同假设，例如任务信号与安全正则的平衡、已实现门控的选择或不同迭代课程；不能只改版本号或复制同一配方。共同能力缺口必须在各候选如实记录，不用未注册项填满候选。

当前任务身份以 CURRENT_TASK_IDENTITY 为准：TaskSpec 的 task_id、robot、original_instruction 和每个 RewardPlan 的 task_id 必须完整继承。CONTEXT_SNAPSHOT 中检索结果、记忆和示例的 task_id 属于历史实验，不能复制到当前任务，也不能从历史机器人推断当前机器人。

提交前逐项检查 SCHEMA_REQUIRED_FIELDS 与 OUTPUT_JSON_SCHEMA。TaskSpec 的 phases、required_observations、required_sensors、safety_constraints、visual_evaluation_requirements 即使没有可提供的内容也不能省略；空列表须符合当前任务语义，不得用来删除必须的安全与验收内容。每个 terms 项给出 purpose，说明奖励在当前动作中的训练作用；有 curriculum 时，每项提供非负整数 start_iteration 与 end_iteration，起点不得晚于终点。

safety_constraints 使用 MetricThreshold 对象列表，每个对象明确 name、operator、数值 value 和匹配指标的 unit，不使用字符串列表。SAFETY_CONSTRAINT_FORMAT_EXAMPLE 仅是结构示例，不能成为新增物理阈值的依据；保留已有安全合同，根据当前任务证据记录初次设计假设，证据不足时报告缺口，不补造阈值。

初次 TaskSpec 的物理阈值根据明确用户要求、任务意图和当前环境证据给出；未明确的设定标为设计假设，不以历史失败实验作为新要求的来源。收到 fixed_task_spec 或 immutable_task 时改为完整继承，不重新解释阈值。ALLOW_BUDGETED_EXPLORATION 只允许预算内仿真探索，不代表物理验证通过。探针失败不证明不可学习；保留证据和风险。

按机器人和支撑语义选择实现。前足支撑不等于前足离地；前腿/后腿站立行走应采用当前注册的对应支撑与姿态门控行走项，避免未门控速度奖励绕过姿态要求。velocity_frame 继承 TaskSpec；倒立前进按 heading 水平航向语义。跟踪得分与实际 m/s 速度指标不能混用单位，若需要 front_leg_forward_speed 等指标必须核对当前清单。

不要预先为任意动态动作指定一套起跳—滚转—恢复配方。仅有 roll_rotation 角速度信号时，不得声称 final_roll_angle 净滚转目标已经被覆盖。LOCAL_METRIC_REWARD_SIGNALS 中空列表明确表示当前缺少已登记的目标训练信号，按共享规则报告 [CAPABILITY_GAP]。

TASK_INTENT_SPEC 的动作与禁止行为必须完整保留。地面侧滚不得改写为腾空侧空翻，腾空侧翻也不得用贴地翻身替代。若计划机体接触与现有全非足端禁触合同冲突，报告能力/接触合同缺口；不得为了生成可训练候选而删除硬约束或偷偷添加其他动作阶段。
前翻是绕机体横向轴的前向旋转，侧翻是绕纵向轴的滚转。net_roll_progress、roll_rotation、final_roll_angle 只支持 roll 轴，不能用于前翻完成度。当前清单支持时，腾空前翻使用 net_pitch_progress、pitch_rotation、front_flip_recovery，保留起跳与全足腾空门控；必选验收包含 final_pitch_angle（rad 正向下界）、pitch_axis_error（无量纲上界）、landing_stability 和 stable_stand_duration。所有前翻门控项在学习课程中持续启用，动作阶段由实际接触函数控制。不要使用普通 abs(pitch)/abs(roll) 跌倒终止切断合法前翻，独立禁触、关节、力矩硬约束继续保留。未明确的圈数、恢复时长和轴向误差阈值标为设计假设。不得把普通 pitch 姿态上界或欧拉 pitch 的简单累计当成完整前翻进度。

只返回符合 OUTPUT_JSON_SCHEMA 的严格 JSON，不返回 Markdown 或 JSON 外说明。

当前本地前/后腿支撑审查要求在清单支持时同时列出对应的 front_leg_stand/front_leg_walk 或 rear_leg_stand/rear_leg_walk。仅要求站立时不要因此增加行走验收或非零行走命令，按动作目标设置有效权重；不能用水平 orientation 与倒立目标对抗。
