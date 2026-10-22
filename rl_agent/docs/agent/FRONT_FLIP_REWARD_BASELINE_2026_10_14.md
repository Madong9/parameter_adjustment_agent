# 前空翻奖励候选遗漏与本地基线

## 阻塞原因

最新作业 `ec8149d7d24b`、任务 `task-ccd8109bb2`，2026-10-14 本地时间 19:11:53—19:15:39。模型正常返回，动作语义与仿真环境检查已通过，但初次候选和两次修订均未通过奖励审查。PPO 尚未启动。

三个末次候选的实际问题：

- 候选一把起跳、旋转、恢复奖励的 `active_phases` 设为 TAKEOFF/FLIGHT/ROTATION/LANDING/RECOVERY，但迭代课程名为 `full_flip_chain`。运行时因此关闭这些项；还缺少 `airborne_duration`，并将 `net_pitch_target` 放入不支持该字段的课程参数。
- 候选二缺少 `airborne_duration` 和 `front_flip_recovery`，使用不要求完成净旋转的普通 `landing_recovery`。
- 候选三缺少 `pitch_rotation`，又用 `net_pitch_progress` 对应轴向误差，净旋转不能替代非目标轴惩罚。

本地奖励实现与独立验收均已存在，能力缺口报告为空。这次是模型候选配置遗漏；审查拒绝有依据。错误提示已区分“候选没有启用对应奖励”和“确实缺少本地实现”，并列出可用奖励名称。

## 修改后的行为

共享提示词为 `reward-design-v9`，明确每个前空翻候选必须包含五个动作信号、完整对应表和持续有效的安全负奖励。动作内阶段由奖励的运行时接触状态门控，`active_phases` 使用 all；`net_pitch_target` 放在净旋转及恢复项的 `parameters`，保持一致。机器策略包含准确的四项目标对应关系和起始权重，不要求验收指标具有同名奖励。

当全部模型候选和有限修订被拒绝、没有本地能力缺口时，`rewards/front_flip_baseline.py` 可提交一个独立本地基线候选。其范围仅为有完整实现和合法合同的 Go2 单次腾空前空翻；含糊目标、地面前滚翻、其他机型、连续动作、额外速度/支撑要求、范围外必选指标或冲突硬约束均不使用该模板。

| 奖励 | 本地初始权重 |
| --- | ---: |
| takeoff_velocity | 1 |
| airborne_duration | 0.5 |
| pitch_rotation | 1 |
| net_pitch_progress | 3 |
| front_flip_recovery | 2 |
| collision | -2 |
| dof_pos_limits | -1 |
| torque_limits | -1 |

净旋转目标取原始验收角度与 2π 的较大值，恢复和进度使用同一目标；原始验收阈值不变。使用覆盖全程的学习课程，动作门控项全部有效，平面和 yaw 命令为零。权重为待验证的本地起始配置，不保证收敛。

原模型候选继续保留拒绝结果。基线也经过同一个 RewardReviewAgent、实现白名单、目标覆盖、接触和动作合同校验，以及编译后的运行时合同复核；任何校验失败都阻止它进入训练。安全项、冻结任务与预算继续执行。

采用记录写入 `local_reward_baseline.json`，包含模板版本、run_id、任务身份和完整计划；原审查写入 `reward_review_before_baseline.json`，含基线审查写入 `reward_review_with_baseline.json`，事件为 `local_reward_baseline`。新运行清理当前展示记录，保留旧候选计划和 checkpoint。

## 本轮合同验证

只读加载本轮保存的 TaskSpec 与意图，生成本地版本 4：原有审查通过，冲突为空，配置编译和运行时前翻合同复核通过。验证文件位于 `artifacts/front_flip_baseline_validation/`；没有改写旧作业的阻塞结论或冻结合同。

该配置在 RTX 4080、16 个 Go2 仿真环境、1 轮 PPO 下实际运行成功：退出码 0、约 5.63 秒、生成两个 checkpoint、记录指标全部有限、前翻合同安装成功。原生接触、超时及任务硬约束仍生效。启动记录为 `artifacts/front_flip_baseline_validation/startup_report.json`，临时目录 `/tmp/front-flip-baseline-startup-vq6i9hy8`。

该检查证明本地候选能够启动，未执行长训练或证明机器人已完成前空翻。旧作业没有可续训 checkpoint，在上位机重新下发“向前腾空翻跟头”即可使用新提示词及审查后的本地后备候选。

全量 **706 项测试通过，31.79 秒**，Python 编译、JavaScript 语法及 `git diff --check` 通过。新增回归覆盖真实注册表、中文行为对应、冻结合同不变、缺后端/机型不符/额外要求/硬约束冲突拒绝、模型拒绝后进入相同审查与编译、当前展示记录清理和提示词机械约束。
