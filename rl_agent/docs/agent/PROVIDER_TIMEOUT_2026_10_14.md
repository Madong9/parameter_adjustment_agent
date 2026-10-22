# 奖励设计网页超时与退出状态修复

## 最新作业证据

作业 `14e93c094a5a`、任务 `task-ccd8109bb2`、指令“向前腾空翻跟头”、Go2，2026-10-14 本地时间 15:35:38 开始，15:52:15 退出，退出码 2。

动作语义检查为 `CONSISTENT / AERIAL_FRONT_FLIP`；真实 Isaac Gym 默认姿态健康检查通过，允许 3000 次累计迭代以内的仿真探索。随后奖励设计 Provider 等待约 924.53 秒，记录 `ProviderTimeout`，尚未生成任务/奖励设计结果、冻结预算或启动 PPO。该作业不是 PPO 收敛失败或 GPU 训练进程中断。

超时快照为 `generating=true`，一条本轮助手消息，正文长度 40415。旧诊断未保存 JSON 完整性，因此不能仅凭长度证明回复已完成。Provider 退出后解绑页面；检查会话时只得到 `about:blank`，未读取到该轮实时回复。

## 本地修复

- `providers/chatgpt_dom.py` 共用生成状态探针：只认可见、启用、位于聊天正文以外的停止生成控件。闲置的 `Thinking` / “正在思考”模式按钮、隐藏/禁用按钮及助手正文内的停止文字均不能证明回复仍在生成。快照和独立生成检查使用同一规则。
- 保留本轮消息身份、稳定性、完整 JSON 与真实生成停止的检查，保留 300 秒初始等待和 900 秒总等待上限，不以超时为由重复发送已确认的请求。
- 超时诊断增加 `response_json_complete` 布尔值，不保存聊天正文、完整 HTML 或带查询参数的地址。
- `orchestration/orchestrator.py` 捕获初次奖励设计的 Provider 故障，保存 `blocking_report.json` 并进入 `HUMAN_REVIEW`，标记真实阶段 `REWARD_DESIGNING`、`restart_required=true`、`resume_allowed=false`。沿用任务理解阶段相同处理。本轮没有 checkpoint，恢复方式为检查推理页面后重新下发原始任务。
- `web_ui.py` 对旧的异常退出、未写终态作业只读展示 `FAILED` 和最后实际工作阶段的阻塞状态，不改写历史证据。

这些修复排除了已确认的本地状态识别缺陷；若网页真实生成超过上限，仍会明确阻塞。没有证据证明这轮超时一定由闲置 Thinking 按钮引起，也没有重跑真实模型请求或长训练。

## 使用

在训练使用的 Chrome 中确认 ChatGPT 页面可正常访问并已登录，再在上位机重新下发“向前腾空翻跟头”。每个训练作业启动独立 Python 进程，会读取修复后的 Provider 和编排代码；刷新界面可读取新的前端资源，常驻上位机的 Python 展示逻辑则需重启服务生效。

旧作业仍保留原始超时记录；重新下发是新运行，不冒充旧运行成功或重置已冻结训练预算。

## 验证

新增浏览器 DOM 测试覆盖 13 种生成控件状态及完整本轮 JSON 与闲置 Thinking 共存；奖励设计覆盖新旧接口的超时和人工恢复，验证不创建 checkpoint、不冻结预算、不自动重发；另覆盖只读旧作业展示及诊断隐私。

全量 **687 项通过，31.45 秒**；Python 编译、JavaScript 语法与 `git diff --check` 通过。

## 常驻上位机复检

用户再次报告停止时，整个 `/home/robot/idea_work` 范围内最新作业/日志仍是 `14e93c094a5a`，结束时间仍为 15:52，没有修复后新运行记录。只读查询正在运行的上位机接口，确认它仍返回 `status=failed` 但 `state=REWARD_DESIGNING`，说明常驻 Python 服务未加载展示修复。

确认没有活动训练、排队作业或 Viewer 后，安全重启该项目的旧桌面上位机。新服务实际接口返回 `state=FAILED`、`stage_label=训练失败 · 设计奖励`、奖励步骤 `blocked`，PPO 和联合验收均为 `not_started`、无 checkpoint。新窗口已加载更新后的代码，重启日志位于 `artifacts/ui_jobs/desktop_restart.log`。原始作业状态和失败日志保持原样；没有提交新的模型请求或启动 PPO。
