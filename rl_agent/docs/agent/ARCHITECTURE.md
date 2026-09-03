# 系统架构

命令层创建 `Settings`、推理 Provider 和 `TrainingOrchestrator`。编排器首先运行 `EnvironmentInspector`，用实际环境张量校验 Provider 返回的 `TaskSpec`，再通过 `RewardCompiler` 编译各个 `RewardPlan`。第一版默认只允许使用已注册奖励。

`UnitreeProjectAdapter` 是唯一的命令构造器，只为受限真实训练包装器和 rollout 包装器生成参数数组。`ProcessManager` 使用 `shell=False`、独立进程组、有限等待、PID/日志/退出码文件和进程组终止机制。系统不提供通用 shell 执行接口。

证据处理链如下：

```text
轨迹 + 奖励 + 仿真器 MP4
  -> 基于轨迹的 EventDetector
  -> 干净接触图与标注接触图
  -> 纯视觉报告
  -> 确定性安全/任务评估
  -> 融合后的 TrainingDiagnosis
  -> 受限 RewardPlanReviser
  -> 新奖励版本编译与多种子续训/重训
  -> 下一轮 rollout 与联合验收
  -> 完成、预算耗尽或真实阻塞
```

上述链路由 `TrainingOrchestrator` 循环执行。诊断输入不仅包含视觉和 PPO 结果，还包含当前任务、奖励计划、注册奖励白名单、可计算指标、逐项奖励均值、历史轮次和剩余预算。`RewardPlanReviser` 只接受结构化的 add/remove/update 和课程修改；修改后重新触发 Pydantic、奖励符号、权重上限、任务指标覆盖和编译校验。`continue` 可以保持奖励版本不变直接追加训练，`restart` 从随机初始化开始，`continue_from_parent` 可回退到父 checkpoint。

确定性评估器是完成状态的最终裁决者。只有任务指标、硬安全约束和视觉对齐在同一轮全部通过时才进入 `COMPLETED`；普通未达标继续自动闭环，Provider 不可用、不可计算需求、安全修订失败、真实歧义或预算耗尽才进入 `HUMAN_REVIEW`。

`PersistentStateMachine` 会原子写入每次状态变化。`ExperimentStore` 校验路径，并使用 `fcntl` 文件锁串行化写入。重启、回滚或奖励修订不会删除父实验和原 checkpoint。

生产推理由 `OpenCLIChatGPTWebProvider` 完成；`MockLLMReasoningProvider` 只有在显式选择时才可使用，主要服务于测试和 dry-run。
