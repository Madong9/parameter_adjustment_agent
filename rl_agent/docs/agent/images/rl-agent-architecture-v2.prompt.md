# 架构图 v2 图像生成提示词（对应版本化 Mermaid 源图）

当前仓库的准确、可维护架构图以 `rl-agent-architecture-v2.mmd` 为准。此提示词只作为早期视觉风格参考，角色路由和流程变更时应同步更新 Mermaid 源图。

生成方式：内置图像生成工具。

```text
Use case: infographic-diagram
Asset type: software architecture diagram for a Chinese technical README, landscape 16:9
Primary request: redraw a complete architecture overview for a natural-language-driven quadruped robot reinforcement-learning training Agent. The diagram must clearly show a closed loop and distinguish probabilistic AI agents from deterministic local control.
Scene/backdrop: clean near-black technical canvas with subtle grid, elegant modern engineering dashboard aesthetic
Composition/framing: left-to-right main flow with three horizontal zones and one feedback loop. Zone 1 at top: 用户与上位机. Zone 2 center: 本地确定性协调器. Zone 3 bottom: 外部模型、仿真训练、知识与记忆. Use rounded rectangular modules, thin luminous connecting arrows, ample spacing, no overlapping text.
Required exact Chinese labels, verbatim and legible: 动作输入 / 机器人选择；上位机；本地确定性协调器；任务理解 Agent；GPT / OpenCLI；TaskIntentSpec；Task Feasibility Pipeline；MotionType 分流；静态姿态检查；动态速度轨迹；Pinocchio IK；IsaacGymDynamicValidator；STATIC_PHYSICS_VALIDATED；DYNAMIC_PHYSICS_VALIDATED；TRAINING_READY；能力检查；Unitree Isaac Gym 短时预检；SUPPORTED；CONDITIONAL；UNSUPPORTED；上下文构建器；环境能力 · 机器人配置 · RAG · 四层记忆；提示词编译器；奖励设计 Agent；百炼 GLM；TaskRewardBundle；奖励审查与安全编译；PPO 多种子训练；数值评估器；视觉评估 Agent；诊断与修订 Agent；未通过：继续闭环；通过：COMPLETED；工作记忆；情景记忆；语义记忆；程序记忆；HUMAN_REVIEW。
Main flow: 动作输入 / 机器人选择 -> 上位机 -> 本地确定性协调器 -> 任务理解 Agent (GPT / OpenCLI) -> TaskIntentSpec -> Task Feasibility Pipeline (本地能力检查 -> MotionType 分流: STATIC_POSE uses semantic MotionPrototype -> Pinocchio IK -> static Unitree Isaac Gym validator; LOCOMOTION uses semantic GaitPattern + FootTrajectory -> per-step Cartesian foot targets -> Pinocchio IK -> PPO position-action mapping -> short Unitree Isaac Gym rollout. This IK-derived open-loop reference is not a policy; no PPO runner, no policy generation, no training). Only covered real physics validation can continue; Mock never represents physics evidence. Conditional/unknown motion goes to HUMAN_REVIEW; explicit unsupported hardware requirements go to FAILED. Dynamic TRAINING_READY is added only after dynamic physics validation, compiled reward config and deterministic evaluation metrics all exist. Then context builder (环境能力 · 机器人配置 · RAG · 四层记忆) -> prompt compiler -> reward designer (百炼 GLM) -> TaskRewardBundle -> reward review and safety compile -> PPO multi-seed -> numeric evaluator and visual evaluator -> diagnosis/revision. Diagnosis loops to reward review when needed; accepted tasks reach COMPLETED and memory governance. HUMAN_REVIEW is a separate guarded terminal branch. RAG and memory feed only the context builder, not visual evaluator.
Style/medium: polished vector-like technical infographic, minimalist, high contrast, subtle neon lime and cyan accents, off-white Chinese typography, no gradients behind text
Color palette: charcoal black, graphite, lime green for verified paths, cyan for data paths, amber for HUMAN_REVIEW
Constraints: all Chinese text must be exact and readable; architecture accuracy is more important than decoration; show arrow directions clearly; no logos; no watermark; no English title; no human or robot illustrations
Avoid: garbled Chinese, tiny fonts, 3D perspective, decorative icons that obscure data flow, excessive glow, crossing arrows, crowded boxes
```
