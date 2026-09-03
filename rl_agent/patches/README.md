# Unitree RL Gym 补丁

本目录保存 Agent 运行所需、尚未包含在 Unitree 上游仓库中的定制修改。

在 `rl_agent` 与 `unitree_rl_gym` 位于同一父目录时执行：

```bash
git -C ../unitree_rl_gym apply --check ../rl_agent/patches/unitree_rl_gym.patch
git -C ../unitree_rl_gym apply ../rl_agent/patches/unitree_rl_gym.patch
```

补丁基于 Unitree `unitree_rl_gym` 的提交 `276801e`，主要增加：

- 离屏摄像机录制支持；
- 原始及加权奖励统计；
- 跳跃、同步触地、落地稳定和水平漂移奖励；
- 前/后腿站立与行走任务奖励。
