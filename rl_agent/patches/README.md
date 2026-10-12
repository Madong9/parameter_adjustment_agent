# Unitree RL Gym 补丁

本目录保存 Agent 运行所需、尚未包含在 Unitree 上游仓库中的定制修改。

在 `rl_agent` 与 `unitree_rl_gym` 位于同一父目录时执行：

```bash
git -C ../unitree_rl_gym apply --check ../rl_agent/patches/unitree_rl_gym.patch
git -C ../unitree_rl_gym apply ../rl_agent/patches/unitree_rl_gym.patch
git -C ../unitree_rl_gym apply --check ../rl_agent/patches/unitree_numpy_compat.patch
git -C ../unitree_rl_gym apply ../rl_agent/patches/unitree_numpy_compat.patch
```

补丁基于 Unitree `unitree_rl_gym` 的提交 `276801e`，主要增加：

- 离屏摄像机录制支持；
- 原始及加权奖励统计；
- 跳跃、同步触地、落地稳定和水平漂移奖励；
- 前/后腿站立与行走任务奖励。

`unitree_numpy_compat.patch` 仅将上游 `setup.py` 的过时精确依赖 `numpy==1.20` 改为与 Agent 一致的 `numpy>=1.20,<2`，不会安装或降级任何依赖。本项目 Python 3.8 的当前验证版本是 `1.24.4`，训练入口会补齐旧 Isaac Gym 所需的标量别名。

已安装过 Unitree 时，应用补丁后需要刷新其安装元数据（不修改其余依赖）：

```bash
python -m pip install --no-deps --no-build-isolation -e ../unitree_rl_gym
```

补丁只应用一次；若源码已修复，`apply --check` 会提示不能再次应用。
