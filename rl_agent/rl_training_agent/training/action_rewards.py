"""由 Agent 注册、按真实接触状态门控的动作奖励；不改写 Unitree 源码。"""
from __future__ import annotations

ACTION_REWARDS = {
    "takeoff_velocity": ["base_lin_vel", "base_quat", "contact_forces"],
    "airborne_duration": ["contact_forces"],
    "roll_rotation": ["base_ang_vel", "contact_forces"],
    "landing_recovery": ["contact_forces", "rpy", "base_lin_vel", "base_ang_vel"],
    "stable_support": ["contact_forces", "rpy", "base_lin_vel", "base_ang_vel"],
    "net_roll_progress": ["rpy", "contact_forces"],
}
RUNTIME_SAFETY_REWARDS = {'torque_limits': ['torques', 'torque_limits']}


def _reward_net_roll_progress(self):
    """展开 roll 的净进度，正收益有单次上限，往复和重复起跳不能重复刷分。"""
    import torch
    roll = self.rpy[:, 0]
    if not hasattr(self, '_agent_roll_previous'):
        self._agent_roll_previous = roll.clone()
        self._agent_roll_net = torch.zeros_like(roll)
        self._agent_roll_peak = torch.zeros_like(roll)
    fresh = self.episode_length_buf <= 1
    delta = torch.atan2(torch.sin(roll - self._agent_roll_previous),
                        torch.cos(roll - self._agent_roll_previous))
    previous_progress = torch.clamp(self._agent_roll_net.abs() / self.cfg.rewards.net_roll_target, 0., 1.)
    self._agent_roll_net = torch.where(fresh, torch.zeros_like(roll), self._agent_roll_net + delta)
    self._agent_roll_peak = torch.where(fresh, torch.zeros_like(roll), self._agent_roll_peak)
    progress = torch.clamp(self._agent_roll_net.abs() / self.cfg.rewards.net_roll_target, 0., 1.)
    gain = torch.clamp(progress - self._agent_roll_peak, min=0.)
    backtrack = torch.clamp(progress - previous_progress, max=0.)
    reward = torch.where(fresh, torch.zeros_like(roll), gain + backtrack)
    self._agent_roll_peak = torch.maximum(self._agent_roll_peak, progress)
    self._agent_roll_previous = roll.clone()
    # Unitree 会再乘 dt；抵消该缩放，使每 episode 正向总贡献最多为 term.weight。
    return reward * (~_contacts(self).any(dim=1)).float() / self.dt


def _contacts(env):
    return env.contact_forces[:, env.feet_indices, 2] > 1.0


def _reward_takeoff_velocity(self):
    import torch
    target = self.cfg.rewards.takeoff_velocity_target
    return _contacts(self).any(dim=1).float() * torch.clamp(self.root_states[:, 9] / target, 0., 1.)


def _reward_airborne_duration(self):
    import torch
    airborne = ~_contacts(self).any(dim=1)
    threshold = getattr(self.cfg.rewards, 'min_air_time_threshold', 0.)
    if threshold <= 0:
        return airborne.float()
    if not hasattr(self, '_agent_air_time'):
        self._agent_air_time = torch.zeros_like(self.base_ang_vel[:, 0])
    self._agent_air_time *= self.episode_length_buf > 1
    self._agent_air_time = torch.where(airborne, self._agent_air_time + self.dt,
                                      torch.zeros_like(self._agent_air_time))
    return (airborne & (self._agent_air_time >= threshold)).float()


def _reward_roll_rotation(self):
    import torch
    # 横向翻转需要纵轴角速度，不能用 yaw 跟踪替代。目标有方向且只在腾空时激活。
    error = (self.base_ang_vel[:, 0] - self.cfg.rewards.roll_rate_target) ** 2
    # 不调用有计时副作用的腾空奖励，避免同一步重复累计时长。
    airborne = (~_contacts(self).any(dim=1)).float()
    return airborne * torch.exp(-error / self.cfg.rewards.roll_rate_sigma)


def _reward_landing_recovery(self):
    import torch
    contacts = _contacts(self)
    if not hasattr(self, '_agent_has_flown'):
        self._agent_has_flown = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
    self._agent_has_flown &= self.episode_length_buf > 1
    self._agent_has_flown |= ~contacts.any(dim=1)
    return self._agent_has_flown.float() * _reward_stable_support(self)


def _reward_stable_support(self):
    import torch
    contacts = _contacts(self)
    tilt = (self.rpy[:, :2] ** 2).sum(dim=1)
    motion = (self.root_states[:, 7:10] ** 2).sum(dim=1) + (self.base_ang_vel ** 2).sum(dim=1)
    return contacts.all(dim=1).float() * torch.exp(-4. * tilt - .25 * motion)


def register_action_rewards(env_class):
    """必须在环境初始化、解析 reward_functions 之前注册。"""
    for name in ACTION_REWARDS:
        setattr(env_class, '_reward_' + name, globals()['_reward_' + name])
    from .contact_semantics import collision_count
    env_class._reward_collision = collision_count
