"""由 Agent 注册、按真实接触状态门控的动作奖励；不改写 Unitree 源码。"""
from __future__ import annotations

ACTION_REWARDS = {
    "takeoff_velocity": ["base_lin_vel", "base_quat", "contact_forces"],
    "airborne_duration": ["contact_forces"],
    "roll_rotation": ["base_ang_vel", "contact_forces"],
    "landing_recovery": ["contact_forces", "rpy", "base_lin_vel", "base_ang_vel"],
}


def _contacts(env):
    return env.contact_forces[:, env.feet_indices, 2] > 1.0


def _reward_takeoff_velocity(self):
    import torch
    target = self.cfg.rewards.takeoff_velocity_target
    return _contacts(self).any(dim=1).float() * torch.clamp(self.root_states[:, 9] / target, 0., 1.)


def _reward_airborne_duration(self):
    return (~_contacts(self).any(dim=1)).float()


def _reward_roll_rotation(self):
    import torch
    # 横向翻转需要纵轴角速度，不能用 yaw 跟踪替代。目标有方向且只在腾空时激活。
    error = (self.base_ang_vel[:, 0] - self.cfg.rewards.roll_rate_target) ** 2
    return _reward_airborne_duration(self) * torch.exp(-error / self.cfg.rewards.roll_rate_sigma)


def _reward_landing_recovery(self):
    import torch
    contacts = _contacts(self)
    if not hasattr(self, '_agent_has_flown'):
        self._agent_has_flown = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
    self._agent_has_flown &= self.episode_length_buf > 1
    self._agent_has_flown |= ~contacts.any(dim=1)
    tilt = (self.rpy[:, :2] ** 2).sum(dim=1)
    motion = (self.root_states[:, 7:10] ** 2).sum(dim=1) + (self.base_ang_vel ** 2).sum(dim=1)
    return self._agent_has_flown.float() * contacts.all(dim=1).float() * torch.exp(-4. * tilt - .25 * motion)


def register_action_rewards(env_class):
    """必须在环境初始化、解析 reward_functions 之前注册。"""
    for name in ACTION_REWARDS:
        setattr(env_class, '_reward_' + name, globals()['_reward_' + name])
