"""只记录确实采样的 episode 奖励统计，空重置不伪造极值或训练数据。"""
from __future__ import annotations

from types import MethodType


def install_observed_episode_statistics(env):
    """保留原生物理重置，只用已采样环境汇总奖励，真实 NaN/Inf 继续暴露。"""
    statistics = getattr(env, 'reward_step_stats', None)
    if not statistics or getattr(env, '_agent_episode_statistics_installed', False):
        return
    import torch
    original = env.reset_idx

    def reset_with_observed_statistics(self, env_ids):
        """先保存真实样本的统计，再正常重置全部请求环境并替换其奖励日志。"""
        if len(env_ids) == 0:
            return original(env_ids)
        # 原生奖励函数每个控制步共同记录，只有各项均有采样的 episode 才可汇总。
        observed = torch.ones(len(env_ids), dtype=torch.bool, device=env_ids.device)
        for stats in self.reward_step_stats.values():
            observed &= stats['count'][env_ids] > 0
        selected = env_ids[observed]
        episode = {}
        if len(selected):
            total = sum(stats['weighted_abs_sum'][selected] for stats in self.reward_step_stats.values())
            for name, stats in self.reward_step_stats.items():
                count = stats['count'][selected]
                raw_mean = stats['raw_sum'][selected] / count
                weighted_mean = stats['weighted_sum'][selected] / count
                raw_std = torch.sqrt(torch.clamp(stats['raw_square_sum'][selected] / count - raw_mean.square(), min=0.))
                weighted_std = torch.sqrt(torch.clamp(stats['weighted_square_sum'][selected] / count - weighted_mean.square(), min=0.))
                values = {'raw_mean': raw_mean.mean(), 'raw_std': raw_std.mean(),
                          'raw_min': stats['raw_min'][selected].min(), 'raw_max': stats['raw_max'][selected].max(),
                          'weight': self.reward_config_weights[name], 'weighted_mean': weighted_mean.mean(),
                          'weighted_std': weighted_std.mean(),
                          'contribution_ratio': (stats['weighted_abs_sum'][selected] / torch.clamp(total, min=1e-12)).mean(),
                          'nonzero_ratio': (stats['nonzero_count'][selected] / count).mean(),
                          'saturation_ratio': (stats['saturation_count'][selected] / count).mean(),
                          'phase_all': raw_mean.mean(), 'raw_episode_sum': stats['raw_sum'][selected].mean()}
                episode['rew_' + name] = self.episode_sums[name][selected].mean() / self.max_episode_length_s
                episode.update({'rew_' + key + '_' + name: value for key, value in values.items()})
        original(env_ids)
        if episode:
            self.extras.setdefault('episode', {}).update(episode)
        else:
            self.extras.pop('episode', None)

    env.reset_idx = MethodType(reset_with_observed_statistics, env)
    env._agent_episode_statistics_installed = True
    if all(not (stats['count'] > 0).any() for stats in statistics.values()):
        env.extras.pop('episode', None)
