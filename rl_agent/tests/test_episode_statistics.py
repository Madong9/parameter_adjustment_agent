"""初始空 episode 不产生无穷极值，已采样 episode 仍按真实数据记录。"""
import pytest

from rl_training_agent.training.reward_statistics import install_observed_episode_statistics


def environment():
    """模拟未采样和采样两次的环境，以及原生清理后的无穷极值哨兵。"""
    torch = pytest.importorskip('torch')
    class Env:
        """只提供统计与物理重置接口。"""

        def reset_idx(self, ids):
            """记录全部重置请求并清理其计数，模拟初始原生日志极值错误。"""
            self.reset_ids = ids.clone()
            self.extras['episode'] = {'rew_raw_min_test': float('inf'), 'rew_raw_max_test': float('-inf')}
            for value in self.reward_step_stats['test'].values():
                value[ids] = 0.

    env = Env()
    values = {'count': [0., 2.], 'raw_sum': [0., 6.], 'raw_square_sum': [0., 20.],
              'raw_min': [float('inf'), 2.], 'raw_max': [float('-inf'), 4.],
              'weighted_sum': [0., -12.], 'weighted_square_sum': [0., 80.],
              'weighted_abs_sum': [0., 12.], 'nonzero_count': [0., 2.], 'saturation_count': [0., 2.]}
    env.reward_step_stats = {'test': {name: torch.tensor(value) for name, value in values.items()}}
    env.reward_config_weights, env.episode_sums = {'test': -2.}, {'test': torch.tensor([0., -12.])}
    env.max_episode_length_s, env.extras = 10., {'episode': {'old': 'empty'}}
    return env


def test_empty_reset_omits_episode_statistics_but_resets_physics():
    """没有样本时不报告零值或无穷极值，实际物理重置照常发生。"""
    import torch
    env = environment()
    install_observed_episode_statistics(env)
    env.reset_idx(torch.tensor([0]))
    assert 'episode' not in env.extras and env.reset_ids.tolist() == [0]


def test_mixed_reset_aggregates_only_observed_samples_and_is_idempotent():
    """空环境不污染已采样环境的均值、标准差和极值，重复安装不叠加包装。"""
    import torch
    env = environment()
    install_observed_episode_statistics(env)
    wrapper = env.reset_idx
    install_observed_episode_statistics(env)
    assert env.reset_idx == wrapper
    env.reset_idx(torch.tensor([0, 1]))
    result = env.extras['episode']
    assert env.reset_ids.tolist() == [0, 1]
    assert result['rew_raw_mean_test'] == 3. and result['rew_raw_std_test'] == 1.
    assert result['rew_raw_min_test'] == 2. and result['rew_raw_max_test'] == 4.
    assert result['rew_weighted_mean_test'] == -6. and result['rew_weighted_std_test'] == 2.
    assert result['rew_test'] == pytest.approx(-1.2)
    assert result['rew_contribution_ratio_test'] == 1.


def test_real_nonfinite_reward_is_not_hidden_as_empty_episode():
    """有采样却产生 NaN 的奖励继续保留未知证据，不能用空 episode 修复掩盖。"""
    import torch
    env = environment()
    env.reward_step_stats['test']['raw_sum'][1] = float('nan')
    install_observed_episode_statistics(env)
    env.reset_idx(torch.tensor([1]))
    assert torch.isnan(env.extras['episode']['rew_raw_mean_test'])
