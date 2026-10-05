"""初次设计、兼容入口与失败重设计共用的运行时约束。"""
import json
from pathlib import Path

from ..rewards.quality_gate import METRIC_SIGNALS, SAFETY_SIGNALS, METRIC_SIGNAL_OVERRIDES
from ..training.config_runtime import REWARD_PARAMETER_NAMES, COMMAND_RANGE_NAMES
from ..training.contact_semantics import (CONTACT_SEMANTICS_VERSION, COLLISION_FORCE_THRESHOLD,
                                         BODY_TERMINATION_FORCE_THRESHOLD)

REWARD_PROMPT_VERSION = 'reward-design-v6'


def render_reward_policy():
    common = (Path(__file__).parents[1] / 'prompts' / 'reward_design_common.md').read_text(encoding='utf-8')
    policy = {
        'version': REWARD_PROMPT_VERSION,
        'safety_metric_rewards': SAFETY_SIGNALS,
        'metric_signal_overrides': [dict(metric=metric, aggregation=aggregation, reward_terms=sorted(terms))
                                   for (metric, aggregation), terms in sorted(METRIC_SIGNAL_OVERRIDES.items())],
        'runtime_parameter_names': sorted(REWARD_PARAMETER_NAMES),
        'command_range_names': sorted(COMMAND_RANGE_NAMES),
        'contact_semantics': {'version': CONTACT_SEMANTICS_VERSION, 'scope': 'all_non_foot_rigid_bodies',
                             'collision_force_threshold_n': COLLISION_FORCE_THRESHOLD,
                             'body_termination_force_threshold_n': BODY_TERMINATION_FORCE_THRESHOLD},
        'only_positive_rewards': False,
        'known_term_semantics': {
            'roll_rotation': '腾空时的有方向机体纵轴角速度跟踪，不是净滚转角或单次完成奖励',
            'airborne_duration': '全足同时离地；min_air_time_threshold 为连续腾空非负秒数门槛',
            'feet_air_time': '原生步态单足腾空奖励，不对应全足同时腾空验收',
            'stable_support': '全部足端支撑、低姿态误差和低速度；不要求先腾空',
            'landing_recovery': '先腾空后全足稳定支撑；不是完成旋转的证明',
            'torques': '一般力矩消耗代价，不是超限计数',
            'torque_limits': '力矩接近或超过环境限制的代价，对应力矩安全项',
        },
        'capability_gap_output': {'field': 'design_rationale', 'marker': '[CAPABILITY_GAP]',
                                  'objective_reward_terms': [], 'executable': False},
    }
    return '\n'.join([common, 'REWARD_POLICY_VERSION: ' + REWARD_PROMPT_VERSION,
        'LOCAL_METRIC_REWARD_SIGNALS: ' + json.dumps({k: sorted(v) for k, v in METRIC_SIGNALS.items()}, ensure_ascii=False),
        'RUNTIME_REWARD_POLICY: ' + json.dumps(policy, ensure_ascii=False, sort_keys=True)])
