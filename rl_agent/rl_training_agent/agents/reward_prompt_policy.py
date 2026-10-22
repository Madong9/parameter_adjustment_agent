"""初次设计、兼容入口与失败重设计共用的运行时约束。"""
import json
from pathlib import Path

from ..rewards.quality_gate import METRIC_SIGNALS, SAFETY_SIGNALS, METRIC_SIGNAL_OVERRIDES
from ..training.config_runtime import REWARD_PARAMETER_NAMES, COMMAND_RANGE_NAMES
from ..training.contact_semantics import (CONTACT_SEMANTICS_VERSION, COLLISION_FORCE_THRESHOLD,
                                         BODY_TERMINATION_FORCE_THRESHOLD, CONTACT_FORCE_METRICS)
from ..rewards.front_flip_baseline import FRONT_FLIP_OBJECTIVES, FRONT_FLIP_WEIGHTS

REWARD_PROMPT_VERSION = 'reward-design-v12'


def render_reward_policy():
    common = (Path(__file__).parents[1] / 'prompts' / 'reward_design_common.md').read_text(encoding='utf-8')
    policy = {
        'version': REWARD_PROMPT_VERSION,
        'safety_metric_rewards': SAFETY_SIGNALS,
        'contact_metric_units': dict(
            {name: ['N'] for name in sorted(CONTACT_FORCE_METRICS)},
            forbidden_collisions=['count', 'frames']),
        'metric_signal_overrides': [dict(metric=metric, aggregation=aggregation, reward_terms=sorted(terms))
                                   for (metric, aggregation), terms in sorted(METRIC_SIGNAL_OVERRIDES.items())],
        'runtime_parameter_names': sorted(REWARD_PARAMETER_NAMES),
        'command_range_names': sorted(COMMAND_RANGE_NAMES),
        'contact_semantics': {'version': CONTACT_SEMANTICS_VERSION, 'scope': 'all_non_foot_rigid_bodies',
                             'collision_force_threshold_n': COLLISION_FORCE_THRESHOLD,
                             'body_termination_force_threshold_n': BODY_TERMINATION_FORCE_THRESHOLD},
        'only_positive_rewards': False,
        'known_term_semantics': {
            'net_pitch_progress': 'xyzw 四元数相邻控制步机体横轴增量；正向为前翻，只累计首次从全足支撑开始的腾空区间（含离地与首次落地边界），正收益封顶、倒退扣分、无重复起跳叠加，net_pitch_target 为正弧度目标',
            'pitch_rotation': '全足腾空时跟踪正向机体横轴角速度，并抑制纵轴和竖轴角速度，pitch_rate_target/pitch_rate_sigma 为正数',
            'front_flip_recovery': '首个腾空动作完成正向净旋转目标后，全足落地、低姿态误差与低速度；net_pitch_target 必须与净旋转项一致',
            'net_roll_progress': '展开 roll 的净进度，正增益只奖励历史最高进度增加，目标后封顶，倒退扣分；只在全足腾空时发放，按 dt 归一化，net_roll_target 为正弧度目标',
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
        'front_flip_contract': {
            'rotation_axis': 'body_y', 'direction': 'positive', 'quaternion_order': 'xyzw',
            'required_metrics': ['final_pitch_angle', 'pitch_axis_error', 'landing_stability', 'stable_stand_duration'],
            'required_rewards': ['takeoff_velocity', 'airborne_duration', 'pitch_rotation', 'net_pitch_progress', 'front_flip_recovery'],
            'objective_reward_terms': FRONT_FLIP_OBJECTIVES,
            'baseline_weights': FRONT_FLIP_WEIGHTS,
            'active_phases': ['all'],
            'parameter_location': 'net_pitch_target 放在 net_pitch_progress 和 front_flip_recovery 的 terms.parameters，并使用相同数值；不要放在 curriculum.parameter_changes',
            'execution_gating': '起跳、腾空、旋转和落地是动作内运行时接触门控，不是迭代课程名；每个候选须完整包含全部 required_rewards 和持续有效的安全负奖励',
            'pitch_axis_error': '首个有效腾空区间的非横轴旋转长度/总旋转长度，无量纲；必须给出小于1的必选上界',
            'final_pitch_angle': '从首次全足支撑后的单个腾空区间及其离地/落地边界累计有符号横轴旋转；没有完整落地、多次起跳为0，缺四元数/重置/欠采样为未知',
            'termination_conflicts': ['ordinary abs(pitch)/abs(roll) fall termination'],
            'validation_scope': '已实现可执行信号与数值验收；不代表动作已学会或真实物理验证通过',
        },
    }
    return '\n'.join([common, 'REWARD_POLICY_VERSION: ' + REWARD_PROMPT_VERSION,
        'LOCAL_METRIC_REWARD_SIGNALS: ' + json.dumps({k: sorted(v) for k, v in METRIC_SIGNALS.items()}, ensure_ascii=False),
        'RUNTIME_REWARD_POLICY: ' + json.dumps(policy, ensure_ascii=False, sort_keys=True)])
