"""使用离屏仿真摄像机的确定性 Isaac Gym rollout 录制器。"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import MethodType

from rl_training_agent.training.contact_semantics import (
    nonfoot_indices as contact_indices,
    contact_audit_metadata,
    forbidden_contact_flags,
)
from rl_training_agent.training.real_train import (
    _ensure_environment_tools_on_path,
    _install_numpy_compatibility_aliases,
)
from rl_training_agent.training.config_runtime import (
    install_runtime_terminations,
    prepare_evaluation_env_config,
)


def parse_args():
    """解析并返回受限命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["go2", "h1", "h1_2", "g1"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--command-x", type=float)
    parser.add_argument("--command-y", type=float)
    parser.add_argument("--command-yaw", type=float)
    return parser.parse_args()


def _gym_args(task, seed):
    """构造单环境确定性 Isaac Gym 参数。"""
    from legged_gym.utils.helpers import get_args
    original = sys.argv
    sys.argv = [original[0], "--task", task, "--seed", str(seed), "--num_envs", "1", "--headless"]
    try:
        return get_args()
    finally:
        sys.argv = original


def _tracked_camera_pose(base, yaw, offset):
    """把机体坐标相机偏移按 yaw 旋转到世界坐标并返回位置与观察点。"""
    cos_yaw, sin_yaw = math.cos(float(yaw)), math.sin(float(yaw))
    world_x = float(base[0]) + float(offset[0]) * cos_yaw - float(offset[1]) * sin_yaw
    world_y = float(base[1]) + float(offset[0]) * sin_yaw + float(offset[1]) * cos_yaw
    position = (world_x, world_y, float(base[2]) + float(offset[2]))
    target = (float(base[0]), float(base[1]), float(base[2]))
    return position, target


def _install_terminal_state_capture(env, collect, on_terminal=None):
    """仅给单环境采集器安装终止前快照，原生重置始终执行且普通 PPO 不受影响。"""
    if env.num_envs != 1:
        raise ValueError('terminal state capture requires a single rollout environment')
    original = env.reset_idx
    env._agent_rollout_terminal_snapshot = None

    def reset_with_terminal_capture(self, env_ids):
        """在奖励已计算而原生物理重置尚未执行时保存同一步证据及视频。"""
        if len(env_ids) == 0 or not bool((env_ids == 0).any()):
            return original(env_ids)
        try:
            snapshot = copy.deepcopy(collect())
            snapshot['trajectory']['terminal_state_preserved'] = True
            self._agent_rollout_terminal_snapshot = snapshot
            if on_terminal is not None:
                on_terminal(snapshot)
        finally:
            result = original(env_ids)
        return result

    env.reset_idx = MethodType(reset_with_terminal_capture, env)


def _sample_after_step(env, collect, done):
    """终止步必须使用重置前快照，不能把新姿态与旧奖励拼成合法采样。"""
    snapshot = env._agent_rollout_terminal_snapshot
    if done:
        if snapshot is None or not snapshot['trajectory'].get('terminal_state_preserved'):
            raise ValueError('rollout terminal step has no synchronized pre-reset snapshot')
        return snapshot
    if snapshot is not None:
        raise ValueError('rollout terminal snapshot disagrees with the returned done flag')
    return collect()


def _sample_current_state(env, rigid, nonfoot_indices, joint_position_limits, joint_torque_limits):
    """从当前物理步统一采集姿态、关节、接触、奖励和终止状态，返回独立标量列表。"""
    import torch
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    feet = rigid[0, env.feet_indices, :]
    done = bool(env.reset_buf[0])
    timed_out = done and bool(env.time_out_buf[0])
    contacts = env.contact_forces[0, env.feet_indices, 2] > 1.0
    contact_flags = (contacts.detach().cpu().tolist() + [False, False, False, False])[:4]

    def maximum_force(indices):
        """读取指定刚体集合的真实最大接触力，空集合保持零。"""
        forces = torch.norm(env.contact_forces[0, indices, :], dim=-1)
        return float(torch.max(forces)) if forces.numel() else 0.0

    collision_flags, forbidden_forces, _ = forbidden_contact_flags(env)
    row = {
        'episode_step': int(env.episode_length_buf[0]), 'terminal_state_preserved': False,
        'base_x': float(env.base_pos[0, 0]), 'base_y': float(env.base_pos[0, 1]), 'base_z': float(env.base_pos[0, 2]),
        'quat_x': float(env.base_quat[0, 0]), 'quat_y': float(env.base_quat[0, 1]),
        'quat_z': float(env.base_quat[0, 2]), 'quat_w': float(env.base_quat[0, 3]),
        'roll': float(env.rpy[0, 0]), 'pitch': float(env.rpy[0, 1]), 'yaw': float(env.rpy[0, 2]),
        'base_vx': float(env.base_lin_vel[0, 0]), 'base_vy': float(env.base_lin_vel[0, 1]),
        'base_vz': float(env.base_lin_vel[0, 2]), 'angular_velocity': float(torch.norm(env.base_ang_vel[0])),
        'base_wx': float(env.base_ang_vel[0, 0]), 'base_wy': float(env.base_ang_vel[0, 1]),
        'base_wz': float(env.base_ang_vel[0, 2]),
        'joint_positions': env.dof_pos[0].detach().cpu().tolist(),
        'joint_velocities': env.dof_vel[0].detach().cpu().tolist(),
        'joint_torques': env.torques[0].detach().cpu().tolist(), 'actions': env.actions[0].detach().cpu().tolist(),
        'joint_position_limits': joint_position_limits,
        'joint_soft_position_limits': env.dof_pos_limits.detach().cpu().tolist(),
        'joint_torque_limits': joint_torque_limits,
        'feet_positions': feet[:, :3].detach().cpu().tolist(), 'feet_velocities': feet[:, 7:10].detach().cpu().tolist(),
        'contact_fl': bool(contact_flags[0]), 'contact_fr': bool(contact_flags[1]),
        'contact_rl': bool(contact_flags[2]), 'contact_rr': bool(contact_flags[3]),
        'contact_forces': env.contact_forces[0, env.feet_indices, :].detach().cpu().tolist(),
        'command': env.commands[0].detach().cpu().tolist(),
        'termination_reason': ('timeout' if timed_out else 'reset') if done else '',
        'penalized_contact_force_max': maximum_force(env.penalised_contact_indices),
        'termination_contact_force_max': maximum_force(env.termination_contact_indices),
        'nonfoot_contact_force_max': maximum_force(nonfoot_indices),
        'forbidden_contact_force_max': float(forbidden_forces[0]),
        'body_collision': bool(collision_flags[0]), 'fall': done and not timed_out,
        'foot_slip': float(torch.max(torch.norm(feet[:, 7:9], dim=-1) * contacts.float())),
        'energy': float(torch.sum(torch.abs(env.torques[0] * env.dof_vel[0])) * env.dt),
        'base_speed': float(torch.norm(env.base_lin_vel[0, :2])), 'reward_total': float(env.rew_buf[0]),
    }
    reward_row = {'reward_total': row['reward_total']}
    for name, scale in env.reward_scales.items():
        raw = env.last_raw_rewards[name]
        reward_row['raw_' + name] = float(raw[0])
        reward_row['weighted_' + name] = float(raw[0] * scale)
    return {'trajectory': row, 'rewards': reward_row}


def main() -> int:
    """解析命令行参数并执行对应的 Agent 工作流。"""
    args = parse_args()
    _install_numpy_compatibility_aliases()
    _ensure_environment_tools_on_path()
    import isaacgym
    from isaacgym import gymapi, gymtorch
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from legged_gym.envs import task_registry

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.test = False
    env_cfg.env.record_video = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.curriculum = False
    prepare_evaluation_env_config(env_cfg, config)
    train_cfg.seed = args.seed
    train_cfg.runner.resume = False
    gym_args = _gym_args(args.task, args.seed)
    env, _ = task_registry.make_env(name=args.task, args=gym_args, env_cfg=env_cfg)
    install_runtime_terminations(env, config)
    runner, _ = task_registry.make_alg_runner(env=env, args=gym_args, train_cfg=train_cfg, log_root=None)
    runner.load(str(Path(args.checkpoint).resolve()), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    width, height = 640, 480
    camera_offsets = {
        "front": (1.4, 0.0, 0.38),
        "side": (0.0, -1.4, 0.38),
        "overview": (1.1, -1.1, 1.15),
    }
    cameras, writers = {}, {}
    properties = gymapi.CameraProperties()
    properties.width, properties.height = width, height
    properties.horizontal_fov = 55.0
    properties.enable_tensors = False
    policy_dt = env.dt
    video_interval = max(1, round(1.0 / (args.fps * policy_dt)))
    recorded_fps = 1.0 / (video_interval * policy_dt)
    initial_base = env.base_pos[0].detach().cpu().tolist()
    initial_yaw = float(env.rpy[0, 2])
    for name, offset in camera_offsets.items():
        handle = env.gym.create_camera_sensor(env.envs[0], properties)
        position, target = _tracked_camera_pose(initial_base, initial_yaw, offset)
        env.gym.set_camera_location(handle, env.envs[0], gymapi.Vec3(*position), gymapi.Vec3(*target))
        cameras[name] = handle
        writer = cv2.VideoWriter(str(output / (name + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"),
                                 recorded_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("could not open MP4 writer")
        writers[name] = writer

    rigid_state = env.gym.acquire_rigid_body_state_tensor(env.sim)
    rigid = gymtorch.wrap_tensor(rigid_state).view(1, -1, 13)
    nonfoot_indices = contact_indices(env)
    steps = max(1, int(args.seconds / policy_dt))
    # 环境 dof_pos_limits 已按奖励 soft_dof_pos_limit 收缩；硬安全限位必须读实际 actor。
    dof_properties = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    joint_position_limits = np.column_stack((dof_properties['lower'], dof_properties['upper'])).tolist()
    joint_torque_limits = dof_properties['effort'].tolist()
    rows, reward_rows = [], []

    def collect_current_sample():
        """采集正常步或重置钩子调用时尚未被覆盖的同一步状态。"""
        return _sample_current_state(env, rigid, nonfoot_indices, joint_position_limits, joint_torque_limits)

    def record_frame(sample):
        """按照当前同步状态定位相机，终止帧在原生重置前完成渲染。"""
        row = sample['trajectory']
        base = [row['base_x'], row['base_y'], row['base_z']]
        for name, handle in cameras.items():
            position, target = _tracked_camera_pose(base, row['yaw'], camera_offsets[name])
            env.gym.set_camera_location(handle, env.envs[0], gymapi.Vec3(*position), gymapi.Vec3(*target))
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        for name, handle in cameras.items():
            rgba = env.gym.get_camera_image(env.sim, env.envs[0], handle, gymapi.IMAGE_COLOR)
            rgba = np.asarray(rgba).reshape(height, width, 4)
            writers[name].write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))

    record_video = False

    def record_terminal_frame(sample):
        """仅在本步应输出视频时保存终止前画面，其余步不额外渲染。"""
        if record_video:
            record_frame(sample)

    _install_terminal_state_capture(env, collect_current_sample, record_terminal_frame)
    override = None if args.command_x is None else (
        float(args.command_x), float(args.command_y or 0.0), float(args.command_yaw or 0.0))
    if override is not None:
        env.commands[0, 0], env.commands[0, 1], env.commands[0, 2] = override
    obs = env.get_observations()
    video_frame = 0
    episode_id = 0
    for step in range(steps):
        record_video = step % video_interval == 0
        env._agent_rollout_terminal_snapshot = None
        with torch.inference_mode():
            actions = policy(obs.detach())
        obs, _, rewards, dones, infos = env.step(actions.detach())
        done = bool(dones[0])
        sample = _sample_after_step(env, collect_current_sample, done)
        if override is not None:
            env.commands[0, 0], env.commands[0, 1], env.commands[0, 2] = override
            obs = env.get_observations()
        frame_index = video_frame if record_video else max(0, video_frame - 1)
        if record_video and not done:
            record_frame(sample)
        row = sample['trajectory']
        row.update(sim_time=step * policy_dt, video_frame=frame_index, episode_id=episode_id)
        rows.append(row)
        reward_row = sample['rewards']
        reward_row.update(sim_time=row['sim_time'], video_frame=frame_index, episode_id=episode_id)
        reward_rows.append(reward_row)
        if done:
            episode_id += 1
        if record_video:
            video_frame += 1
    for writer in writers.values():
        writer.release()
    pd.DataFrame(rows).to_parquet(output / "trajectory.parquet", index=False)
    pd.DataFrame(reward_rows).to_parquet(output / "rewards.parquet", index=False)
    metadata = {"video_fps": recorded_fps, "requested_video_fps": args.fps,
                "trajectory_dt": policy_dt, "trajectory_steps": len(rows),
                "frame_count": video_frame, "width": width, "height": height,
                "duration": video_frame / recorded_fps, "simulation_dt": env_cfg.sim.dt,
                "control_decimation": env_cfg.control.decimation, "video_start_sim_time": 0.0,
                "camera": "side", "seed": args.seed, "checkpoint": Path(args.checkpoint).name,
                "camera_tracking": "base_position_and_yaw", "camera_horizontal_fov_deg": 55.0,
                "body_collision_force_threshold_n": env._agent_contact_contract['collision_force_threshold_n'],
                "body_collision_scope": env._agent_contact_contract['scope'],
                "joint_limit_source": "Isaac Gym actor DOF properties lower/upper",
                "joint_soft_limit_source": "environment dof_pos_limits; reward soft_dof_pos_limit",
                "torque_limit_source": "Isaac Gym actor DOF properties effort",
                "command_override": override,
                "state_sampling": "post_reward_before_native_reset_on_terminal_steps",
                "terminal_frames_preserved": sum(row['terminal_state_preserved'] for row in rows),
                "rollout_episode_count": episode_id + int(not bool(dones[0])),
                "nonfoot_rigid_body_count": int(nonfoot_indices.numel()),
                "penalized_contact_patterns": list(env.cfg.asset.penalize_contacts_on),
                "termination_contact_patterns": list(env.cfg.asset.terminate_after_contacts_on)}
    metadata.update(contact_audit_metadata(env))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
