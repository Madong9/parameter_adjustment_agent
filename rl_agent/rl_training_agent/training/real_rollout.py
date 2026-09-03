"""使用离屏仿真摄像机的确定性 Isaac Gym rollout 录制器。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

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
    foot_index_values = {int(value) for value in env.feet_indices.detach().cpu().tolist()}
    nonfoot_indices = torch.tensor(
        [index for index in range(env.num_bodies) if index not in foot_index_values],
        dtype=torch.long, device=env.device)
    steps = max(1, int(args.seconds / policy_dt))
    rows, reward_rows = [], []
    obs = env.get_observations()
    video_frame = 0
    for step in range(steps):
        with torch.inference_mode():
            actions = policy(obs.detach())
        obs, _, rewards, dones, infos = env.step(actions.detach())
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        if step % video_interval != 0:
            continue
        env.gym.fetch_results(env.sim, True)
        base = env.base_pos[0].detach().cpu().tolist()
        yaw = float(env.rpy[0, 2])
        for name, handle in cameras.items():
            offset = camera_offsets[name]
            position, target = _tracked_camera_pose(base, yaw, offset)
            env.gym.set_camera_location(
                handle, env.envs[0], gymapi.Vec3(*position), gymapi.Vec3(*target))
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        for name, handle in cameras.items():
            rgba = env.gym.get_camera_image(env.sim, env.envs[0], handle, gymapi.IMAGE_COLOR)
            rgba = np.asarray(rgba).reshape(height, width, 4)
            writers[name].write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
        feet = rigid[0, env.feet_indices, :]
        contacts = (env.contact_forces[0, env.feet_indices, 2] > 1.0)
        contact_flags = [bool(value) for value in contacts.detach().cpu().tolist()]
        # 四足任务使用 FL/FR/RL/RR；人形任务足端较少时补 False，保证通用轨迹表可写。
        contact_flags = (contact_flags + [False, False, False, False])[:4]
        penalized_contact_forces = torch.norm(
            env.contact_forces[0, env.penalised_contact_indices, :], dim=-1)
        penalized_contact_force_max = float(torch.max(penalized_contact_forces)) \
            if penalized_contact_forces.numel() else 0.0
        termination_contact_forces = torch.norm(
            env.contact_forces[0, env.termination_contact_indices, :], dim=-1)
        termination_contact_force_max = float(torch.max(termination_contact_forces)) \
            if termination_contact_forces.numel() else 0.0
        nonfoot_contact_forces = torch.norm(
            env.contact_forces[0, nonfoot_indices, :], dim=-1)
        nonfoot_contact_force_max = float(torch.max(nonfoot_contact_forces)) \
            if nonfoot_contact_forces.numel() else 0.0
        forbidden_contact_force_max = nonfoot_contact_force_max
        row = {
            "sim_time": step * policy_dt, "video_frame": video_frame,
            "base_x": float(env.base_pos[0, 0]), "base_y": float(env.base_pos[0, 1]), "base_z": float(env.base_pos[0, 2]),
            "quat_x": float(env.base_quat[0, 0]), "quat_y": float(env.base_quat[0, 1]),
            "quat_z": float(env.base_quat[0, 2]), "quat_w": float(env.base_quat[0, 3]),
            "roll": float(env.rpy[0, 0]), "pitch": float(env.rpy[0, 1]), "yaw": float(env.rpy[0, 2]),
            "base_vx": float(env.base_lin_vel[0, 0]), "base_vy": float(env.base_lin_vel[0, 1]),
            "base_vz": float(env.base_lin_vel[0, 2]), "angular_velocity": float(torch.norm(env.base_ang_vel[0])),
            "joint_positions": env.dof_pos[0].detach().cpu().tolist(),
            "joint_velocities": env.dof_vel[0].detach().cpu().tolist(),
            "joint_torques": env.torques[0].detach().cpu().tolist(), "actions": env.actions[0].detach().cpu().tolist(),
            "feet_positions": feet[:, :3].detach().cpu().tolist(), "feet_velocities": feet[:, 7:10].detach().cpu().tolist(),
            "contact_fl": contact_flags[0], "contact_fr": contact_flags[1],
            "contact_rl": contact_flags[2], "contact_rr": contact_flags[3],
            "contact_forces": env.contact_forces[0, env.feet_indices, :].detach().cpu().tolist(),
            "command": env.commands[0].detach().cpu().tolist(), "termination_reason": "reset" if bool(dones[0]) else "",
            "penalized_contact_force_max": penalized_contact_force_max,
            "termination_contact_force_max": termination_contact_force_max,
            "nonfoot_contact_force_max": nonfoot_contact_force_max,
            "forbidden_contact_force_max": forbidden_contact_force_max,
            "body_collision": bool(forbidden_contact_force_max > 0.1),
            "fall": bool(abs(float(env.rpy[0, 0])) > 0.8 or abs(float(env.rpy[0, 1])) > 1.0),
            "foot_slip": float(torch.max(torch.norm(feet[:, 7:9], dim=-1) * contacts.float())),
            "energy": float(torch.sum(torch.abs(env.torques[0] * env.dof_vel[0])) * policy_dt),
            "base_speed": float(torch.norm(env.base_lin_vel[0, :2])), "reward_total": float(rewards[0]),
        }
        rows.append(row)
        reward_row = {"sim_time": row["sim_time"], "video_frame": video_frame, "reward_total": row["reward_total"]}
        for reward_name in env.reward_scales.keys():
            raw = env.last_raw_rewards[reward_name]
            reward_row["raw_" + reward_name] = float(raw[0])
            reward_row["weighted_" + reward_name] = float(raw[0] * env.reward_scales[reward_name])
        reward_rows.append(reward_row)
        video_frame += 1
    for writer in writers.values():
        writer.release()
    pd.DataFrame(rows).to_parquet(output / "trajectory.parquet", index=False)
    pd.DataFrame(reward_rows).to_parquet(output / "rewards.parquet", index=False)
    metadata = {"video_fps": recorded_fps, "requested_video_fps": args.fps,
                "frame_count": video_frame, "width": width, "height": height,
                "duration": video_frame / recorded_fps, "simulation_dt": env_cfg.sim.dt,
                "control_decimation": env_cfg.control.decimation, "video_start_sim_time": 0.0,
                "camera": "side", "seed": args.seed, "checkpoint": Path(args.checkpoint).name,
                "camera_tracking": "base_position_and_yaw", "camera_horizontal_fov_deg": 55.0,
                "body_collision_force_threshold_n": 0.1,
                "body_collision_scope": "all_non_foot_rigid_bodies",
                "nonfoot_rigid_body_count": int(nonfoot_indices.numel()),
                "penalized_contact_patterns": list(env.cfg.asset.penalize_contacts_on),
                "termination_contact_patterns": list(env.cfg.asset.terminate_after_contacts_on)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
