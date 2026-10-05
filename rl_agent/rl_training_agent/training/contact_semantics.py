"""训练奖励与安全采集共用的非足端接触定义。"""
COLLISION_FORCE_THRESHOLD = 0.1
BODY_TERMINATION_FORCE_THRESHOLD = 1.0
CONTACT_SEMANTICS_VERSION = 'all-non-foot-v1'


def nonfoot_indices(env):
    import torch
    if hasattr(env, '_agent_nonfoot_indices'):
        return env._agent_nonfoot_indices
    feet = set(int(i) for i in env.feet_indices.detach().cpu().tolist())
    env._agent_nonfoot_indices = torch.tensor([i for i in range(env.num_bodies) if i not in feet],
                                              dtype=torch.long, device=env.device)
    return env._agent_nonfoot_indices


def nonfoot_forces(env):
    import torch
    return torch.norm(env.contact_forces[:, nonfoot_indices(env), :], dim=-1)


def collision_count(env):
    return (nonfoot_forces(env) > COLLISION_FORCE_THRESHOLD).sum(dim=1).float()
