from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field


class EnvironmentVariable(BaseModel):
    name: str
    source_file: str
    source_symbol: str
    shape: List[Any]
    unit: str
    coordinate_frame: str
    normalized: bool
    available_to_policy: bool
    available_to_reward: bool
    simulation_only: bool
    derivation: str = "direct"


class RewardRegistryItem(BaseModel):
    name: str
    implementation: str
    config_key: str
    parameters: dict = Field(default_factory=dict)
    expected_raw_range: List[float]
    default_weight: float
    sign: str
    dependencies: List[str]
    supported_phases: List[str] = Field(default_factory=lambda: ["all"])


class CapabilityManifest(BaseModel):
    project: str
    robot: str
    robots: List[str]
    observations: List[EnvironmentVariable]
    reward_variables: List[EnvironmentVariable]
    rewards: List[RewardRegistryItem]
    terminations: List[str]
    command_space: List[str]
    checkpoint_format: str
    logger: str
    training_entry: str
    evaluation_entry: str
    evaluation_metrics: List[str] = Field(default_factory=list)
    unsupported: List[str] = Field(default_factory=list)
