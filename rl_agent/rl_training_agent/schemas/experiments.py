from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ExperimentManifest(BaseModel):
    experiment_id: str
    parent_experiment_id: Optional[str] = None
    task_id: str
    git_commit: str = "unknown"
    config_hash: str
    reward_version: int
    seed: int
    robot: str
    training_command: List[str]
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    iteration: int = 0
    checkpoint: Optional[str] = None
    provider_status: str = "unknown"
    training_result: str = "pending"
    failure_reason: Optional[str] = None


class ProviderHealth(BaseModel):
    available: bool
    opencli_available: bool
    extension_connected: bool
    chatgpt_logged_in: bool
    image_upload_supported: bool
    recoverable: bool = True
    details: List[str] = Field(default_factory=list)


class ConversationHandle(BaseModel):
    conversation_id: str
    title_hint: str
    owned: bool = False
