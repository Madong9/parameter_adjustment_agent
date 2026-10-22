"""奖励修复只返回绑定原始候选的字段补丁。"""
from typing import Any, List, Literal, Union

from pydantic import BaseModel, Field, StrictInt, StrictStr, conint, constr, root_validator, validator


class RewardFieldPatch(BaseModel):
    op: Literal['add', 'replace', 'remove']
    path: List[Union[StrictStr, StrictInt]] = Field(..., min_items=1, max_items=32)
    value: Any = None

    @root_validator(pre=True)
    def explicit_value(cls, values):
        if values.get('op') in ('add', 'replace') and 'value' not in values:
            raise ValueError('add/replace patch must include value')
        return values

    @validator('path')
    def nonnegative_indices(cls, value):
        if any(type(part) is int and part < 0 for part in value):
            raise ValueError('negative patch indices are forbidden')
        return value

    class Config:
        extra = 'forbid'


class RewardCandidatePatch(BaseModel):
    candidate_index: conint(strict=True, gt=0)
    base_sha256: constr(strict=True, regex=r'^[0-9a-f]{64}$')
    patches: List[RewardFieldPatch] = Field(..., max_items=100)

    class Config:
        extra = 'forbid'


class RewardRepairBundle(BaseModel):
    repairs: List[RewardCandidatePatch] = Field(..., max_items=20)

    class Config:
        extra = 'forbid'


class RewardSchemaFieldRepair(BaseModel):
    """尚未通过 RewardPlan 类型校验时，按精确错误位置修复原始 JSON。"""
    base_sha256: constr(strict=True, regex=r'^[0-9a-f]{64}$')
    patches: List[RewardFieldPatch] = Field(..., max_items=100)

    class Config:
        extra = 'forbid'
