"""为静态平衡目标生成可审计的笛卡尔候选，不输出关节控制。"""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class StaticPoseCandidate(BaseModel):
    """保存一个静态候选姿态、其语义依据和模型坐标系目标。"""

    candidate_id: str
    support_legs: List[str] = Field(default_factory=list)
    lifted_legs: List[str] = Field(default_factory=list)
    target: Dict[str, Any]
    assumptions: List[str] = Field(default_factory=list)

    class Config:
        """候选只允许高层笛卡尔字段，拒绝混入关节或力矩命令。"""

        extra = "forbid"


class StaticPoseCandidateGenerator:
    """基于真实 Go2 标称 FK 生成少量单足支撑候选。"""

    LEGS = ("FL", "FR", "RL", "RR")

    @classmethod
    def single_leg_candidates(cls, robot_model: Any) -> List[StaticPoseCandidate]:
        """为四个可能的单足支撑腿生成抬起其他三足的笛卡尔候选。"""
        from ..robot_models.go2 import Go2RobotModel

        descriptor = (robot_model.get("_descriptor") if isinstance(robot_model, dict)
                      else robot_model)
        if descriptor is None or str(getattr(descriptor, "robot_name", "")).lower() != "go2":
            return []
        feet = Go2RobotModel.nominal_feet_positions(descriptor, {})
        expected_frames = {leg: [name for name in feet if name.startswith(leg + "_")]
                           for leg in cls.LEGS}
        if any(len(expected_frames[leg]) != 1 for leg in cls.LEGS):
            return []
        lift = float(descriptor.nominal_lift_height)
        candidates: List[StaticPoseCandidate] = []
        for support_leg in cls.LEGS:
            target_feet = {name: list(position) for name, position in feet.items()}
            lifted = [leg for leg in cls.LEGS if leg != support_leg]
            for leg in lifted:
                frame = expected_frames[leg][0]
                target_feet[frame][2] += lift
            candidates.append(StaticPoseCandidate(
                candidate_id="single_support_%s" % support_leg,
                support_legs=[support_leg], lifted_legs=lifted,
                target={
                    "time_seconds": 0.0,
                    "feet": target_feet,
                    "base_height": float(descriptor.base_initial_height),
                    "base_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                assumptions=[
                    "足端位置来自 Go2 URDF 标称关节姿态 Pinocchio FK",
                    "抬脚高度来自 robot_models.yaml nominal_lift_height",
                    "固定基座 IK 不包含躯干平移/旋转与单足接触力分配",
                ],
            ))
        return candidates

    @classmethod
    def leg_pair_candidates(cls, robot_model: Any, support_pair: str
                            ) -> List[StaticPoseCandidate]:
        """为前/后腿支撑生成三档抬脚目标；不改躯干姿态或伪造动力学。"""
        from ..robot_models.go2 import Go2RobotModel

        descriptor = (robot_model.get("_descriptor") if isinstance(robot_model, dict)
                      else robot_model)
        if descriptor is None or str(getattr(descriptor, "robot_name", "")).lower() != "go2":
            return []
        pair = str(support_pair).lower()
        if pair not in ("front", "hind"):
            raise ValueError("support_pair must be front or hind")
        support_legs = ["FL", "FR"] if pair == "front" else ["RL", "RR"]
        lifted_legs = ["RL", "RR"] if pair == "front" else ["FL", "FR"]
        feet = Go2RobotModel.nominal_feet_positions(descriptor, {})
        frame_for_leg = {leg: [name for name in feet if name.startswith(leg + "_")]
                         for leg in cls.LEGS}
        if any(len(frame_for_leg[leg]) != 1 for leg in cls.LEGS):
            return []
        lift = float(descriptor.nominal_lift_height)
        candidates: List[StaticPoseCandidate] = []
        for scale in (0.75, 1.0, 1.25):
            target_feet = {name: list(position) for name, position in feet.items()}
            for leg in lifted_legs:
                target_feet[frame_for_leg[leg][0]][2] += lift * scale
            candidates.append(StaticPoseCandidate(
                candidate_id="support_%s_lift_%s" %
                (pair, str(scale).replace(".", "_")),
                support_legs=list(support_legs), lifted_legs=list(lifted_legs),
                target={
                    "time_seconds": 0.0,
                    "feet": target_feet,
                    "base_height": float(descriptor.base_initial_height),
                    "base_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                assumptions=[
                    "足端位置来自 Go2 URDF 标称姿态 Pinocchio FK",
                    "抬脚高度是 nominal_lift_height 的确定性比例候选",
                    "固定基座模型未搜索躯干姿态、质心迁移或接触力",
                ],
            ))
        return candidates
