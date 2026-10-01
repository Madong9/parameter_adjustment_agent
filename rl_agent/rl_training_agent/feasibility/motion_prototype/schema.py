"""定义与具体关节角解耦的语义动作阶段。"""
from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class MotionType(str, Enum):
    """用有限类别区分静态姿态与动态/未支持动作。"""

    STATIC_POSE = "STATIC_POSE"
    LOCOMOTION = "LOCOMOTION"
    TURNING = "TURNING"
    JUMP = "JUMP"
    MANIPULATION = "MANIPULATION"
    BALANCE = "BALANCE"
    ACROBATIC = "ACROBATIC"
    UNKNOWN = "UNKNOWN"


class GaitType(str, Enum):
    """枚举有限的四足步态类别；类别标签本身不构成物理验收。"""

    WALK = "WALK"
    TROT = "TROT"
    PACE = "PACE"
    BOUND = "BOUND"
    UNKNOWN = "UNKNOWN"


class GaitPattern(BaseModel):
    """描述周期步态参数；相位偏移是周期归一化值，范围为 [0, 1)。"""

    type: GaitType = GaitType.TROT
    frequency: float = 2.0
    duty_factor: float = 0.5
    phase_offsets: Dict[str, float] = Field(default_factory=lambda: {
        "FL": 0.5, "FR": 0.0, "RL": 0.0, "RR": 0.5,
    })

    class Config:
        """拒绝步态原型中混入关节目标或底层控制输出。"""
        extra = "forbid"

    @staticmethod
    def default_phase_offsets(gait_type: GaitType) -> Dict[str, float]:
        """返回可审计的归一化步态预设相位，不生成关节动作。"""
        presets = {
            GaitType.TROT: {"FL": 0.5, "FR": 0.0, "RL": 0.0, "RR": 0.5},
            GaitType.PACE: {"FL": 0.0, "FR": 0.5, "RL": 0.0, "RR": 0.5},
            GaitType.BOUND: {"FL": 0.0, "FR": 0.0, "RL": 0.5, "RR": 0.5},
            GaitType.WALK: {"FL": 0.0, "FR": 0.5, "RL": 0.75, "RR": 0.25},
        }
        if gait_type not in presets:
            raise ValueError("UNKNOWN gait has no supported phase preset")
        return dict(presets[gait_type])

    @validator("frequency")
    def frequency_is_bounded(cls, value: float) -> float:
        """限制步态频率为有限且适合低成本预检的范围。"""
        if not math.isfinite(float(value)) or not 0.0 < float(value) <= 5.0:
            raise ValueError("gait frequency must be within (0, 5] Hz")
        return float(value)

    @validator("duty_factor")
    def duty_factor_is_bounded(cls, value: float) -> float:
        """要求支撑占空比为有限值且保留实际摆动相。"""
        if not math.isfinite(float(value)) or not 0.1 <= float(value) <= 0.9:
            raise ValueError("gait duty_factor must be within [0.1, 0.9]")
        return float(value)

    @validator("phase_offsets", pre=True)
    def normalize_leg_names(cls, value: Dict[str, float]) -> Dict[str, float]:
        """统一将 HL/HR 别名转换为 Go2 使用的 RL/RR 关节侧名称。"""
        aliases = {"HL": "RL", "HR": "RR"}
        normalized: Dict[str, float] = {}
        for raw_name, offset in dict(value or {}).items():
            name = aliases.get(str(raw_name).upper(), str(raw_name).upper())
            if name in normalized:
                raise ValueError("duplicate gait phase offset for leg %s" % name)
            normalized[name] = float(offset)
        return normalized

    @validator("phase_offsets")
    def phase_offsets_cover_quadruped(cls, value: Dict[str, float]) -> Dict[str, float]:
        """要求四条腿均有有限的归一化步态相位。"""
        expected = {"FL", "FR", "RL", "RR"}
        if set(value) != expected:
            raise ValueError("gait phase_offsets must define FL, FR, RL and RR")
        if not all(math.isfinite(float(item)) and 0.0 <= float(item) < 1.0
                   for item in value.values()):
            raise ValueError("gait phase offsets must be finite values in [0, 1)")
        return {name: float(value[name]) for name in sorted(value)}


class FootTrajectory(BaseModel):
    """描述四足足端周期轨迹的低维参数，不包含关节或执行器命令。"""

    swing_height: float = 0.08
    step_length: float
    step_period: float
    direction_x: float = 1.0
    duty_factor: float = 0.5
    phase_offset: Dict[str, float] = Field(default_factory=dict)

    class Config:
        """拒绝在足端轨迹中混入关节角或力矩。"""
        extra = "forbid"

    @validator("swing_height")
    def swing_height_is_bounded(cls, value: float) -> float:
        """约束摆动高度为有限的机器人尺度内目标。"""
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 0.5:
            raise ValueError("swing_height must be within [0, 0.5] meters")
        return float(value)

    @validator("step_length")
    def step_length_is_bounded(cls, value: float) -> float:
        """拒绝零长度、非有限或明显超出低成本预检范围的步幅。"""
        if not math.isfinite(float(value)) or not 0.0 < float(value) <= 2.0:
            raise ValueError("step_length must be within (0, 2] meters")
        return float(value)

    @validator("step_period")
    def step_period_is_bounded(cls, value: float) -> float:
        """确保足端周期有限且不会导致采样别名。"""
        if not math.isfinite(float(value)) or not 0.1 <= float(value) <= 10.0:
            raise ValueError("step_period must be within [0.1, 10] seconds")
        return float(value)

    @validator("direction_x")
    def direction_is_forward_or_backward(cls, value: float) -> float:
        """编码足端步态相对于 Go2 机身 x 轴的前进或后退方向。"""
        if float(value) not in (-1.0, 1.0):
            raise ValueError("direction_x must be either -1 or 1")
        return float(value)

    @validator("duty_factor")
    def foot_duty_factor_is_bounded(cls, value: float) -> float:
        """保持足端支撑比例与步态协议一致。"""
        if not math.isfinite(float(value)) or not 0.1 <= float(value) <= 0.9:
            raise ValueError("foot trajectory duty_factor must be within [0.1, 0.9]")
        return float(value)

    @validator("phase_offset", pre=True)
    def normalize_foot_phase_names(cls, value: Dict[str, float]) -> Dict[str, float]:
        """将四足相位别名规范到 FL/FR/RL/RR。"""
        aliases = {"HL": "RL", "HR": "RR"}
        normalized = {aliases.get(str(name).upper(), str(name).upper()): float(offset)
                      for name, offset in dict(value or {}).items()}
        if len(normalized) != len(dict(value or {})):
            raise ValueError("duplicate foot trajectory phase offset")
        if normalized and set(normalized) != {"FL", "FR", "RL", "RR"}:
            raise ValueError("phase_offset must define all four legs when provided")
        if any(not math.isfinite(item) or not 0.0 <= item < 1.0
               for item in normalized.values()):
            raise ValueError("foot phase offsets must be finite values in [0, 1)")
        return normalized


class TrajectoryPhaseName(str, Enum):
    """定义跳跃及空翻的语义阶段名称，不代表已有控制轨迹。"""

    PRELOAD = "PRELOAD"
    TAKEOFF = "TAKEOFF"
    FLIGHT = "FLIGHT"
    ROTATION = "ROTATION"
    LANDING = "LANDING"
    RECOVERY = "RECOVERY"


class MotionTrajectoryPhase(BaseModel):
    """描述跳跃/空翻时序阶段及可选高层目标，不含关节控制。"""

    name: TrajectoryPhaseName
    start: float
    end: float
    semantic_goal: str = ""

    @validator("start", "end")
    def phase_time_is_finite(cls, value: float) -> float:
        """要求阶段边界为有限时间。"""
        if not math.isfinite(float(value)):
            raise ValueError("trajectory phase time must be finite")
        return float(value)


class TrajectoryPrototype(BaseModel):
    """为跳跃/特技提供待补全的阶段骨架，而不是物理可执行轨迹。"""

    action: str
    motion_type: MotionType
    duration: float
    phases: List[MotionTrajectoryPhase]
    source: str = "local_phase_template"
    assumptions: List[str] = Field(default_factory=list)

    class Config:
        """避免阶段骨架混入关节角、力矩或策略输出。"""
        extra = "forbid"

    @root_validator
    def phases_are_contiguous_and_match_duration(cls, values):
        """验证阶段时间连续覆盖整段原型。"""
        duration = values.get("duration")
        phases = values.get("phases") or []
        if duration is None or not math.isfinite(float(duration)) or duration <= 0.0:
            raise ValueError("trajectory prototype duration must be positive and finite")
        if not phases or abs(phases[0].start) > 1.0e-6:
            raise ValueError("trajectory prototype must start at time zero")
        for index, phase in enumerate(phases):
            if phase.end <= phase.start:
                raise ValueError("trajectory phase end must be after start")
            if index and abs(phase.start - phases[index - 1].end) > 1.0e-6:
                raise ValueError("trajectory phases must be contiguous")
        if abs(phases[-1].end - float(duration)) > 1.0e-6:
            raise ValueError("trajectory phases must cover duration")
        return values


class VelocityTrajectory(BaseModel):
    """保存单个动作原型的目标机身速度；时间阶段仍由 phases 描述。"""

    duration: float
    target_linear_velocity: Dict[str, float] = Field(default_factory=dict)
    target_angular_velocity: Dict[str, float] = Field(default_factory=dict)

    class Config:
        """保持速度轨迹字段封闭，拒绝关节和执行器命令。"""
        extra = "forbid"

    @validator("duration")
    def velocity_duration_is_bounded(cls, value: float) -> float:
        """保证速度目标时长与短时预检预算相容。"""
        if not math.isfinite(float(value)) or not 0.0 < float(value) <= 30.0:
            raise ValueError("velocity trajectory duration must be within (0, 30] seconds")
        return float(value)

    @validator("target_linear_velocity")
    def linear_velocity_is_low_dimensional(cls, value: Dict[str, float]) -> Dict[str, float]:
        """只允许有限的低维机身线速度分量。"""
        if set(value) - {"x", "y", "z"}:
            raise ValueError("target linear velocity only supports x, y and z")
        if not all(math.isfinite(float(item)) for item in value.values()):
            raise ValueError("target linear velocity must be finite")
        return {str(key): float(item) for key, item in value.items()}

    @validator("target_angular_velocity")
    def angular_velocity_is_low_dimensional(cls, value: Dict[str, float]) -> Dict[str, float]:
        """当前 Unitree 命令接口仅允许声明偏航角速度。"""
        if set(value) - {"yaw"}:
            raise ValueError("target angular velocity only supports yaw")
        if not all(math.isfinite(float(item)) for item in value.values()):
            raise ValueError("target angular velocity must be finite")
        return {str(key): float(item) for key, item in value.items()}


class DynamicMotionPhase(BaseModel):
    """描述动态轨迹中的时间段与低维机身速度命令。"""

    name: str
    start: float
    end: float
    target_velocity: Dict[str, float] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)

    class Config:
        """拒绝动作阶段中出现关节角、力矩或策略输出。"""
        extra = "forbid"

    @validator("start", "end")
    def time_is_finite(cls, value: float) -> float:
        """拒绝无穷时间点。"""
        if not math.isfinite(float(value)):
            raise ValueError("dynamic motion phase time must be finite")
        return float(value)

    @validator("target_velocity")
    def velocity_is_low_dimensional(cls, value: Dict[str, float]) -> Dict[str, float]:
        """只允许有限的机身线速度和偏航角速度，不接受底层控制量。"""
        allowed = {"x", "y", "yaw"}
        if set(value) - allowed:
            raise ValueError("dynamic target velocity only supports x, y and yaw")
        if not all(math.isfinite(float(item)) for item in value.values()):
            raise ValueError("dynamic target velocity must be finite")
        return {str(key): float(item) for key, item in value.items()}


class DynamicMotionPrototype(BaseModel):
    """保存动态动作意图的低维时序轨迹，不包含关节轨迹或控制策略。"""

    robot: str
    action: str
    motion_type: MotionType
    duration: float
    phases: List[DynamicMotionPhase]
    velocity: Optional[VelocityTrajectory] = None
    gait: Optional[GaitPattern] = None
    foot_trajectory: Optional[FootTrajectory] = None
    constraints: List[str] = Field(default_factory=list)
    source: str = "task_intent_rules"
    notes: List[str] = Field(default_factory=list)

    class Config:
        """保持轨迹协议封闭，禁止 LLM 注入未经审查的控制字段。"""
        extra = "forbid"

    @validator("duration")
    def duration_is_bounded(cls, value: float) -> float:
        """限制单次低成本可行性轨迹的最大时长。"""
        if not math.isfinite(value) or value <= 0.0 or value > 30.0:
            raise ValueError("dynamic trajectory duration must be within (0, 30] seconds")
        return float(value)

    @validator("phases")
    def phases_cover_duration(cls, value: List[DynamicMotionPhase], values) -> List[DynamicMotionPhase]:
        """检查阶段按时间排序、连续覆盖且不越出轨迹时长。"""
        duration = values.get("duration")
        if not value:
            raise ValueError("dynamic motion prototype requires at least one phase")
        ordered = sorted(value, key=lambda item: item.start)
        if ordered != value or abs(value[0].start) > 1.0e-6:
            raise ValueError("dynamic motion phases must start at zero and be ordered")
        for index, phase in enumerate(value):
            if phase.end <= phase.start:
                raise ValueError("dynamic motion phase end must be after start")
            if duration is not None and phase.end > float(duration) + 1.0e-6:
                raise ValueError("dynamic motion phase exceeds trajectory duration")
            if index and abs(phase.start - value[index - 1].end) > 1.0e-6:
                raise ValueError("dynamic motion phases must be contiguous")
        if duration is not None and abs(value[-1].end - float(duration)) > 1.0e-6:
            raise ValueError("dynamic motion phases must cover the full trajectory")
        return value

    @root_validator
    def locomotion_requires_gait_and_velocity(cls, values):
        """防止没有显式周期步态或速度目标的 LOCOMOTION 被当作可验证原型。"""
        if values.get("motion_type") != MotionType.LOCOMOTION:
            return values
        gait = values.get("gait")
        velocity = values.get("velocity")
        if gait is None or gait.type == GaitType.UNKNOWN:
            raise ValueError("LOCOMOTION prototype requires a known gait pattern")
        if velocity is None:
            raise ValueError("LOCOMOTION prototype requires a velocity trajectory")
        duration = values.get("duration")
        if duration is not None and abs(velocity.duration - float(duration)) > 1.0e-6:
            raise ValueError("velocity trajectory duration must match prototype duration")
        phases = values.get("phases") or []
        moving_phases = [phase for phase in phases if phase.name == "locomotion"]
        if len(moving_phases) != 1:
            raise ValueError("LOCOMOTION prototype requires one locomotion phase")
        target_x = velocity.target_linear_velocity.get("x")
        phase_x = moving_phases[0].target_velocity.get("x")
        if target_x is None or phase_x is None or abs(target_x - phase_x) > 1.0e-6:
            raise ValueError("velocity trajectory and locomotion phase x targets must match")
        return values


class TargetPose(BaseModel):
    """表示可选的末端笛卡尔目标；不允许在语义原型中给出关节角。"""

    frame: str
    position: List[float]
    quaternion_xyzw: Optional[List[float]] = None

    class Config:
        """拒绝未声明字段以避免关节角混入末端目标。"""
        extra = "forbid"

    @validator("position")
    def position_has_three_values(cls, value: List[float]) -> List[float]:
        """保证末端位置恰有三个有限笛卡尔坐标。"""
        if len(value) != 3 or not all(math.isfinite(float(item)) for item in value):
            raise ValueError("target position must contain exactly three values")
        return value

    @validator("quaternion_xyzw")
    def quaternion_has_four_values(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        """保证可选姿态四元数格式正确。"""
        if value is not None and (len(value) != 4 or
                                  not all(math.isfinite(float(item)) for item in value)):
            raise ValueError("target quaternion must contain exactly four values")
        return value


class ManipulationPrototype(BaseModel):
    """描述末端执行器任务目标；只含笛卡尔语义，不含关节命令。"""

    robot: str
    action: str
    end_effector_frame: str
    target: TargetPose
    duration_seconds: float = 1.0
    grasp_required: bool = False
    approach_distance: Optional[float] = None
    source: str = "task_intent"
    assumptions: List[str] = Field(default_factory=list)

    class Config:
        """拒绝未知字段，尤其是 joint angle、torque 和 actuator command。"""

        extra = "forbid"

    @validator("duration_seconds")
    def manipulation_duration_is_valid(cls, value: float) -> float:
        """限制操作原型时长为有限且正值。"""
        if not math.isfinite(float(value)) or not 0.0 < float(value) <= 30.0:
            raise ValueError("manipulation duration must be within (0, 30] seconds")
        return float(value)

    @validator("approach_distance")
    def approach_distance_is_valid(cls, value: Optional[float]) -> Optional[float]:
        """校验可选接近距离，不替模型或用户猜测默认值。"""
        if value is not None and (not math.isfinite(float(value)) or
                                  not 0.0 <= float(value) <= 2.0):
            raise ValueError("approach_distance must be within [0, 2] meters")
        return value


class RobotMotionTarget(BaseModel):
    """表达时间点上的脚端笛卡尔位置和躯干目标，不包含任何关节角。"""

    time_seconds: float = 0.0
    feet: Dict[str, List[float]] = Field(default_factory=dict)
    base_height: float
    base_orientation_xyzw: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])

    class Config:
        """拒绝在机器人目标中混入模型未声明的关节或控制量。"""
        extra = "forbid"

    @validator("feet")
    def foot_targets_are_xyz(cls, value: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """要求每个脚端目标均为有限笛卡尔三维坐标。"""
        import math
        if not value:
            raise ValueError("robot motion target requires at least one foot target")
        for name, xyz in value.items():
            if not name or len(xyz) != 3 or not all(math.isfinite(float(item)) for item in xyz):
                raise ValueError("foot target %s must contain three finite coordinates" % name)
        return value

    @validator("base_orientation_xyzw")
    def base_orientation_is_quaternion(cls, value: List[float]) -> List[float]:
        """校验躯干朝向为四元素有限四元数。"""
        import math
        if len(value) != 4 or not all(math.isfinite(float(item)) for item in value):
            raise ValueError("base orientation must contain four finite quaternion values")
        if sum(float(item) * float(item) for item in value) <= 1.0e-12:
            raise ValueError("base orientation quaternion must not be zero")
        return value

class MotionPhase(BaseModel):
    """表示一个高层动作阶段，而非底层控制策略。"""

    name: str
    duration_seconds: float
    body_goal: Dict[str, str] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    target_poses: List[TargetPose] = Field(default_factory=list)
    robot_targets: List[RobotMotionTarget] = Field(default_factory=list)

    class Config:
        """拒绝模型扩展出未审查的底层控制字段。"""
        extra = "forbid"

    @validator("duration_seconds")
    def duration_is_positive(cls, value: float) -> float:
        """拒绝非正或过长阶段，限制预检开销。"""
        if not math.isfinite(value) or value <= 0.0 or value > 30.0:
            raise ValueError("motion phase duration must be within (0, 30] seconds")
        return value


class MotionPrototype(BaseModel):
    """保存动作名称、机器人和高层阶段序列。"""

    robot: str
    action: str
    phases: List[MotionPhase]
    source: str = "llm_semantic"
    notes: List[str] = Field(default_factory=list)

    class Config:
        """使语义原型的顶层 JSON Schema 严格封闭。"""
        extra = "forbid"

    @validator("phases")
    def phases_are_nonempty_and_bounded(cls, value: List[MotionPhase]) -> List[MotionPhase]:
        """要求至少一个阶段并将短时预检的总时长限制为 30 秒。"""
        if not value:
            raise ValueError("motion prototype requires at least one phase")
        total = sum(item.duration_seconds for item in value)
        if not math.isfinite(total) or total > 30.0:
            raise ValueError("motion prototype duration cannot exceed 30 seconds")
        return value
