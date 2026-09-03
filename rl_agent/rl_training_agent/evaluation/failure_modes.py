from enum import Enum


class FailureMode(str, Enum):
    NO_TASK_PROGRESS = "no_task_progress"
    NO_TAKEOFF = "no_takeoff"
    PARTIAL_TAKEOFF = "partial_takeoff"
    FRONT_LEG_ONLY_TAKEOFF = "front_leg_only_takeoff"
    HIND_LEG_ONLY_TAKEOFF = "hind_leg_only_takeoff"
    LEFT_RIGHT_ASYMMETRY = "left_right_asymmetry"
    EXCESSIVE_FORWARD_MOTION = "excessive_forward_motion"
    EXCESSIVE_LATERAL_MOTION = "excessive_lateral_motion"
    AIRBORNE_ROLL_INSTABILITY = "airborne_roll_instability"
    AIRBORNE_PITCH_INSTABILITY = "airborne_pitch_instability"
    BODY_COLLISION = "body_collision"
    HEAD_COLLISION = "head_collision"
    UNSTABLE_LANDING = "unstable_landing"
    FOOT_SLIP = "foot_slip"
    MULTIPLE_UNCONTROLLED_BOUNCES = "multiple_uncontrolled_bounces"
    FALL_AFTER_LANDING = "fall_after_landing"
    UNNATURAL_LEG_MOTION = "unnatural_leg_motion"
    HIGH_FREQUENCY_JITTER = "high_frequency_jitter"
    POOR_FOOT_CLEARANCE = "poor_foot_clearance"
    LATERAL_BODY_OSCILLATION = "lateral_body_oscillation"
    WRONG_STRATEGY = "task_completed_with_wrong_strategy"
    PHASE_ORDER_ERROR = "phase_order_error"
    REWARD_HACKING = "reward_hacking_suspected"
    UNCERTAIN_VISUAL_EVIDENCE = "uncertain_visual_evidence"


REWARD_DIRECTION_HINTS = {
    FailureMode.NO_TASK_PROGRESS: ["increase task progress signal", "check sparse success activation"],
    FailureMode.AIRBORNE_ROLL_INSTABILITY: ["strengthen orientation control during flight"],
    FailureMode.AIRBORNE_PITCH_INSTABILITY: ["strengthen pitch control during flight"],
    FailureMode.FOOT_SLIP: ["penalize trajectory-derived slip"],
    FailureMode.HIGH_FREQUENCY_JITTER: ["strengthen action-rate or acceleration penalty"],
    FailureMode.REWARD_HACKING: ["replace proxy with deterministic task metric"],
}
