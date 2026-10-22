"""共享指标单位的等价拼写，不改变指标数值或物理维度。"""
from __future__ import annotations


DIMENSIONLESS_UNITS = frozenset(('', '1', 'ratio', 'dimensionless', 'unitless', '无量纲'))


def is_dimensionless_unit(unit: str) -> bool:
    """只识别无量纲单位别名；百分比需要换算，不能当作比例直接接受。"""
    return unit.strip().lower() in DIMENSIONLESS_UNITS
