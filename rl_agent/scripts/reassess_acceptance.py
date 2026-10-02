#!/usr/bin/env python3
"""只重算已有轨迹；--apply 审计迁移已知的前腿速度验收错误，不训练或扩大预算。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from rl_training_agent.evaluation.deterministic import DeterministicEvaluator
from rl_training_agent.metrics.trajectory_metrics import TrajectoryMetrics
from rl_training_agent.orchestration.orchestrator import TrainingOrchestrator
from rl_training_agent.schemas.task import TaskSpec
from rl_training_agent.utils.io import read_json, write_json


def reassess(task_dir, apply=False):
    raw = read_json(task_dir / 'task_spec.json')
    contract = read_json(task_dir / 'acceptance_contract.json')
    state = read_json(task_dir / 'state.json')
    if apply and state['state'] != 'HUMAN_REVIEW':
        raise ValueError('只能迁移已停止并等待人工复核的任务')
    for key in ('robot', 'success_metrics', 'safety_constraints'):
        if contract[key] != raw[key]:
            raise ValueError('原始验收合同不一致：' + key)
    if contract.get('run_id') != state.get('context', {}).get('run_id'):
        raise ValueError('验收合同运行身份不一致')
    for key, value in contract['task_identity'].items():
        if value != raw.get(key, 'body' if key == 'velocity_frame' else None):
            raise ValueError('原始任务身份不一致：' + key)
    task = TaskSpec.parse_obj(raw)
    adjustments = TrainingOrchestrator._normalize_task_for_instruction(task)
    summary = read_json(task_dir / 'summary.json')
    root = task_dir / 'candidates' / summary['selected_experiment'] / 'rollouts'
    round_dir = sorted(root.glob('round_*'))[-1]
    records = []
    for path in sorted(round_dir.glob('rollout_*/trajectory.parquet')):
        metrics = TrajectoryMetrics().compute(pd.read_parquet(path))
        records.append({'metrics': metrics})
    if not records:
        raise ValueError('没有完整的已保存轨迹')
    metrics = TrainingOrchestrator._aggregate_rollout_metrics(task, records)
    checks = [DeterministicEvaluator()._check(m, metrics).dict() for m in task.success_metrics]
    report = {'task_id':task.task_id, 'adjustments':adjustments,
              'rollout_count':len(records), 'metrics':metrics, 'success_checks':checks,
              'numeric_task_passed':all(c['passed'] for c in checks),
              'requires_visual_reassessment':True,
              'note':'只重算物理验收；历史视觉和诊断不适用新合同，恢复时须重新评估。没有宣告任务成功或重置预算。'}
    if apply:
        if adjustments:
            audit = task_dir / 'acceptance_migrations' / 'horizontal-speed-v2'
            if (audit / 'before.json').exists():
                raise ValueError('迁移备份已存在，拒绝覆盖')
            updated_contract = dict(contract)
            updated_contract.update({
                'success_metrics':[m.dict() for m in task.success_metrics],
                'task_identity':TrainingOrchestrator._task_acceptance_identity(task),
                'acceptance_version':2})
            write_json(audit / 'before.json', {'task':raw, 'contract':contract, 'summary':summary})
            write_json(audit / 'after.json', {'task':task.dict(), 'contract':updated_contract,
                                            'adjustments':adjustments})
            write_json(task_dir / 'task_spec.json', task)
            write_json(task_dir / 'acceptance_contract.json', updated_contract)
        write_json(task_dir / 'acceptance_reassessment.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_dir', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    report = reassess(args.task_dir, args.apply)
    print(json.dumps({k:report[k] for k in ('task_id','adjustments','rollout_count',
                     'success_checks','numeric_task_passed','requires_visual_reassessment')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
