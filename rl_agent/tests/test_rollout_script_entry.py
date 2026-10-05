"""Rollout is launched by file path, without a Python package context."""
import ast
import os
from pathlib import Path
import subprocess
import sys


def test_rollout_file_entry_loads_contact_semantics_without_isaacgym():
    root = Path(__file__).resolve().parents[1]
    script = root / 'rl_training_agent' / 'training' / 'real_rollout.py'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(
        [sys.executable, str(script), '--help'], cwd=str(root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '--checkpoint' in result.stdout
    # A deferred relative import would pass --help but crash after loading the
    # checkpoint. Every import in this executable must work with __package__=None.
    tree = ast.parse(script.read_text(encoding='utf-8'))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(node.level == 0 for node in imports)
    assert any(node.module == 'rl_training_agent.training.contact_semantics'
               for node in tree.body if isinstance(node, ast.ImportFrom))
