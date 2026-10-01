"""通过受限 nvidia-smi 查询采集 GPU 利用率和显存快照。"""
from __future__ import annotations

import csv
import io
import subprocess
from typing import Any, Dict


def sample_gpu(gpu_id: int) -> Dict[str, Any]:
    """读取指定 GPU 的利用率、已用显存和总显存，失败时返回错误字段。"""
    command = [
        "nvidia-smi", "--id=%d" % gpu_id,
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=5, check=False)
    except Exception as exc:
        return {"gpu_id": gpu_id, "available": False, "error": str(exc)}
    if result.returncode != 0:
        return {"gpu_id": gpu_id, "available": False, "error": result.stderr.strip()[-500:]}
    row = next(csv.reader(io.StringIO(result.stdout)))
    return {"gpu_id": gpu_id, "available": True, "utilization_percent": float(row[0]),
            "memory_used_mb": float(row[1]), "memory_total_mb": float(row[2])}
