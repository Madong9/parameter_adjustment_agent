#!/usr/bin/env python3
from pathlib import Path
import sys
import time
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.schemas.experiments import ConversationHandle

# 用法: python scripts/opencli_debug_run.py <task_rollout_dir>
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: opencli_debug_run.py <rollout_visual_provider_records_dir>')
        sys.exit(2)
    record_dir = Path(sys.argv[1])
    record_dir.mkdir(parents=True, exist_ok=True)
    provider = OpenCLIChatGPTWebProvider(record_dir=record_dir)
    try:
        provider.open_or_bind()
        conv = provider.new_conversation('debug-manual')
        prompt = '这是一个用于调试的测试消息，请不要回复。'
        # 强制采集提交前快照
        provider._collect_page_debug_artifacts(conv, 'before_send')
        raw = provider.send_text(prompt, conv, submission_timeout=60)
        print('assistant reply:', raw[:400])
        provider._collect_page_debug_artifacts(conv, 'after_response')
    finally:
        provider.close()
