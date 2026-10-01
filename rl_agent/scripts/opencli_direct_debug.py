#!/usr/bin/env python3
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

# 直接使用 opencli 命令与浏览器扩展交互，避免导入项目包

def run_opencli(args, timeout=30, allow_failure=False):
    cmd = ["opencli", "browser"] + args
    try:
        res = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"opencli timeout: {' '.join(cmd[:4])}") from exc
    out = res.stdout or res.stderr
    if res.returncode != 0 and not allow_failure:
        raise RuntimeError(f"opencli failed: {' '.join(cmd)}\n{out}")
    return out

submission_snapshot_script = r"""
(() => {
  const users = Array.from(document.querySelectorAll('[data-message-author-role=user]'))
    .map(element => element.innerText || '').filter(Boolean);
  const assistants = Array.from(document.querySelectorAll('[data-message-author-role=assistant]'))
    .map(element => element.innerText || '').filter(Boolean);
  const composer = document.querySelector(
    '#prompt-textarea, [data-testid="prompt-textarea"], [contenteditable="true"][role="textbox"]');
  const composerText = composer
    ? (composer.isContentEditable ? (composer.innerText || composer.textContent || '')
      : String(composer.value || '')) : '';
  return JSON.stringify({
    latest_user: users.slice(-1)[0] || '',
    user_count: users.length,
    latest_assistant: assistants.slice(-1)[0] || '',
    assistant_count: assistants.length,
    composer_found: Boolean(composer),
    composer_text: composerText,
    url: location.href,
    generating: Boolean(document.querySelector('button[data-testid="stop-button"]'))
  });
})()
"""

latest_assistant_script = "JSON.stringify(Array.from(document.querySelectorAll('[data-message-author-role=assistant]')).map(e=>e.innerText).filter(Boolean).slice(-1)[0]||'')"

send_selectors = [
    'button[data-testid="send-button"]:not([disabled])',
    '#composer-submit-button:not([disabled])',
    'button[aria-label="Send prompt"]:not([disabled])',
    'button[aria-label="发送提示"]:not([disabled])',
    'button[aria-label="发送"]:not([disabled])',
]


def write_debug(base_dir: Path, conv: str, label: str, content: str):
    base = base_dir / conv
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"debug_{label}_{int(time.time())}.json"
    path.write_text(content, encoding='utf-8')
    print('wrote', path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: opencli_direct_debug.py <record_dir>')
        sys.exit(2)
    record_dir = Path(sys.argv[1])
    conv = uuid.uuid4().hex
    try:
        # quick doctor check
        try:
            doctor = subprocess.run(['opencli', 'doctor'], text=True, capture_output=True, timeout=30)
            print('doctor:', (doctor.stdout or doctor.stderr)[:400])
        except Exception as exc:
            print('doctor failed:', exc)
        # try bind
        try:
            print('binding to existing tab...')
            run_opencli(['bind'], timeout=10, allow_failure=True)
        except Exception as exc:
            print('bind failed', exc)
        # capture before_send snapshot
        out = run_opencli(['eval', submission_snapshot_script], timeout=10, allow_failure=True)
        write_debug(record_dir, conv, 'before_send_snapshot', out)
        # fill prompt
        prompt = '调试测试消息：此消息用于检查发送与快照。请勿真实回复。'
        print('filling prompt')
        run_opencli(['fill', '--role', 'textbox', prompt], timeout=15, allow_failure=True)
        # click send (try selectors)
        clicked = False
        last_err = None
        for sel in send_selectors:
            try:
                print('trying click', sel)
                out = run_opencli(['click', sel], timeout=5, allow_failure=True)
                write_debug(record_dir, conv, f'click_attempt_{sel.replace(chr(34),"_")}', out)
                # parse output
                try:
                    parsed = json.loads(out)
                    if parsed.get('clicked'):
                        clicked = True
                        break
                except Exception:
                    # fallback: if output includes 'clicked' string
                    if 'clicked' in out:
                        clicked = True
                        break
            except Exception as exc:
                last_err = exc
        if not clicked:
            print('no selector clicked, last_err:', last_err)
        else:
            print('clicked send')
        # poll for submission confirmation
        prev_user = ''
        deadline = time.time() + 60
        while time.time() < deadline:
            out = run_opencli(['eval', submission_snapshot_script], timeout=10, allow_failure=True)
            write_debug(record_dir, conv, 'during_submission_poll', out)
            try:
                parsed = json.loads(out)
            except Exception:
                parsed = {}
            latest_user = parsed.get('latest_user', '') if isinstance(parsed, dict) else ''
            composer_found = parsed.get('composer_found', False) if isinstance(parsed, dict) else False
            composer_text = parsed.get('composer_text', '') if isinstance(parsed, dict) else ''
            if latest_user and latest_user != prev_user:
                print('latest_user appeared')
                break
            if composer_found and not composer_text.strip():
                print('composer cleared; assuming submitted')
                break
            time.sleep(1.0)
        # wait for assistant reply
        deadline = time.time() + 180
        prev_assistant = ''
        assistant_text = ''
        while time.time() < deadline:
            out = run_opencli(['eval', latest_assistant_script], timeout=10, allow_failure=True)
            write_debug(record_dir, conv, 'assistant_poll', out)
            try:
                assistant_text = json.loads(out)
            except Exception:
                assistant_text = out.strip()
            if assistant_text and assistant_text != prev_assistant:
                print('assistant began reply')
                # wait a bit to stabilize
                time.sleep(2.0)
                break
            time.sleep(1.0)
        # final snapshot
        out = run_opencli(['eval', submission_snapshot_script], timeout=10, allow_failure=True)
        write_debug(record_dir, conv, 'after_response_snapshot', out)
        print('done')
    except Exception as exc:
        print('error during debug run:', exc)
        sys.exit(1)
