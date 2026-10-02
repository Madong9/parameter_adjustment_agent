#!/usr/bin/env python3
"""复用生产 Provider 排查 OpenCLI；默认只读，显式 --send 才发自检消息。"""

import argparse
import json
import sys
from pathlib import Path

# 支持从仓库任意目录执行，不依赖安装包或硬编码机器路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_training_agent.providers.errors import ProviderError
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider
from rl_training_agent.settings import OpenCLISettings


def main(argv=None):
    """检查指定会话，或在独立的新普通聊天会话执行端到端 JSON 自检。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_dir", type=Path, help="本地诊断记录目录，请勿提交其中的对话内容")
    parser.add_argument("--session", default="rl-opencli-debug", help="OpenCLI 会话名")
    parser.add_argument("--profile", default="", help="Browser Bridge 浏览器 profile")
    parser.add_argument("--send", action="store_true", help="新建独立会话并发送无敏感信息的自检消息")
    parser.add_argument("--rounds", type=int, default=1, help="自检重复发送相同提示词的次数（默认 1）")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds 必须至少为 1")
    provider = OpenCLIChatGPTWebProvider(
        settings=OpenCLISettings(
            session=args.session, profile=args.profile, bind_existing_tab=False,
            owned_session=True, max_retries=0, response_timeout=90),
        record_dir=args.record_dir)
    conversation = None
    try:
        if args.send:
            conversation = provider.new_conversation("opencli-communication-test")
            for index in range(args.rounds):
                reply = provider.send_text(
                    '这是程序通信自检，不涉及机器人训练。请只返回严格 JSON：{"ok":true}',
                    conversation)
                parsed = provider._parse_candidate(reply)
                if parsed != {"ok": True}:
                    raise ProviderError("自检回复不符合预期，请查看本地响应记录。")
                print("第 %s 轮自检通过：普通聊天、消息提交确认及 JSON 回复读取均成功。" %
                      (index + 1), flush=True)
        else:
            # 不 bind、不跳转、不发送，不影响正在进行的训练会话。
            provider._state()
            snapshot = provider._submission_snapshot()
            print(json.dumps({
                "user_count": snapshot.get("user_count", 0),
                "assistant_count": snapshot.get("assistant_count", 0),
                "composer_found": snapshot.get("composer_found", False),
                "generating": snapshot.get("generating", False),
                "extraction_version": snapshot.get("extraction_version", ""),
                "reply_matches_latest_user": bool(snapshot.get("latest_user_id") and
                    snapshot.get("assistant_user_id") == snapshot.get("latest_user_id")),
                "reply_contains_complete_json": provider._response_contains_complete_json(
                    str(snapshot.get("latest_assistant", ""))),
            }, ensure_ascii=False, indent=2))
        provider._collect_page_debug_artifacts(conversation, "direct_debug")
        print("诊断记录目录：%s" % args.record_dir)
        return 0
    except (ProviderError, ValueError) as exc:
        print("自检失败：%s" % exc, file=sys.stderr)
        return 1
    finally:
        if args.send:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
