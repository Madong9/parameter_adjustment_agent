from pathlib import Path

from rl_training_agent.cli import build_parser
from rl_training_agent.desktop_ui import (
    STATUS_LABELS,
    _display_time,
    _shorten,
    choose_font_family,
    normalized_ui_dpi,
)
from rl_training_agent.desktop_launcher import build_browser_command


def test_desktop_command_is_available():
    """验证命令行能够解析桌面软件和 Tk 兼容版启动命令。"""
    args = build_parser().parse_args(["desktop"])
    assert args.command == "desktop"
    assert build_parser().parse_args(["desktop-tk"]).command == "desktop-tk"
    audit = build_parser().parse_args(["visual-audit", "--task-id", "task-demo"])
    assert audit.command == "visual-audit" and audit.rollout == "final/evaluation_rollout"
    play = build_parser().parse_args(["play", "--task-id", "task-demo",
                                      "--checkpoint", "experiments/demo/model_1.pt"])
    assert play.command == "play" and play.num_envs == 1


def test_desktop_text_helpers_are_stable():
    """验证桌面端的长文本和实验时间能够稳定转换为紧凑显示。"""
    assert _shorten("训练机器狗稳定向前行走", 8) == "训练机器狗稳定…"
    assert _shorten("短动作", 8) == "短动作"
    assert _display_time("2026-07-16T10:00:00+08:00") == "07-16  10:00"
    assert _display_time(None) == "--"
    assert STATUS_LABELS["running"] == "训练运行中"
    assert STATUS_LABELS["review"] == "等待人工复核"


def test_desktop_font_selection_uses_tk_reported_family():
    """验证桌面端只选择 Tk 实际报告为可用的中文字体族。"""
    available = ["DejaVu Sans", "song ti", "fangsong ti"]
    assert choose_font_family(available, ["Noto Sans CJK SC", "song ti"], "DejaVu Sans") == "song ti"
    assert choose_font_family(available, ["Missing Font"], "DejaVu Sans") == "DejaVu Sans"


def test_desktop_dpi_rejects_broken_screen_measurement():
    """验证桌面端对异常或过小 DPI 使用可读的 96 DPI 基准。"""
    assert normalized_ui_dpi(None) == 96.0
    assert normalized_ui_dpi("27") == 96.0
    assert normalized_ui_dpi("144") == 144.0
    assert normalized_ui_dpi("not-a-number") == 96.0


def test_desktop_browser_uses_isolated_app_window():
    """验证主桌面入口使用隔离配置和无地址栏的应用窗口参数。"""
    command = build_browser_command("/usr/bin/google-chrome", "http://127.0.0.1:8765/", Path("artifacts/profile"))
    assert command[0] == "/usr/bin/google-chrome"
    assert "--app=http://127.0.0.1:8765/" in command
    assert "--user-data-dir=artifacts/profile" in command
    assert "--disable-extensions" in command
