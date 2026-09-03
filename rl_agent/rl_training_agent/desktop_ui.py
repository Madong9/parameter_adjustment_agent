from __future__ import annotations

import os
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from tkinter import messagebox
from typing import Any, Dict, List, Optional

from .web_ui import JobManager, JobValidationError, ROBOT_CATALOG


COLORS = {
    "bg": "#090c10",
    "surface": "#10151b",
    "surface_alt": "#0c1116",
    "line": "#283029",
    "line_soft": "#1e2527",
    "text": "#edf0e8",
    "muted": "#818a85",
    "faint": "#535c57",
    "accent": "#c9f27b",
    "accent_dark": "#1d2919",
    "amber": "#e7ad68",
    "danger": "#ff806f",
}

STATUS_LABELS = {
    "queued": "正在启动",
    "running": "训练运行中",
    "stopping": "正在停止",
    "completed": "训练完成",
    "review": "等待人工复核",
    "failed": "训练失败",
    "stopped": "已安全停止",
    "interrupted": "服务曾中断",
}

STAGES = [
    (10, "环境检查"),
    (25, "奖励设计"),
    (48, "冒烟训练"),
    (74, "完整训练"),
    (89, "视觉评估"),
    (100, "训练完成"),
]

CHINESE_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "Droid Sans Fallback",
    "song ti",
    "fangsong ti",
    "SimSun",
]

CHINESE_MONO_FONT_CANDIDATES = [
    "Noto Sans Mono CJK SC",
    "Sarasa Mono SC",
    "Source Han Mono SC",
]


def _shorten(text: str, limit: int) -> str:
    """把过长文本截断为适合桌面控件展示的单行内容。"""
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[:limit - 1] + "…"


def _display_time(value: Optional[str]) -> str:
    """将 ISO 时间转换为简短的本地桌面显示格式。"""
    if not value:
        return "--"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%m-%d  %H:%M")
    except (TypeError, ValueError):
        return "--"


def choose_font_family(available: List[str], candidates: List[str], fallback: str) -> str:
    """按不区分大小写的字体族名称选择当前 Tk 实际可用的字体。"""
    normalized = {name.casefold(): name for name in available}
    for candidate in candidates:
        if candidate.casefold() in normalized:
            return normalized[candidate.casefold()]
    return fallback


def detect_ui_font_families(root: tk.Misc) -> tuple:
    """检测 Tk 可用字体，优先返回能够显示简体中文的正文与日志字体。"""
    available = list(tkfont.families(root))
    default_family = str(tkfont.nametofont("TkDefaultFont").actual("family"))
    body_family = choose_font_family(available, CHINESE_FONT_CANDIDATES, default_family)
    mono_family = choose_font_family(available, CHINESE_MONO_FONT_CANDIDATES, body_family)
    return body_family, mono_family


def normalized_ui_dpi(value: Optional[str]) -> float:
    """将可选 DPI 配置规范到适合桌面界面的安全范围。"""
    try:
        dpi = float(value) if value is not None else 96.0
    except (TypeError, ValueError):
        return 96.0
    return dpi if 72.0 <= dpi <= 240.0 else 96.0


def configure_ui_scaling(root: tk.Misc) -> float:
    """修正多显示器下 Tk 错误的物理尺寸推算并返回生效 DPI。"""
    dpi = normalized_ui_dpi(os.getenv("RL_AGENT_UI_DPI"))
    root.tk.call("tk", "scaling", dpi / 72.0)
    return dpi


class ProgressDial(tk.Canvas):
    """绘制训练百分比与阶段名称的圆形进度仪表。"""

    def __init__(self, master: tk.Misc, body_family: str, mono_family: str, size: int = 168):
        """创建具有深色底环的无边框进度画布。"""
        super().__init__(master, width=size, height=size, bg=COLORS["surface"], highlightthickness=0)
        self.size = size
        self.body_family = body_family
        self.mono_family = mono_family
        self.progress = 0
        self.stage = "准备训练"
        self.bind("<Configure>", self._on_resize)
        self._draw()

    def _on_resize(self, event: tk.Event) -> None:
        """在控件尺寸变化后重新绘制清晰的进度环。"""
        self.size = min(event.width, event.height)
        self._draw()

    def set_progress(self, progress: int, stage: str) -> None:
        """更新百分比和阶段文本并刷新仪表。"""
        self.progress = max(0, min(100, int(progress)))
        self.stage = stage
        self._draw()

    def _draw(self) -> None:
        """绘制底环、进度弧、百分比和状态提示。"""
        self.delete("all")
        size = self.size
        inset = 15
        bounds = (inset, inset, size - inset, size - inset)
        self.create_oval(*bounds, outline=COLORS["line_soft"], width=10)
        extent = -3.6 * self.progress
        if self.progress:
            self.create_arc(*bounds, start=90, extent=extent, style="arc", outline=COLORS["accent"], width=10)
        center = size / 2
        self.create_text(center, center - 9, text=str(self.progress), fill=COLORS["text"],
                         font=(self.mono_family, 31, "normal"))
        self.create_text(center + 34, center - 3, text="%", fill=COLORS["muted"],
                         font=(self.body_family, 11))
        self.create_text(center, center + 26, text=_shorten(self.stage, 10), fill=COLORS["accent"],
                         font=(self.body_family, 10))


class RLDesktopApp:
    """提供原生桌面窗口的强化学习训练上位机应用。"""

    def __init__(self, root: tk.Tk, manager: Optional[JobManager] = None):
        """初始化作业管理器、窗口状态、字体和全部桌面控件。"""
        self.root = root
        self.manager = manager or JobManager()
        self.current_job_id: Optional[str] = None
        self.current_job: Optional[Dict[str, Any]] = None
        self.log_offset = 0
        self.auto_scroll = True
        self.robot = tk.StringVar(value=self.manager.settings.default_robot)
        self.mode = tk.StringVar(value="dry-run")
        self.status_text = tk.StringVar(value="等待任务")
        self.stage_text = tk.StringVar(value="训练通道待命")
        self.task_id_text = tk.StringVar(value="尚未创建实验")
        self.robot_text = tk.StringVar(value="--")
        self.mode_text = tk.StringVar(value="--")
        self.clock_text = tk.StringVar(value="--:--:--")
        self.form_message = tk.StringVar(value="")
        self.char_count = tk.StringVar(value="0 / 2000")
        self.robot_buttons: Dict[str, tk.Button] = {}
        self.mode_buttons: Dict[str, tk.Button] = {}
        self.stage_widgets: List[Dict[str, tk.Widget]] = []
        self._configure_window()
        self._build_styles()
        self._build_layout()
        self._bind_events()
        self._restore_latest_job()
        self._refresh_history()
        self._tick()

    def _configure_window(self) -> None:
        """设置桌面窗口标题、尺寸、颜色和关闭协议。"""
        self.ui_dpi = configure_ui_scaling(self.root)
        self.root.title("动境 · 强化学习训练上位机")
        self.root.geometry("1380x860")
        self.root.minsize(1120, 720)
        self.root.configure(bg=COLORS["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_styles(self) -> None:
        """检测可显示中文的 Tk 字体并定义应用通用字体配置。"""
        self.font_family, self.mono_family = detect_ui_font_families(self.root)
        self.font_title = (self.font_family, 28, "normal")
        self.font_heading = (self.font_family, 13, "bold")
        self.font_body = (self.font_family, 11)
        self.font_small = (self.font_family, 9)
        self.font_mono = (self.mono_family, 10)
        for name, size in (("TkDefaultFont", 10), ("TkTextFont", 10), ("TkMenuFont", 10),
                           ("TkHeadingFont", 11), ("TkFixedFont", 10)):
            named_font = tkfont.nametofont(name)
            named_font.configure(family=self.font_family if name != "TkFixedFont" else self.mono_family, size=size)

    def _frame(self, master: tk.Misc, bg: str = "surface", **kwargs: Any) -> tk.Frame:
        """创建使用统一主题背景的 Frame。"""
        return tk.Frame(master, bg=COLORS[bg], **kwargs)

    def _label(self, master: tk.Misc, text: str = "", fg: str = "text", bg: str = "surface",
               font: Optional[tuple] = None, **kwargs: Any) -> tk.Label:
        """创建使用统一颜色和字体的 Label。"""
        return tk.Label(master, text=text, fg=COLORS[fg], bg=COLORS[bg],
                        font=font or self.font_body, **kwargs)

    def _section_heading(self, master: tk.Misc, number: str, title: str) -> tk.Frame:
        """创建带荧光编号的控制区标题。"""
        frame = self._frame(master)
        self._label(frame, number, fg="accent", font=(self.mono_family, 9)).pack(side="left")
        self._label(frame, title, font=self.font_heading).pack(side="left", padx=(10, 0))
        return frame

    def _build_layout(self) -> None:
        """构建顶部状态栏、控制台、遥测、日志和历史记录布局。"""
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self._build_header()
        content = self._frame(self.root, bg="bg")
        content.grid(row=1, column=0, sticky="nsew", padx=22, pady=(0, 20))
        content.grid_columnconfigure(0, weight=11, uniform="main")
        content.grid_columnconfigure(1, weight=9, uniform="main")
        content.grid_rowconfigure(0, weight=13)
        content.grid_rowconfigure(1, weight=7)

        command = self._panel(content)
        command.grid(row=0, column=0, sticky="nsew", padx=(0, 7), pady=(0, 7))
        telemetry = self._panel(content)
        telemetry.grid(row=0, column=1, sticky="nsew", padx=(7, 0), pady=(0, 7))
        log_panel = self._panel(content)
        log_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 7), pady=(7, 0))
        history = self._panel(content)
        history.grid(row=1, column=1, sticky="nsew", padx=(7, 0), pady=(7, 0))
        self._build_command_panel(command)
        self._build_telemetry_panel(telemetry)
        self._build_log_panel(log_panel)
        self._build_history_panel(history)

    def _panel(self, master: tk.Misc) -> tk.Frame:
        """创建具有主题边缘留白的桌面面板。"""
        outer = tk.Frame(master, bg=COLORS["line_soft"], padx=1, pady=1)
        inner = self._frame(outer)
        inner.pack(fill="both", expand=True)
        inner._outer_panel = outer
        return outer

    def _panel_body(self, panel: tk.Frame) -> tk.Frame:
        """返回面板边框内部的实际内容容器。"""
        return panel.winfo_children()[0]

    def _build_header(self) -> None:
        """构建品牌、训练工程状态和本地时钟顶栏。"""
        header = self._frame(self.root, bg="bg", height=92)
        header.grid(row=0, column=0, sticky="ew", padx=24)
        header.grid_propagate(False)
        brand = self._frame(header, bg="bg")
        brand.pack(side="left", fill="y")
        logo = tk.Canvas(brand, width=48, height=48, bg=COLORS["bg"], highlightthickness=0)
        logo.pack(side="left", pady=21)
        logo.create_oval(4, 4, 44, 44, outline=COLORS["line"], width=1)
        logo.create_oval(11, 11, 37, 37, outline=COLORS["accent"], width=1)
        for x, y in ((19, 20), (29, 19), (25, 30)):
            logo.create_oval(x - 2, y - 2, x + 2, y + 2, fill=COLORS["accent"], outline="")
        name_box = self._frame(brand, bg="bg")
        name_box.pack(side="left", pady=23, padx=(8, 0))
        self._label(name_box, "动境", bg="bg", font=(self.font_family, 17, "bold")).pack(anchor="w")
        self._label(name_box, "KINETIC LAB · RL DESKTOP", fg="muted", bg="bg",
                    font=(self.mono_family, 8)).pack(anchor="w")

        status = self._frame(header, bg="bg")
        status.pack(side="right", fill="y")
        project_ready = self.manager.settings.training_root.is_dir()
        self._label(status, "●", fg="accent" if project_ready else "danger", bg="bg",
                    font=("Arial", 10)).pack(side="left", pady=34)
        self._label(status, "训练工程就绪" if project_ready else "训练工程未就绪", fg="muted", bg="bg",
                    font=self.font_small).pack(side="left", padx=(5, 24), pady=34)
        tk.Frame(status, bg=COLORS["line_soft"], width=1).pack(side="left", fill="y", pady=27)
        self._label(status, textvariable=self.clock_text, bg="bg", font=self.font_mono).pack(
            side="left", padx=(24, 0), pady=34)

    def _build_command_panel(self, panel: tk.Frame) -> None:
        """构建动作描述、机型选择、运行模式和任务下发区。"""
        body = self._panel_body(panel)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)
        heading = self._section_heading(body, "01", "动作任务下发")
        heading.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 10))
        self._label(heading, "自然语言训练目标", fg="muted", font=self.font_small).pack(side="right")

        middle = self._frame(body)
        middle.grid(row=1, column=0, sticky="nsew", padx=22)
        middle.grid_columnconfigure(0, weight=1)
        middle.grid_rowconfigure(0, weight=1)
        text_border = tk.Frame(middle, bg=COLORS["line_soft"], padx=1, pady=1)
        text_border.grid(row=0, column=0, sticky="nsew")
        self.task_input = tk.Text(text_border, height=5, wrap="word", undo=True, bg=COLORS["surface_alt"],
                                  fg=COLORS["text"], insertbackground=COLORS["accent"], relief="flat",
                                  padx=14, pady=12, font=self.font_body, selectbackground=COLORS["accent_dark"])
        self.task_input.pack(fill="both", expand=True)
        self.task_input.insert("1.0", "训练机器狗稳定向前小跑，并在受到侧向推力后快速恢复平衡")
        self.task_input.bind("<KeyRelease>", self._update_char_count)

        meta = self._frame(middle)
        meta.grid(row=1, column=0, sticky="ew", pady=(5, 12))
        self._label(meta, "Ctrl + Enter  下发", fg="faint", font=self.font_small).pack(side="left")
        self._label(meta, textvariable=self.char_count, fg="faint", font=self.font_small).pack(side="right")
        self._update_char_count()

        robot_heading = self._section_heading(middle, "02", "选择训练机型")
        robot_heading.grid(row=2, column=0, sticky="ew", pady=(0, 9))
        robot_grid = self._frame(middle)
        robot_grid.grid(row=3, column=0, sticky="ew")
        for column in range(4):
            robot_grid.grid_columnconfigure(column, weight=1)
        allowed = [item for item in ROBOT_CATALOG if item["id"] in self.manager.settings.allowed_robots]
        for index, robot in enumerate(allowed):
            button = tk.Button(robot_grid, text="{}\n{}\n{}".format(robot["mark"], robot["name"], robot["kind"]),
                               command=lambda value=robot["id"]: self._select_robot(value), justify="left",
                               anchor="w", padx=11, pady=9, relief="flat", bd=0, cursor="hand2",
                               font=self.font_small, highlightthickness=1)
            button.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 4, 0))
            self.robot_buttons[robot["id"]] = button
        self._paint_robot_buttons()

        mode_heading = self._section_heading(middle, "03", "运行方式")
        mode_heading.grid(row=4, column=0, sticky="ew", pady=(14, 8))
        mode_grid = self._frame(middle)
        mode_grid.grid(row=5, column=0, sticky="ew")
        mode_grid.grid_columnconfigure((0, 1), weight=1)
        modes = (("dry-run", "离线演练\n模拟全流程 · 无需 GPU"),
                 ("real", "真实训练\nOpenCLI 推理 · GPU 仿真"))
        for index, (mode_id, label) in enumerate(modes):
            button = tk.Button(mode_grid, text=label, command=lambda value=mode_id: self._select_mode(value),
                               justify="left", anchor="w", padx=12, pady=8, relief="flat", bd=0,
                               cursor="hand2", font=self.font_small, highlightthickness=1)
            button.grid(row=0, column=index, sticky="ew", padx=(0, 4) if index == 0 else (4, 0))
            self.mode_buttons[mode_id] = button
        self._paint_mode_buttons()

        self.form_label = self._label(middle, textvariable=self.form_message, fg="danger", font=self.font_small,
                                      anchor="w")
        self.form_label.grid(row=6, column=0, sticky="ew", pady=(7, 2))

        self.launch_button = tk.Button(body, text="下发训练任务   ↗", command=self._launch_job,
                                       bg=COLORS["accent"], fg="#11150d", activebackground="#d7ff8a",
                                       activeforeground="#11150d", relief="flat", bd=0, cursor="hand2",
                                       font=(self.font_family, 12, "bold"), pady=11)
        self.launch_button.grid(row=2, column=0, sticky="ew", padx=22, pady=(7, 20))

    def _build_telemetry_panel(self, panel: tk.Frame) -> None:
        """构建训练状态灯、圆形进度、实验信息、阶段轨迹和停止按钮。"""
        body = self._panel_body(panel)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(2, weight=1)
        heading = self._section_heading(body, "LIVE", "学习遥测")
        heading.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 6))
        self.status_badge = self._label(heading, textvariable=self.status_text, fg="muted", font=self.font_small)
        self.status_badge.pack(side="right")

        overview = self._frame(body)
        overview.grid(row=1, column=0, sticky="ew", padx=22, pady=(12, 8))
        self.progress_dial = ProgressDial(overview, self.font_family, self.mono_family, 168)
        self.progress_dial.pack(side="left")
        copy = self._frame(overview)
        copy.pack(side="left", fill="both", expand=True, padx=(22, 0), pady=20)
        self._label(copy, "当前阶段", fg="accent", font=self.font_small).pack(anchor="w")
        self._label(copy, textvariable=self.stage_text, font=(self.font_family, 18)).pack(anchor="w", pady=(5, 7))
        self.telemetry_task = self._label(copy, "在左侧下发动作目标后，此处将呈现策略生长过程。",
                                          fg="muted", font=self.font_small, wraplength=320, justify="left")
        self.telemetry_task.pack(anchor="w")

        lower = self._frame(body)
        lower.grid(row=2, column=0, sticky="nsew", padx=22)
        lower.grid_columnconfigure(0, weight=1)
        facts = self._frame(lower, bg="surface_alt")
        facts.grid(row=0, column=0, sticky="ew", pady=(4, 14))
        facts.grid_columnconfigure((0, 1, 2), weight=1)
        fact_values = (("实验编号", self.task_id_text), ("训练机型", self.robot_text), ("运行模式", self.mode_text))
        for index, (title, variable) in enumerate(fact_values):
            cell = self._frame(facts, bg="surface_alt")
            cell.grid(row=0, column=index, sticky="ew", padx=12, pady=10)
            self._label(cell, title, fg="muted", bg="surface_alt", font=self.font_small).pack(anchor="w")
            self._label(cell, textvariable=variable, bg="surface_alt", font=self.font_mono).pack(anchor="w", pady=(4, 0))

        stages = self._frame(lower)
        stages.grid(row=1, column=0, sticky="nsew")
        for index, (threshold, label) in enumerate(STAGES):
            dot = self._label(stages, "○", fg="faint", font=(self.mono_family, 11))
            dot.grid(row=index, column=0, sticky="w", pady=3)
            text = self._label(stages, label, fg="faint", font=self.font_small)
            text.grid(row=index, column=1, sticky="w", padx=(8, 0), pady=3)
            self.stage_widgets.append({"threshold": threshold, "dot": dot, "text": text})

        self.stop_button = tk.Button(body, text="■   安全停止训练", command=self._stop_job,
                                     bg=COLORS["surface"], fg=COLORS["danger"], disabledforeground=COLORS["faint"],
                                     activebackground="#251817", activeforeground=COLORS["danger"],
                                     relief="solid", bd=1, cursor="hand2", font=self.font_small, pady=9,
                                     state="disabled")
        self.stop_button.grid(row=3, column=0, sticky="ew", padx=22, pady=(8, 20))

    def _build_log_panel(self, panel: tk.Frame) -> None:
        """构建支持增量显示、复制、清屏和自动跟随的日志终端。"""
        body = self._panel_body(panel)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)
        heading = self._section_heading(body, "LOG", "运行日志")
        heading.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 8))
        actions = self._frame(heading)
        actions.pack(side="right")
        self.follow_button = tk.Button(actions, text="自动跟随", command=self._toggle_follow, relief="flat", bd=0,
                                       bg=COLORS["accent_dark"], fg=COLORS["accent"], cursor="hand2",
                                       font=self.font_small, padx=8, pady=4)
        self.follow_button.pack(side="left", padx=3)
        for text, command in (("复制", self._copy_log), ("清屏", self._clear_log)):
            tk.Button(actions, text=text, command=command, relief="flat", bd=0, bg=COLORS["surface_alt"],
                      fg=COLORS["muted"], cursor="hand2", font=self.font_small, padx=8, pady=4).pack(side="left", padx=3)
        log_frame = self._frame(body, bg="surface_alt")
        log_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 16))
        scrollbar = tk.Scrollbar(log_frame, orient="vertical", bg=COLORS["surface"], troughcolor=COLORS["surface_alt"])
        scrollbar.pack(side="right", fill="y")
        self.log_output = tk.Text(log_frame, wrap="word", state="disabled", bg=COLORS["surface_alt"],
                                  fg="#a9b2ac", insertbackground=COLORS["accent"], relief="flat", bd=0,
                                  padx=12, pady=10, font=self.font_mono, yscrollcommand=scrollbar.set)
        self.log_output.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self.log_output.yview)
        self._append_log("RL TRAINING CONTROL / READY\n等待训练任务。日志将保存在 artifacts/ui_jobs/。\n")

    def _build_history_panel(self, panel: tk.Frame) -> None:
        """构建最近实验列表和手动刷新入口。"""
        body = self._panel_body(panel)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)
        heading = self._section_heading(body, "REC", "最近实验")
        heading.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 8))
        tk.Button(heading, text="↻ 刷新", command=self._refresh_history, relief="flat", bd=0,
                  bg=COLORS["surface_alt"], fg=COLORS["muted"], cursor="hand2",
                  font=self.font_small, padx=8, pady=4).pack(side="right")
        frame = self._frame(body, bg="surface_alt")
        frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 16))
        scrollbar = tk.Scrollbar(frame, orient="vertical", bg=COLORS["surface"], troughcolor=COLORS["surface_alt"])
        scrollbar.pack(side="right", fill="y")
        self.history_list = tk.Listbox(frame, bg=COLORS["surface_alt"], fg=COLORS["text"],
                                       selectbackground=COLORS["accent_dark"], selectforeground=COLORS["accent"],
                                       relief="flat", bd=0, highlightthickness=0, font=self.font_small,
                                       activestyle="none", yscrollcommand=scrollbar.set)
        self.history_list.pack(side="left", fill="both", expand=True, padx=9, pady=7)
        scrollbar.configure(command=self.history_list.yview)

    def _bind_events(self) -> None:
        """绑定任务快捷键和窗口级键盘操作。"""
        self.root.bind("<Control-Return>", lambda _event: self._launch_job())

    def _paint_robot_buttons(self) -> None:
        """根据当前机型更新所有机器人按钮的颜色。"""
        for robot_id, button in self.robot_buttons.items():
            selected = robot_id == self.robot.get()
            button.configure(bg=COLORS["accent_dark"] if selected else COLORS["surface_alt"],
                             fg=COLORS["accent"] if selected else COLORS["muted"],
                             activebackground=COLORS["accent_dark"],
                             activeforeground=COLORS["accent"],
                             highlightbackground=COLORS["accent"] if selected else COLORS["line_soft"])

    def _select_robot(self, robot_id: str) -> None:
        """选择机器人训练配置并刷新按钮样式。"""
        self.robot.set(robot_id)
        self._paint_robot_buttons()

    def _paint_mode_buttons(self) -> None:
        """根据当前模式更新离线与真实训练按钮颜色。"""
        for mode_id, button in self.mode_buttons.items():
            selected = mode_id == self.mode.get()
            button.configure(bg=COLORS["accent_dark"] if selected else COLORS["surface_alt"],
                             fg=COLORS["accent"] if selected else COLORS["muted"],
                             activebackground=COLORS["accent_dark"],
                             activeforeground=COLORS["accent"],
                             highlightbackground=COLORS["accent"] if selected else COLORS["line_soft"])

    def _select_mode(self, mode_id: str) -> None:
        """选择训练运行方式并刷新按钮样式和安全提示。"""
        self.mode.set(mode_id)
        self._paint_mode_buttons()
        self.form_message.set("真实训练会调用 OpenCLI 和 GPU 仿真，不会连接实体机器人" if mode_id == "real" else "")

    def _update_char_count(self, _event: Optional[tk.Event] = None) -> None:
        """统计动作描述字符数并限制输入最大长度。"""
        text = self.task_input.get("1.0", "end-1c")
        if len(text) > 2000:
            self.task_input.delete("1.0+2000c", "end")
            text = text[:2000]
        self.char_count.set("{} / 2000".format(len(text)))

    def _set_launch_available(self, available: bool) -> None:
        """切换任务下发按钮的可用状态和文字。"""
        self.launch_button.configure(state="normal" if available else "disabled",
                                     text="下发训练任务   ↗" if available else "训练任务运行中")

    def _launch_job(self) -> None:
        """校验桌面输入并创建受限训练作业。"""
        task = self.task_input.get("1.0", "end-1c").strip()
        self.form_message.set("")
        self._set_launch_available(False)
        try:
            job = self.manager.start_job({"task": task, "robot": self.robot.get(), "mode": self.mode.get()})
        except JobValidationError as exc:
            self.form_message.set(str(exc))
            self._set_launch_available(True)
            return
        except Exception as exc:
            self.form_message.set("训练启动失败：{}".format(exc))
            self._set_launch_available(True)
            return
        self.current_job_id = job["job_id"]
        self.current_job = job
        self.log_offset = 0
        self._clear_log()
        self._render_job(job)
        self._poll_log()
        self._refresh_history()

    def _stop_job(self) -> None:
        """经用户确认后安全停止当前训练进程组。"""
        if not self.current_job_id or not self.current_job or not self.current_job.get("can_stop"):
            return
        if not messagebox.askyesno("安全停止训练", "确定停止当前训练任务吗？\n已生成的实验文件会保留。", parent=self.root):
            return
        try:
            job = self.manager.stop_job(self.current_job_id)
            self._render_job(job)
        except (JobValidationError, KeyError) as exc:
            messagebox.showerror("停止失败", str(exc), parent=self.root)

    def _render_job(self, job: Dict[str, Any]) -> None:
        """把最新作业信息渲染到状态、进度、事实和阶段控件。"""
        self.current_job = job
        status = str(job.get("status", ""))
        progress = int(job.get("progress", 0))
        stage = str(job.get("stage_label", "准备训练"))
        self.status_text.set("●  " + STATUS_LABELS.get(status, "等待任务"))
        self.status_badge.configure(fg=COLORS["danger"] if status in ("failed", "stopped") else
                                    COLORS["amber"] if status == "review" else
                                    COLORS["accent"] if status in ("running", "completed") else COLORS["muted"])
        self.stage_text.set(stage)
        self.task_id_text.set(str(job.get("task_id", "--")))
        self.robot_text.set(str(job.get("robot", "--")).upper())
        self.mode_text.set("真实训练" if job.get("mode") == "real" else "离线演练")
        self.telemetry_task.configure(text=_shorten(str(job.get("task", "")), 80))
        self.progress_dial.set_progress(progress, stage)
        self.stop_button.configure(state="normal" if job.get("can_stop") else "disabled")
        active = status in ("queued", "running", "stopping")
        self._set_launch_available(not active)
        for item in self.stage_widgets:
            done = progress >= int(item["threshold"])
            item["dot"].configure(text="●" if done else "○", fg=COLORS["accent"] if done else COLORS["faint"])
            item["text"].configure(fg=COLORS["text"] if done else COLORS["faint"])

    def _append_log(self, text: str) -> None:
        """向只读日志控件追加内容并按需滚动到底部。"""
        self.log_output.configure(state="normal")
        self.log_output.insert("end", text)
        self.log_output.configure(state="disabled")
        if self.auto_scroll:
            self.log_output.see("end")

    def _poll_log(self) -> None:
        """从当前作业的字节偏移处读取新增日志。"""
        if not self.current_job_id:
            return
        try:
            result = self.manager.read_log(self.current_job_id, self.log_offset)
        except (KeyError, JobValidationError):
            return
        if result["text"]:
            self._append_log(result["text"])
        self.log_offset = int(result["next_offset"])

    def _toggle_follow(self) -> None:
        """切换日志自动滚动状态。"""
        self.auto_scroll = not self.auto_scroll
        self.follow_button.configure(text="自动跟随" if self.auto_scroll else "暂停跟随",
                                     bg=COLORS["accent_dark"] if self.auto_scroll else COLORS["surface_alt"],
                                     fg=COLORS["accent"] if self.auto_scroll else COLORS["muted"])

    def _copy_log(self) -> None:
        """把当前可见日志复制到系统剪贴板。"""
        text = self.log_output.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_text.set("日志已复制")

    def _clear_log(self) -> None:
        """清除桌面日志控件内容但不删除磁盘日志。"""
        self.log_output.configure(state="normal")
        self.log_output.delete("1.0", "end")
        self.log_output.configure(state="disabled")

    def _refresh_history(self) -> None:
        """从实验目录读取最近记录并刷新列表。"""
        experiments = self.manager.list_experiments(20)
        self.history_list.delete(0, "end")
        if not experiments:
            self.history_list.insert("end", "  暂无实验记录")
            return
        for item in experiments:
            line = "  {}  ·  {}  ·  {}  ·  {}".format(
                _shorten(str(item.get("task", "")), 38),
                str(item.get("robot", "--")).upper(),
                str(item.get("stage_label", "")),
                _display_time(item.get("updated_at")),
            )
            self.history_list.insert("end", line)

    def _restore_latest_job(self) -> None:
        """在桌面应用启动时恢复最近一次界面作业的展示状态。"""
        jobs = self.manager.list_jobs(1)
        if not jobs:
            return
        job = jobs[0]
        self.current_job_id = job["job_id"]
        self.log_offset = 0
        self._clear_log()
        self._render_job(job)
        self._poll_log()

    def _tick(self) -> None:
        """周期更新时钟、作业状态、日志和低频实验历史。"""
        self.clock_text.set(datetime.now().strftime("%H:%M:%S"))
        if self.current_job_id:
            try:
                job = self.manager.get_job(self.current_job_id)
                self._render_job(job)
                self._poll_log()
            except KeyError:
                self.current_job_id = None
        if datetime.now().second % 10 == 0:
            self._refresh_history()
        self.root.after(1000, self._tick)

    def _on_close(self) -> None:
        """在关闭窗口前确认并停止仍由桌面端跟踪的训练。"""
        if self.current_job and self.current_job.get("can_stop"):
            confirmed = messagebox.askyesno(
                "训练仍在运行",
                "关闭上位机前需要安全停止当前训练。\n确定停止训练并退出吗？",
                parent=self.root,
            )
            if not confirmed:
                return
            try:
                self.manager.stop_job(self.current_job["job_id"])
            except (JobValidationError, KeyError):
                pass
        self.root.destroy()


def create_app(manager: Optional[JobManager] = None) -> tuple:
    """创建根窗口与桌面应用实例，便于启动和界面测试复用。"""
    root = tk.Tk()
    app = RLDesktopApp(root, manager)
    return root, app


def serve_tk_desktop() -> int:
    """启动兼容版 Tk 桌面上位机并进入事件循环。"""
    try:
        root, _app = create_app()
    except tk.TclError as exc:
        raise RuntimeError("无法连接图形桌面，请在带显示器的桌面会话中启动上位机") from exc
    root.mainloop()
    return 0
