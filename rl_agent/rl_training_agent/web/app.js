"use strict";

const state = {
  config: null,
  robot: "go2",
  mode: "dry-run",
  currentJobId: localStorage.getItem("rl-current-job") || null,
  currentJob: null,
  logOffset: 0,
  autoScroll: true,
  pollTimer: null,
};

const elements = {
  taskInput: document.querySelector("#taskInput"),
  charCount: document.querySelector("#charCount"),
  robotGrid: document.querySelector("#robotGrid"),
  modeSwitch: document.querySelector("#modeSwitch"),
  modeNote: document.querySelector("#modeNote"),
  launchButton: document.querySelector("#launchButton"),
  formError: document.querySelector("#formError"),
  telemetryEmpty: document.querySelector("#telemetryEmpty"),
  telemetryContent: document.querySelector("#telemetryContent"),
  runState: document.querySelector("#runState"),
  progressRing: document.querySelector("#progressRing"),
  progressValue: document.querySelector("#progressValue"),
  stageLabel: document.querySelector("#stageLabel"),
  activeTask: document.querySelector("#activeTask"),
  taskId: document.querySelector("#taskId"),
  activeRobot: document.querySelector("#activeRobot"),
  activeMode: document.querySelector("#activeMode"),
  loopRound: document.querySelector("#loopRound"),
  loopBudget: document.querySelector("#loopBudget"),
  stageTrack: document.querySelector("#stageTrack"),
  resumeButton: document.querySelector("#resumeButton"),
  stopButton: document.querySelector("#stopButton"),
  terminal: document.querySelector("#terminal"),
  terminalWelcome: document.querySelector("#terminalWelcome"),
  logOutput: document.querySelector("#logOutput"),
  autoScrollButton: document.querySelector("#autoScrollButton"),
  copyLogButton: document.querySelector("#copyLogButton"),
  clearLogButton: document.querySelector("#clearLogButton"),
  historyList: document.querySelector("#historyList"),
  refreshButton: document.querySelector("#refreshButton"),
  projectSignal: document.querySelector("#projectSignal"),
  projectStatus: document.querySelector("#projectStatus"),
  clock: document.querySelector("#clock"),
  toast: document.querySelector("#toast"),
};

/** 调用本地上位机 API，并把非成功响应转换为可读错误。 */
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `请求失败（${response.status}）`);
  }
  return payload;
}

/** 在页面底部短暂显示成功或错误提示。 */
function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("show"), 2600);
}

/** 使用本地时间格式化实验更新时间。 */
function formatTime(value) {
  if (!value) return "时间未知";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "时间未知";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }).format(parsed);
}

/** 对插入字符串模板的内容进行 HTML 转义。 */
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

/** 更新顶栏的本地上位机时钟。 */
function updateClock() {
  elements.clock.textContent = new Intl.DateTimeFormat("zh-CN", {
    hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit",
  }).format(new Date());
}

/** 根据后端配置生成机器人机型选择卡。 */
function renderRobots(robots) {
  elements.robotGrid.innerHTML = robots.map((robot) => `
    <button type="button" class="robot-card ${robot.id === state.robot ? "active" : ""}"
      data-robot="${escapeHtml(robot.id)}" role="radio" aria-checked="${robot.id === state.robot}">
      <span class="check">●</span>
      <span class="robot-mark">${escapeHtml(robot.mark)}</span>
      <b>${escapeHtml(robot.name)}</b>
      <small>${escapeHtml(robot.kind)}</small>
    </button>
  `).join("");
}

/** 切换当前训练机器人并同步卡片无障碍状态。 */
function selectRobot(robotId) {
  state.robot = robotId;
  document.querySelectorAll(".robot-card").forEach((card) => {
    const active = card.dataset.robot === robotId;
    card.classList.toggle("active", active);
    card.setAttribute("aria-checked", String(active));
  });
}

/** 切换离线演练或真实训练，并显示相应的安全说明。 */
function selectMode(mode) {
  state.mode = mode;
  elements.modeSwitch.querySelectorAll("button").forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-checked", String(active));
  });
  elements.modeNote.innerHTML = mode === "real"
    ? "<span aria-hidden=\"true\">◇</span><p><b>真实训练模式</b>将调用 OpenCLI 网页推理并启动 GPU 仿真。实体机器人部署不在此界面范围内。</p>"
    : "<span aria-hidden=\"true\">◇</span><p><b>离线安全模式</b>将生成模拟训练结果，用于检查 Agent 全链路，不会启动实际 GPU 训练。</p>";
}

/** 将作业状态转换为上位机状态灯使用的中文文本。 */
function jobStatusLabel(status) {
  return {
    queued: "正在启动", running: "训练运行中", stopping: "正在停止",
    completed: "训练完成", review: "等待人工复核", failed: "训练失败",
    stopped: "已安全停止", interrupted: "服务曾中断",
  }[status] || "等待任务";
}

/** 用最新作业数据刷新进度仪表、阶段轨迹和急停能力。 */
function renderTelemetry(job) {
  state.currentJob = job;
  elements.telemetryEmpty.classList.add("hidden");
  elements.telemetryContent.classList.remove("hidden");
  elements.runState.className = `run-state ${job.status}`;
  elements.runState.innerHTML = `<i></i>${escapeHtml(jobStatusLabel(job.status))}`;
  elements.progressRing.style.setProperty("--progress", Number(job.progress || 0));
  elements.progressValue.textContent = String(job.progress || 0);
  elements.stageLabel.textContent = job.stage_label || "准备训练";
  elements.activeTask.textContent = job.task;
  elements.taskId.textContent = job.task_id;
  elements.activeRobot.textContent = String(job.robot).toUpperCase();
  elements.activeMode.textContent = job.mode === "real" ? "真实训练" : "离线演练";
  const loop = job.loop_detail || {};
  elements.loopRound.textContent = loop.round
    ? `第 ${loop.round} 轮 / v${loop.reward_version || 1}` : "尚未评估";
  elements.loopBudget.textContent = Number.isFinite(Number(loop.remaining_iterations))
    ? `${loop.remaining_iterations} iter · ${loop.remaining_revisions} 次修订` : "—";
  elements.stopButton.disabled = !job.can_stop;
  elements.resumeButton.disabled = !job.can_resume;
  elements.resumeButton.classList.toggle("hidden", !job.can_resume);
  const active = ["queued", "running", "stopping"].includes(job.status);
  elements.launchButton.disabled = active;
  elements.launchButton.querySelector("span").textContent = active ? "训练任务运行中" : "下发训练任务";
  elements.stageTrack.querySelectorAll("[data-threshold]").forEach((stage) => {
    stage.classList.toggle("done", Number(job.progress) >= Number(stage.dataset.threshold));
  });
}

/** 增量读取当前作业日志，并按开关决定是否滚动到底部。 */
async function pollLog() {
  if (!state.currentJobId) return;
  const result = await api(`/api/jobs/${state.currentJobId}/logs?offset=${state.logOffset}`);
  if (result.text) {
    elements.terminalWelcome.classList.add("hidden");
    elements.logOutput.textContent += result.text;
    if (state.autoScroll) elements.terminal.scrollTop = elements.terminal.scrollHeight;
  }
  state.logOffset = result.next_offset;
}

/** 拉取当前作业状态和日志，遇到短暂错误时保留现有画面。 */
async function pollCurrentJob() {
  if (!state.currentJobId) return;
  try {
    const job = await api(`/api/jobs/${state.currentJobId}`);
    renderTelemetry(job);
    await pollLog();
  } catch (error) {
    console.warn("轮询训练状态失败", error);
  }
}

/** 创建新的训练作业，并切换仪表与日志到该作业。 */
async function launchJob() {
  const task = elements.taskInput.value.trim();
  elements.formError.textContent = "";
  if (task.length < 4) {
    elements.formError.textContent = "请更具体地描述想训练的动作（至少 4 个字符）";
    elements.taskInput.focus();
    return;
  }
  elements.launchButton.disabled = true;
  elements.launchButton.querySelector("span").textContent = "正在下发…";
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({ task, robot: state.robot, mode: state.mode }),
    });
    state.currentJobId = job.job_id;
    state.logOffset = 0;
    elements.logOutput.textContent = "";
    localStorage.setItem("rl-current-job", job.job_id);
    renderTelemetry(job);
    await pollLog();
    await refreshHistory();
    showToast(`训练任务已下发 · ${job.task_id}`);
  } catch (error) {
    elements.formError.textContent = error.message;
    showToast(error.message, true);
  } finally {
    const active = ["queued", "running", "stopping"].includes(state.currentJob?.status);
    elements.launchButton.disabled = active;
    elements.launchButton.querySelector("span").textContent = active ? "训练任务运行中" : "下发训练任务";
  }
}

/** 请求后端安全终止当前训练进程组。 */
async function stopJob() {
  if (!state.currentJobId || !state.currentJob?.can_stop) return;
  elements.stopButton.disabled = true;
  try {
    const job = await api(`/api/jobs/${state.currentJobId}/stop`, { method: "POST", body: "{}" });
    renderTelemetry(job);
    showToast("已发送安全停止请求");
  } catch (error) {
    showToast(error.message, true);
  }
}

/** 从人工复核点恢复当前任务，复用既有策略、预算和已采集的评估数据。 */
async function resumeJob() {
  if (!state.currentJobId || !state.currentJob?.can_resume) return;
  elements.resumeButton.disabled = true;
  try {
    const job = await api(`/api/jobs/${state.currentJobId}/resume`, {
      method: "POST", body: "{}",
    });
    state.currentJobId = job.job_id;
    state.logOffset = 0;
    elements.logOutput.textContent = "";
    elements.terminalWelcome.classList.remove("hidden");
    localStorage.setItem("rl-current-job", job.job_id);
    renderTelemetry(job);
    await pollLog();
    await refreshHistory();
    showToast(`已从 ${job.task_id} 的当前策略恢复闭环`);
  } catch (error) {
    showToast(error.message, true);
    if (state.currentJob) renderTelemetry(state.currentJob);
  }
}

/** 渲染来自实验目录的最近训练记录。 */
function renderHistory(experiments) {
  if (!experiments.length) {
    elements.historyList.innerHTML = '<div class="history-empty">还没有实验记录</div>';
    return;
  }
  elements.historyList.innerHTML = experiments.map((experiment) => {
    const failed = ["FAILED", "HUMAN_REVIEW"].includes(experiment.state);
    return `<div class="history-item">
      <div class="history-row">
        <div class="history-main">
          <h3 title="${escapeHtml(experiment.task)}">${escapeHtml(experiment.task)}</h3>
          <p>${escapeHtml(experiment.task_id)} · ${escapeHtml(String(experiment.robot).toUpperCase())} · ${formatTime(experiment.updated_at)}</p>
        </div>
        <span class="history-badge ${failed ? "failed" : ""}">${escapeHtml(experiment.stage_label)}</span>
      </div>
    </div>`;
  }).join("");
}

/** 刷新界面作业与历史实验，并在初次打开时恢复最后作业。 */
async function refreshHistory() {
  try {
    const result = await api("/api/jobs");
    renderHistory(result.experiments || []);
    if (state.currentJobId && !result.jobs?.some((job) => job.job_id === state.currentJobId)) {
      state.currentJobId = null;
      localStorage.removeItem("rl-current-job");
    }
    if (!state.currentJobId && result.jobs?.length) {
      state.currentJobId = result.jobs[0].job_id;
      localStorage.setItem("rl-current-job", state.currentJobId);
      renderTelemetry(result.jobs[0]);
      await pollLog();
    }
  } catch (error) {
    elements.historyList.innerHTML = `<div class="history-empty">${escapeHtml(error.message)}</div>`;
  }
}

/** 绑定页面控件事件，包括快捷示例、模式、日志和训练控制。 */
function bindEvents() {
  elements.taskInput.addEventListener("input", () => {
    elements.charCount.textContent = `${elements.taskInput.value.length} / 2000`;
  });
  elements.taskInput.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter") launchJob();
  });
  document.querySelectorAll("[data-example]").forEach((button) => button.addEventListener("click", () => {
    elements.taskInput.value = button.dataset.example;
    elements.taskInput.dispatchEvent(new Event("input"));
    elements.taskInput.focus();
  }));
  elements.robotGrid.addEventListener("click", (event) => {
    const card = event.target.closest("[data-robot]");
    if (card) selectRobot(card.dataset.robot);
  });
  elements.modeSwitch.addEventListener("click", (event) => {
    const button = event.target.closest("[data-mode]");
    if (button) selectMode(button.dataset.mode);
  });
  elements.launchButton.addEventListener("click", launchJob);
  elements.resumeButton.addEventListener("click", resumeJob);
  elements.stopButton.addEventListener("click", stopJob);
  elements.refreshButton.addEventListener("click", refreshHistory);
  elements.autoScrollButton.addEventListener("click", () => {
    state.autoScroll = !state.autoScroll;
    elements.autoScrollButton.classList.toggle("active", state.autoScroll);
    elements.autoScrollButton.textContent = state.autoScroll ? "自动跟随" : "暂停跟随";
  });
  elements.clearLogButton.addEventListener("click", () => {
    elements.logOutput.textContent = "";
    elements.terminalWelcome.classList.remove("hidden");
  });
  elements.copyLogButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(elements.logOutput.textContent);
      showToast("日志已复制");
    } catch (error) {
      showToast("浏览器未授权复制日志", true);
    }
  });
}

/** 加载上位机配置、恢复作业并启动周期轮询。 */
async function initialize() {
  bindEvents();
  updateClock();
  window.setInterval(updateClock, 1000);
  try {
    state.config = await api("/api/config");
    state.robot = state.config.default_robot;
    renderRobots(state.config.robots);
    const projectReady = Boolean(state.config.system?.training_project_ready && state.config.system?.training_entry_ready);
    elements.projectSignal.classList.toggle("online", projectReady);
    elements.projectStatus.textContent = projectReady ? "训练工程就绪" : "训练工程未就绪";
  } catch (error) {
    elements.projectStatus.textContent = "控制服务异常";
    showToast(error.message, true);
  }
  await refreshHistory();
  if (state.currentJobId) await pollCurrentJob();
  state.pollTimer = window.setInterval(pollCurrentJob, 1000);
}

initialize();
