"use strict";

const state = {
  config: null,
  robot: "go2",
  mode: "dry-run",
  currentJobId: localStorage.getItem("rl-current-job") || null,
  currentJob: null,
  currentJobStatus: null,
  playback: { running: false },
  playableExperiments: [],
  memory: {},
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
  reviewReason: document.querySelector("#reviewReason"),
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
  playStrategySelect: document.querySelector("#playStrategySelect"),
  playStrategyButton: document.querySelector("#playStrategyButton"),
  playbackNote: document.querySelector("#playbackNote"),
  feasibilityStatus: document.querySelector("#feasibilityStatus"),
  feasibilitySummary: document.querySelector("#feasibilitySummary"),
  feasibilityMotion: document.querySelector("#feasibilityMotion"),
  feasibilityConstraint: document.querySelector("#feasibilityConstraint"),
  feasibilityPlanning: document.querySelector("#feasibilityPlanning"),
  feasibilityWholeBody: document.querySelector("#feasibilityWholeBody"),
  feasibilityPhysics: document.querySelector("#feasibilityPhysics"),
  feasibilityComponents: document.querySelector("#feasibilityComponents"),
  feasibilityReason: document.querySelector("#feasibilityReason"),
  feasibilityViewButton: document.querySelector("#feasibilityViewButton"),
  memoryEnabled: document.querySelector("#memoryEnabled"),
  memoryWorking: document.querySelector("#memoryWorking"),
  memoryEpisodic: document.querySelector("#memoryEpisodic"),
  memorySemantic: document.querySelector("#memorySemantic"),
  memoryProcedural: document.querySelector("#memoryProcedural"),
  memoryPromotion: document.querySelector("#memoryPromotion"),
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
    ? "<span aria-hidden=\"true\">◇</span><p><b>真实训练模式</b>将由 GPT 理解动作、百炼 GLM 设计奖励，再启动 GPU 仿真。实体机器人部署不在此界面范围内。</p>"
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

/** 将内部英文证据状态转换为简洁中文，保留未知值供排障。 */
function evidenceLabel(value) {
  return {
    PENDING: "等待生成", PASSED: "已通过", FAILED: "未通过", CONDITIONAL: "有条件",
    GENERATED: "已生成", READY_FOR_SOLVER: "规划就绪", READY_FOR_PHYSICS: "求解就绪",
    LOCOMOTION: "直线行走", STATIC_POSE: "静态姿态", TURNING: "转向", BALANCE: "平衡动作",
    JUMP: "跳跃", ACROBATIC: "高动态动作", MANIPULATION: "操作动作",
    INCONCLUSIVE: "证据不足", NOT_RUN: "未运行", UNKNOWN: "未知",
    CAPABILITY_ONLY: "仅能力检查", MOCK_VALIDATED: "仅 Mock",
    STATIC_PHYSICS_VALIDATED: "静态物理通过", DYNAMIC_PHYSICS_VALIDATED: "动态物理通过",
    TRAINING_READY: "可进入训练", PHYSICS_FAILED: "本次物理探针未通过",
    ALLOW_TRAINING: "允许仿真训练", ALLOW_BUDGETED_EXPLORATION: "允许预算探索",
    DRY_RUN_ONLY: "仅离线演练", NEEDS_REVIEW: "准入待复核", REJECT: "能力不支持",
    REFERENCE_TRACKING_FAILED: "参考轨迹跟踪失败", CANDIDATE_ROLLOUT_FAILED: "候选执行失败",
    RUNTIME_INVALID: "仿真出现非有限数值",
    OPTIMIZATION_INCONCLUSIVE: "候选求解证据不足", BACKEND_UNAVAILABLE: "探针后端不可用",
    CAPABILITY_SUPPORTED: "能力支持", PHYSICS_VALIDATED: "物理通过",
    MODEL_UNAVAILABLE: "模型不可用", NEEDS_CLARIFICATION: "需要澄清",
    UNSUPPORTED: "不支持",
  }[value] || value || "—";
}

/** 渲染训练前可行性证据链，并只为受支持的真实探针开放 Viewer。 */
function renderFeasibility(feasibility = {}) {
  const available = Boolean(feasibility.available);
  const validation = feasibility.validation_level || feasibility.status || "PENDING";
  const passed = ["STATIC_PHYSICS_VALIDATED", "DYNAMIC_PHYSICS_VALIDATED", "TRAINING_READY", "PHYSICS_VALIDATED"].includes(validation);
  const failed = ["PHYSICS_FAILED", "FAILED", "UNSUPPORTED", "MODEL_UNAVAILABLE"].includes(validation) ||
    ["PHYSICS_FAILED", "UNSUPPORTED", "MODEL_UNAVAILABLE"].includes(feasibility.status);
  elements.feasibilityStatus.className = `insight-badge ${passed ? "success" : failed ? "failed" : available ? "warning" : "pending"}`;
  elements.feasibilityStatus.textContent = available ? evidenceLabel(validation) : "等待报告";
  elements.feasibilityMotion.textContent = evidenceLabel(feasibility.motion_type);
  elements.feasibilityConstraint.textContent = feasibility.constraint_ready ? "已生成" : "等待生成";
  elements.feasibilityPlanning.textContent = evidenceLabel(feasibility.planner_status);
  elements.feasibilityWholeBody.textContent = evidenceLabel(feasibility.whole_body_status);
  elements.feasibilityPhysics.textContent = evidenceLabel(feasibility.physics_status);
  elements.feasibilitySummary.textContent = available
    ? `证据层级 ${evidenceLabel(feasibility.feasibility_level)} · 后端 ${feasibility.backend || "未运行"} · 置信度 ${Number(feasibility.confidence || 0).toFixed(2)}`
    : "下发任务后，将显示从动作约束、确定性规划、全身求解到物理验证的证据等级。";
  const admission = feasibility.training_admission || {};
  if (admission.decision) {
    elements.feasibilitySummary.textContent += ` · 训练准入：${evidenceLabel(admission.decision)}`;
    if (admission.decision === "ALLOW_BUDGETED_EXPLORATION") {
      elements.feasibilitySummary.textContent += ` · 总预算 ${admission.max_iterations} 次迭代 / ${admission.max_revisions} 次修订`;
    }
  }
  const components = Object.entries(feasibility.component_status || {});
  elements.feasibilityComponents.innerHTML = components.length
    ? components.map(([name, status]) => `<span class="${status === "PASSED" ? "passed" : status === "FAILED" ? "failed" : ""}">${escapeHtml(name)} · ${escapeHtml(evidenceLabel(status))}</span>`).join("")
    : '<span>规划组件尚无结果</span>';
  const missing = feasibility.missing_solvers || [];
  const notes = [...missing.map((item) => `缺少求解器：${item}`), ...(feasibility.limitations || [])];
  elements.feasibilityReason.textContent = admission.reason || feasibility.recommended_next_step || notes[0] ||
    "这里显示的是训练前可行性证据，不代表机器人已经学会动作。";
  const viewerRunning = state.playback.running && state.playback.kind === "feasibility";
  const trainingActive = ["queued", "running", "stopping"].includes(state.currentJob?.status);
  elements.feasibilityViewButton.disabled = !viewerRunning && (!feasibility.viewer_available || trainingActive || state.playback.running);
  elements.feasibilityViewButton.textContent = viewerRunning ? "■ 停止可行性 Viewer" : "在 Isaac Gym 中观察检验";
}

/** 渲染全局四层记忆统计与当前实验的工作记忆、晋升结论。 */
function renderMemory(memory = {}, detail = {}) {
  state.memory = memory;
  const enabled = Boolean(memory.enabled);
  elements.memoryEnabled.className = `insight-badge ${enabled ? "success" : "failed"}`;
  elements.memoryEnabled.textContent = enabled ? "记忆已启用" : "记忆已关闭";
  elements.memoryWorking.textContent = detail.working_available
    ? `${evidenceLabel(detail.working_state)} · 第${Number(detail.loop_round || 0)}轮 · v${Number(detail.reward_version || 0)}`
    : "等待任务";
  elements.memoryEpisodic.textContent = `${Number(memory.episodic_active || 0)} 活跃 / ${Number(memory.episodic_archived || 0)} 归档`;
  elements.memorySemantic.textContent = `${Number(memory.semantic_active || 0)} 生效 / ${Number(memory.semantic_candidates || 0)} 候选`;
  elements.memoryProcedural.textContent = memory.procedural_snapshot ? "快照已生成" : "未生成";
  elements.memoryPromotion.textContent = detail.promoted
    ? `当前任务已晋升为情景记忆：${detail.memory_id || "已写入"}`
    : `当前任务未晋升：${detail.promotion_reason || "尚未完成证据门控"}`;
}

/** 刷新长期记忆统计；记忆正文不会通过界面接口返回。 */
async function pollMemory() {
  try {
    renderMemory(await api("/api/memory"), state.currentJob?.memory_detail || {});
  } catch (error) {
    console.warn("轮询记忆状态失败", error);
  }
}

/** 启动或停止当前任务的 Isaac Gym 可行性观察窗口。 */
async function toggleFeasibilityViewer() {
  if (state.playback.running && state.playback.kind === "feasibility") {
    try {
      renderPlaybackStatus(await api("/api/play/stop", { method: "POST" }));
      showToast("已请求停止可行性 Viewer");
    } catch (error) {
      showToast(error.message, true);
    }
    return;
  }
  if (!state.currentJobId) return;
  elements.feasibilityViewButton.disabled = true;
  try {
    const result = await api("/api/feasibility-view", {
      method: "POST", body: JSON.stringify({ job_id: state.currentJobId }),
    });
    showToast(`可行性 Viewer 已启动 · ${result.task_id}`);
    await pollPlayback();
  } catch (error) {
    showToast(error.message, true);
    renderFeasibility(state.currentJob?.feasibility || {});
  }
}

/** 用最新作业数据刷新进度仪表、可行性和记忆状态。 */
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
  const hasReviewReason = job.status === "review" && Boolean(job.review_reason);
  elements.reviewReason.textContent = hasReviewReason ? `复核原因：${job.review_reason}` : "";
  elements.reviewReason.classList.toggle("hidden", !hasReviewReason);
  const active = ["queued", "running", "stopping"].includes(job.status);
  elements.launchButton.disabled = active;
  elements.launchButton.querySelector("span").textContent = active ? "训练任务运行中" : "下发训练任务";
  elements.stageTrack.querySelectorAll("[data-threshold]").forEach((stage) => {
    stage.classList.toggle("done", Number(job.progress) >= Number(stage.dataset.threshold));
  });
  renderFeasibility(job.feasibility || {});
  renderMemory(state.memory, job.memory_detail || {});
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
    if (["completed", "failed", "review", "stopped", "interrupted"].includes(job.status) &&
        state.currentJobStatus !== job.status) {
      await refreshHistory();
    }
    state.currentJobStatus = job.status;
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

/** 渲染只包含真实训练联合验收通过的可播放策略下拉列表。 */
function renderPlayableExperiments(experiments, playback = {}) {
  state.playableExperiments = experiments;
  const previous = elements.playStrategySelect.value;
  const options = ['<option value="">选择已通过验收的策略</option>'];
  experiments.forEach((experiment) => {
    const label = `${experiment.task} · ${String(experiment.robot).toUpperCase()} · ${experiment.task_id}`;
    options.push(`<option value="${escapeHtml(experiment.task_id)}">${escapeHtml(label)}</option>`);
  });
  elements.playStrategySelect.innerHTML = options.join("");
  if (experiments.some((experiment) => experiment.task_id === previous)) {
    elements.playStrategySelect.value = previous;
  }
  renderPlaybackStatus(playback);
}

/** 根据播放器子进程状态恢复策略选择，或展示停止入口。 */
function renderPlaybackStatus(playback = {}) {
  state.playback = playback;
  const running = Boolean(playback.running);
  const strategyRunning = running && playback.kind !== "feasibility";
  const feasibilityRunning = running && playback.kind === "feasibility";
  elements.playStrategySelect.disabled = running;
  elements.playStrategyButton.disabled = feasibilityRunning || (!strategyRunning && !elements.playStrategySelect.value);
  elements.playStrategyButton.textContent = strategyRunning ? "■ 停止播放" : feasibilityRunning ? "Viewer 已占用" : "▶ 播放策略";
  elements.playbackNote.textContent = strategyRunning
    ? `正在播放 ${playback.task_id}；关闭 Viewer 窗口或点击“停止播放”即可结束。`
    : feasibilityRunning
      ? `正在观察 ${playback.task_id} 的可行性探针；请在“动作可行性”面板停止。`
      : state.playableExperiments.length
        ? "仅显示已通过联合验收的真实训练策略；播放将在仿真 Viewer 中运行。"
        : "还没有可播放策略。离线演练、人工复核和失败实验不会出现在这里。";
  renderFeasibility(state.currentJob?.feasibility || {});
}

/** 在 Isaac Gym 仿真 Viewer 中播放用户选中的已验收策略。 */
async function playSelectedStrategy() {
  if (state.playback.running && state.playback.kind === "feasibility") return;
  if (state.playback.running) {
    try {
      const status = await api("/api/play/stop", { method: "POST" });
      renderPlaybackStatus(status);
      showToast("已请求停止策略播放");
    } catch (error) {
      showToast(error.message, true);
    }
    return;
  }
  const taskId = elements.playStrategySelect.value;
  if (!taskId) return;
  elements.playStrategyButton.disabled = true;
  try {
    const result = await api("/api/play", {
      method: "POST",
      body: JSON.stringify({ task_id: taskId, seed: 1, num_envs: 1 }),
    });
    showToast(`策略已开始播放 · ${result.task_id}`);
    await refreshHistory();
  } catch (error) {
    showToast(error.message, true);
    elements.playStrategyButton.disabled = false;
  }
}

/** 定时同步 Viewer 进程状态，让关闭窗口后策略列表自动恢复可选。 */
async function pollPlayback() {
  try {
    renderPlaybackStatus(await api("/api/playback"));
  } catch (error) {
    console.warn("轮询策略播放状态失败", error);
  }
}

/** 刷新界面作业与历史实验，并在初次打开时恢复最后作业。 */
async function refreshHistory() {
  try {
    const result = await api("/api/jobs");
    renderHistory(result.experiments || []);
    renderPlayableExperiments(result.playable_experiments || [], result.playback || {});
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
  elements.playStrategySelect.addEventListener("change", () => {
    elements.playStrategyButton.disabled = !elements.playStrategySelect.value;
  });
  elements.playStrategyButton.addEventListener("click", playSelectedStrategy);
  elements.feasibilityViewButton.addEventListener("click", toggleFeasibilityViewer);
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
  window.setInterval(pollPlayback, 1000);
  window.setInterval(pollMemory, 5000);
  try {
    state.config = await api("/api/config");
    state.robot = state.config.default_robot;
    renderRobots(state.config.robots);
    const projectReady = Boolean(state.config.system?.training_project_ready && state.config.system?.training_entry_ready);
    const rag = state.config.system?.rag || {};
    const memory = state.config.system?.memory || {};
    renderMemory(memory, state.currentJob?.memory_detail || {});
    const providers = state.config.system?.providers || {};
    const modelStatus = providers.bailian_model
      ? `${providers.bailian_model}${providers.bailian_configured ? "" : " 未配置"}`
      : "GLM";
    elements.projectSignal.classList.toggle("online", projectReady);
    elements.projectStatus.textContent = projectReady
      ? `训练工程就绪 · RAG ${Number(rag.chunks || 0)} · 情景记忆 ${Number(memory.episodic_active || 0)} · 语义记忆 ${Number(memory.semantic_active || 0)} · ${modelStatus}`
      : "训练工程未就绪";
  } catch (error) {
    elements.projectStatus.textContent = "控制服务异常";
    showToast(error.message, true);
  }
  await refreshHistory();
  if (state.currentJobId) await pollCurrentJob();
  state.pollTimer = window.setInterval(pollCurrentJob, 1000);
}

initialize();
