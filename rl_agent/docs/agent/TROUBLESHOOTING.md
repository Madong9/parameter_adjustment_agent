# 故障排查

## 找不到 Python 模块

激活 `rl_agent` 环境，执行 `python -m pip install -e '.[test]'`，并从 Agent 目录运行命令。

## pytest 加载了 ROS 的 Python 3.10 插件

使用：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

## OpenCLI 扩展不可用

Provider 会在绑定标签页前执行连接预检。若 Browser Bridge 扩展断线，会自动重启一次 OpenCLI 守护进程并在 `connect_timeout` 时间内等待扩展重连，不会直接卡在 `bind` 命令直到超时。

若自动恢复后仍提示扩展未连接，请打开安装了 Browser Bridge 扩展的 Chrome，确认扩展已启用，并保持一个已登录 ChatGPT 的标签页。执行 `opencli doctor`，看到 `Extension: connected` 后重新下发任务。若扩展已连接但无法绑定，把 ChatGPT 标签页切到前台并处理登录或验证码弹窗。

## ChatGPT 输入内容后没有发送

Provider 不再使用通用 Enter 键提交。当前流程会优先定位 `#prompt-textarea`，验证填入内容，等待发送按钮可用并显式点击，再检查用户消息是否真正出现在会话中。`config/opencli.yaml` 的 `submit_timeout` 控制填入后的提交确认时间。

当提示词达到 `prompt_attachment_threshold`（默认 4000 字符）时，Provider 不再把环境清单和 JSON Schema 全部塞入输入框，而是在当前实验的 `provider_records/` 下生成 `*_需求文档.md`，上传该文档后只发送简短中文指令。首次设计请求和 Schema 校验失败后的修复请求使用同一附件通道，修复文档以 `*-repair-01_需求文档.md` 命名。文档与模型原始回复一并保留，可用于检查和复现。OpenCLI Browser Bridge 若拒绝本地文件路径并返回 `Not allowed`，Provider 会直接在页面内构造标准文本文件对象，因此需求文档上传不依赖扩展读取本地路径。

视觉评估图片采用相同的自动恢复原则。Provider 会先尝试 OpenCLI 原生上传；若 Chrome 返回 `Not allowed`、`Unknown action` 或不支持 `set-file-input`，则按 48 KiB 分块把图片传到页面，在浏览器内还原二进制并通过 DataTransfer 附加到通用 `#upload-files` 输入框。需求文档与图片分开处理，只有图片预览出现后才发送。真实错误会写入所选 rollout 的 `visual_provider_error.txt`，同时进入 `state.json` 和 `summary.json`，不再只显示笼统的“Provider 不可用”。

若视觉上疑似机身接地但旧轨迹的 `body_collision` 为 false，不要只检查机器人配置中的 `penalize_contacts_on`。增强录制器会扫描除四只脚外的全部刚体，并写入 `nonfoot_contact_force_max`；重新执行一次 `evaluate` 后再运行 `visual-audit`。这只会重新录制评估，不会重新训练。

ChatGPT 会把附件文件名和“文件”标签一起放入用户消息的 `innerText`。提交确认只允许受支持的文档文件名元数据出现在短指令之前，并仍严格比较短指令全文，避免把任意旧页面文本误判为本次提交。

ChatGPT 使用 ProseMirror 富文本编辑器，它可能把一个段落换行展开成多个空行，导致 OpenCLI 1.7.22 的逐字符校验误报失败。Provider 会对连续空白进行规范化并在页面内严格比较全文长度和首尾，确认内容完整后才允许发送；不会接受只填入一部分的提示词。

消息发送后，ChatGPT 会把 Markdown 反引号渲染为代码样式，读取用户消息的 `innerText` 时反引号本身会消失。提交确认会忽略反引号并严格比较其余全文；正文截断、字段缺失或其他字符差异仍会失败。

失败日志会同时保留 OpenCLI 的 stdout 和 stderr，代理产生的 `UNDICI-EHPA` 警告不会再遮住真实页面错误。若日志提示找不到发送按钮，请保持已登录的 ChatGPT 标签页位于前台，并执行 `opencli doctor` 和 `opencli chatgpt status -f json`。不要手动点击仍残留在输入框中的旧提示词，以免与下一次任务重复。

## ChatGPT 回复无法通过 Schema 校验

检查对应任务或候选目录下的 `prompts/`、`responses/` 和 `provider_records/`。Provider 已经执行了有限次数的同会话修复，不会无限重试。

部分网页模型会把比较运算符写成 `"operator": "<=""` 或 `"operator": ">=""`。解析器只会删除 `operator` 枚举值之后确定多出的单个引号，再执行完整 JSON 解析和 Pydantic Schema 校验；不会对其他字段进行猜测式修补。

视觉工具分析比纯文本回复慢。`config/opencli.yaml` 默认允许 300 秒，并会检测“停止回答”按钮；若超时，可确认页面没有验证码后适当提高 `response_timeout`，不要在模型仍分析图片时重复提交同一请求。

## 奖励或物理量缺失

检查 `artifacts/environment_manifest.json`。Agent 不会假装不存在的张量或奖励可用；需要新增并测试隔离实现，或修改任务规格。

## 训练退出或 GPU OOM

检查候选目录中的 `stderr.log` 和 `process_result.json`，降低环境数量，保留现有候选和 checkpoint 后恢复。不要删除父实验。

Isaac Gym 的 `torch_utils.py` 仍使用新版 NumPy 已删除的 `np.float` 等旧别名。真实训练包装器会在导入 Isaac Gym 前安装仅限当前子进程的兼容别名，无需修改或降级第三方项目。

训练后的真实 rollout 使用同一兼容层，避免训练完成后在录像评估入口再次因 `np.float` 退出。

PyTorch 首次加载 `gymtorch` 扩展时需要 Ninja。即使 Ninja 已安装在 Conda 环境中，从绝对 Python 路径启动时也可能因为 PATH 未激活而找不到它；包装器会把当前解释器的 `bin` 目录加入子进程 PATH。

训练、恢复和 rollout 命令使用启动 Agent 的 `sys.executable`，不依赖 PATH 中含义可能变化的 `python` 命令，从而保证上位机子进程始终运行在同一个 `rl_agent` 环境。

## 真实评估没有 MP4 帧

确认 CUDA 图形设备可用。离屏摄像机需要 `env.record_video`，该开关只由 rollout 包装器启用。

## 找不到 checkpoint

真实 checkpoint 格式为候选时间戳运行目录内的 `model_<iteration>.pt`。把该路径传给 `evaluate`。
