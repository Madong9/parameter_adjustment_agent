# OpenCLI ChatGPT 网页集成

检查到的 OpenCLI 同时提供 ChatGPT 适配器和底层 browser 驱动。本 Provider 使用 browser 驱动完成固定会话状态检查、稳定 CSS 与语义输入框定位、带验证的填充、显式发送按钮点击、用户消息提交确认、回复轮询、最新助手消息提取和文件上传。源码不保存任何数字元素 ref；导航或页面变化后会重新检查状态。

配置位于 `config/opencli.yaml`。绑定模式会连接用户当前标签页，并在关闭时解除绑定；owned 模式会创建并释放 Agent 自有标签页。命令超时、提交确认超时和回复超时分别配置，所有重试都有明确上限。只有尚未确认点击发送的页面操作可以重试；发送按钮点击成功后不会重复提交消息。

文本发送顺序固定为：验证提示词完整填入、等待发送按钮解除禁用、显式点击按钮、确认新的用户消息出现在会话中，最后等待助手回复。OpenCLI 的精确校验失败时，Provider 会在页面内对 ProseMirror 展开的段落空白进行规范化，然后严格比较全文、规范化长度及首尾；不会用截断前缀代替完整校验。任务设计环境清单会删除重复观察项和源码路径，但保留变量、形状、单位、坐标系、奖励依赖和终止条件，当前输入量减少约 20%。

JSON 处理顺序如下：先提取 JSON Markdown 代码块，再查找首个括号平衡的 JSON 对象，最后执行 Pydantic 校验。校验失败时，Provider 会在同一个网页会话中发送 Schema 和错误信息，请求有限次数的修复。检测到验证码或登录过期后会立即进入人工恢复状态。浏览器日志不会读取 Cookie、本地存储或网络凭据。

检查命令：

```bash
opencli list
opencli doctor
opencli chatgpt status -f json
python -m rl_training_agent opencli-test
```

集成记录位于 `artifacts/opencli_test/records/`，只包含提示词和助手文本，不包含凭据。

在当前工作站上，真实文本往返已经通过。2026-07-17 发现 ChatGPT 富文本输入框填入后依赖通用 Enter 事件会出现未提交问题，现已改为显式点击发送按钮并验证用户消息。2026-07-18 确认 OpenCLI 1.8.6 与扩展 1.0.22 执行 `set-file-input` 时会返回 CDP `Not allowed`；Provider 现在会自动把二进制文件编码为小于系统单参数限制的分块，在页面内重组为 `File`，再通过通用 `#upload-files` 输入框和 DataTransfer 触发 React 附件事件。文档与图片会分开附加，图片预览确认后才发送消息。

2026-07-18 已使用真实训练产物 `contact_sheet_clean.png`、`contact_sheet_annotated.png` 和 `event_landing.png` 完成端到端视觉评估，ChatGPT 成功返回并通过 `VisualBehaviorReport` 校验。视觉工具调用可能接近三分钟，因此默认 `response_timeout` 为 300 秒；等待逻辑会识别“停止回答”按钮，在视觉工具仍运行时不会把暂时稳定的文本误判为最终回复。
