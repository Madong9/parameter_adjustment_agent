"""集中定义新旧 ChatGPT 网页的消息快照，供提交确认和回复读取共同使用。"""

# React onChange 可以同步清空 input.files；返回交给处理器的文件清单，
# 页面是否接收附件仍由编辑器预览验证，不能靠这里的 ok 判断。
CHATGPT_ATTACH_FILES_SCRIPT = r"""
const attachFiles = (input, transfer) => {
  input.files = transfer.files;
  const names = Array.from(input.files).map(file => file.name);
  const propsKey = Object.keys(input).find(key => key.startsWith('__reactProps$'));
  if (propsKey && input[propsKey] && typeof input[propsKey].onChange === 'function') {
    const nativeEvent = new Event('change', {bubbles: true});
    input[propsKey].onChange({
      target: input, currentTarget: input, nativeEvent,
      preventDefault() {}, stopPropagation() {},
      isDefaultPrevented() { return false; },
      isPropagationStopped() { return false; }, persist() {}
    });
  } else {
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
  }
  return {ok: true, count: names.length, names};
};
"""

CHATGPT_SNAPSHOT_SCRIPT = r"""
(() => {
  const roleOf = node => {
    const explicit = node.getAttribute('data-message-author-role');
    if (explicit === 'user' || explicit === 'assistant') return explicit;
    const key = node.getAttribute('data-chatgpt-search-unit-key') ||
      node.getAttribute('data-content-search-unit-key') || '';
    return key.endsWith(':assistant') ? 'assistant' : key.endsWith(':user') ? 'user' : '';
  };
  // 新网页按搜索单元标记消息，旧网页按作者角色标记。绝不从整页正文提取 JSON。
  const candidates = Array.from(document.querySelectorAll(
    '[data-message-author-role="user"], [data-message-author-role="assistant"], ' +
    '[data-chatgpt-search-unit-key$=":user"], [data-chatgpt-search-unit-key$=":assistant"], ' +
    '[data-content-search-unit-key$=":user"], [data-content-search-unit-key$=":assistant"]'));
  const roots = candidates.filter(node => !candidates.some(other =>
    other !== node && roleOf(other) === roleOf(node) && other.contains(node)));
  const readText = node => {
    const copy = node.cloneNode(true);
    copy.querySelectorAll('button, script, style, textarea, [contenteditable="true"], ' +
      '[data-conversation-role], [aria-hidden="true"]').forEach(item => item.remove());
    copy.querySelectorAll('h1,h2,h3,h4,h5,h6').forEach(item => {
      if (/^(你说|您说|You said|ChatGPT\s*说|ChatGPT said|ChatGPT says)\s*[:：]?$/i
          .test(item.textContent.trim())) item.remove();
    });
    const walk = item => {
      if (item.nodeType === 3) return item.textContent || '';
      if (item.nodeType !== 1) return '';
      if (item.tagName === 'BR') return '\n';
      if (item.tagName === 'PRE' || item.tagName === 'CODE') return item.textContent || '';
      const text = Array.from(item.childNodes).map(walk).join('');
      return /^(P|DIV|LI|SECTION|ARTICLE|H[1-6])$/.test(item.tagName) ? text + '\n' : text;
    };
    return walk(copy).trim();
  };
  let lastUserId = '';
  const messages = roots.map((node, index) => {
    const role = roleOf(node);
    const id = node.getAttribute('data-message-id') ||
      node.getAttribute('data-chatgpt-search-message-ids') ||
      node.getAttribute('data-chatgpt-search-unit-key') ||
      node.getAttribute('data-content-search-unit-key') || role + ':' + index;
    if (role === 'user') lastUserId = id;
    return {role, id, text: readText(node), user_id: lastUserId};
  });
  const users = messages.filter(item => item.role === 'user');
  const assistants = messages.filter(item => item.role === 'assistant');
  const user = users.slice(-1)[0] || {};
  const assistant = assistants.slice(-1)[0] || {};
  const composer = document.querySelector(
    '#prompt-textarea, [data-testid="prompt-textarea"], [contenteditable="true"][role="textbox"]');
  return JSON.stringify({
    latest_user: user.text || '', user_count: users.length, latest_user_id: user.id || '',
    latest_assistant: assistant.text || '', assistant_count: assistants.length,
    latest_assistant_id: assistant.id || '', assistant_user_id: assistant.user_id || '',
    composer_found: Boolean(composer),
    composer_text: composer ? (composer.isContentEditable ?
      (composer.innerText || composer.textContent || '') : String(composer.value || '')) : '',
    url: location.href, extraction_version: 'chatgpt-dom-v2',
    generating: Boolean(document.querySelector('button[data-testid="stop-button"]')) ||
      Array.from(document.querySelectorAll('button')).some(button =>
        /Stop generating|Stop responding|停止生成|停止回答|Thinking|正在思考|^(?:停止|Stop)$/i.test(
          (button.getAttribute('aria-label') || '').trim()))
  });
})()
"""
