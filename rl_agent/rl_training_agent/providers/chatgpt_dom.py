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

# 只读取站点的可见控件，不把用户消息或模型正文当作登录、验证码、额度证据。
CHATGPT_ACCESS_STATE_SCRIPT = r"""
(() => {
  const visible = node => Boolean(node && node.getClientRects().length &&
    getComputedStyle(node).visibility !== 'hidden' && getComputedStyle(node).display !== 'none');
  const messageSelector = '[data-message-author-role], [data-chatgpt-search-unit-key], ' +
    '[data-content-search-unit-key]';
  const outsideMessage = node => !node.closest(messageSelector);
  const controls = Array.from(document.querySelectorAll('button, a, input')).filter(node =>
    visible(node) && outsideMessage(node));
  const label = node => (node.getAttribute('aria-label') || node.textContent || '').trim();
  const composer = Array.from(document.querySelectorAll(
    '#prompt-textarea, [data-testid="prompt-textarea"], [contenteditable="true"][role="textbox"]'))
    .find(visible);
  const account = controls.some(node => /^(profile-button|user-menu-button)$/.test(
    node.getAttribute('data-testid') || '') ||
    /^(Open profile menu|Profile|Account|打开个人资料菜单|个人资料|账户菜单)$/i.test(label(node)));
  const login = controls.filter(node =>
    /^(Log in|Sign in|Sign up|Log in to ChatGPT|登录|登入|立即登录|登录 ChatGPT|注册)$/i.test(label(node)) ||
    /^(login-button|signup-button)$/.test(node.getAttribute('data-testid') || '') ||
    /\/auth\/(login|signup)(?:[/?#]|$)/.test(node.getAttribute('href') || ''));
  const authForm = Array.from(document.querySelectorAll('input[type="password"], ' +
    'form input[autocomplete="username"]')).some(node => visible(node) && outsideMessage(node));
  const authDialog = login.some(node => node.closest('[role="dialog"]'));
  const challenge = Array.from(document.querySelectorAll(
    'iframe[src*="challenges.cloudflare.com"], iframe[src*="recaptcha"], ' +
    '[name="cf-turnstile-response"], #challenge-running, #challenge-stage'))
    .some(node => visible(node) && outsideMessage(node));
  const uiNotices = Array.from(document.querySelectorAll(
    '[role="alert"], [role="dialog"], [data-testid="usage-limit-banner"], ' +
    '[data-testid="rate-limit-banner"]')).filter(node => visible(node) && outsideMessage(node));
  const usageNotice = uiNotices.map(node => {
    const copy = node.cloneNode(true);
    copy.querySelectorAll(messageSelector + ', textarea, [contenteditable="true"], pre, code')
      .forEach(item => item.remove());
    return String(copy.textContent || '').slice(0, 2000);
  }).join('\n');
  return JSON.stringify({probe_version: 'chatgpt-access-v1', url: location.href,
    composer_found: Boolean(composer), account_controls_found: account,
    login_required: authForm || authDialog || (login.length > 0 && !account),
    challenge_required: challenge, usage_notice: usageNotice});
})()
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
