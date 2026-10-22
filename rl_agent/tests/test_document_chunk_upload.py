"""验证长需求文档按原始字节分块上传、确认协议和失败清理，不连接浏览器。"""
import base64
import errno
import json
import re

import pytest

from rl_training_agent.providers.errors import ProviderError, ProviderTimeout
from rl_training_agent.providers.opencli_chatgpt import OpenCLIChatGPTWebProvider


class DocumentUploadPage:
    """模拟页面分块存储与确认结果，并独立还原最终附件字节。"""

    def __init__(self, failures=None):
        self.failures = failures or {}
        self.scripts = []
        self.events = []
        self.files = []
        self.uploaded = []
        self.cleaned = False

    def __call__(self, args, allow_failure=False):
        assert args[0] == 'eval'
        script = args[1]
        self.scripts.append(script)
        if 'delete window.__rlAgentFileUploads' in script:
            stage = 'cleanup'
        elif '= {files: []}' in script:
            stage = 'initialize'
        elif 'upload.files.push(' in script:
            stage = 'register'
        elif '.chunks.push(' in script:
            stage = 'chunk'
        elif 'new File' in script:
            stage = 'commit'
        elif 'media.length' in script:
            stage = 'preview'
        else:
            raise AssertionError('Unexpected browser operation')
        self.events.append(stage)
        if stage in self.failures:
            failure = self.failures[stage]
            if isinstance(failure, Exception):
                raise failure
            return json.dumps(failure)
        if stage == 'register':
            descriptor = json.loads(script.split('upload.files.push(', 1)[1].split(');', 1)[0])
            self.files.append(descriptor)
            reply = {'ok': True, 'files': len(self.files)}
        elif stage == 'chunk':
            match = re.search(r'upload\.files\[(\d+)\]\.chunks\.push\(("[^"]*")\);', script)
            assert match is not None
            descriptor = self.files[int(match.group(1))]
            descriptor['chunks'].append(json.loads(match.group(2)))
            reply = {'ok': True, 'chunks': len(descriptor['chunks'])}
        elif stage == 'commit':
            for descriptor in self.files:
                raw = base64.b64decode(''.join(descriptor['chunks']), validate=True)
                assert len(raw) == descriptor['size']
                self.uploaded.append((descriptor['name'], descriptor['type'], raw))
            reply = {'ok': True, 'count': len(self.files), 'names': [item['name'] for item in self.files]}
        elif stage == 'preview':
            assert 'if (!false)' in script
            reply = True
        else:
            if stage == 'cleanup':
                assert allow_failure is True
                self.cleaned = True
            reply = {'ok': True}
        return json.dumps(reply)


def provider_with_page(monkeypatch, failures=None):
    """仅替换浏览器传输，其余分块、确认和预览流程使用生产实现。"""
    provider = OpenCLIChatGPTWebProvider()
    page = DocumentUploadPage(failures)
    monkeypatch.setattr(provider, '_browser', page)
    return provider, page


def test_long_chinese_document_preserves_bytes_with_bounded_eval(monkeypatch, tmp_path):
    """超过单参数上限的中文及 emoji 文档跨 UTF-8 字符分块后仍逐字节一致。"""
    document = tmp_path / '奖励设计_需求文档.md'
    original = ('A' + '中文🙂\n' * 20000).encode('utf-8')
    document.write_bytes(original)
    assert len(original) > 128 * 1024
    provider, page = provider_with_page(monkeypatch)
    provider._upload_text_documents([document])
    assert page.uploaded == [(document.name, 'text/markdown', original)]
    assert len(page.files[0]['chunks']) > 3
    first_chunk = base64.b64decode(page.files[0]['chunks'][0])
    with pytest.raises(UnicodeDecodeError):
        first_chunk.decode('utf-8')
    assert all(len(script.encode('utf-8')) <= 64 * 1024 for script in page.scripts)
    assert page.events[-2:] == ['preview', 'cleanup'] and page.cleaned
    commit = next(script for script in page.scripts if 'new File' in script)
    assert 'new File([bytes]' in commit


def test_multiple_documents_and_empty_file_keep_order_and_mime(monkeypatch, tmp_path):
    """每个文档独立确认分块数，空文件不造块，最终附件顺序和类型保持一致。"""
    markdown = tmp_path / '说明.md'
    data = tmp_path / '配置.json'
    empty = tmp_path / '空文件.txt'
    markdown.write_text('奖励配置说明' * 15000, encoding='utf-8')
    data.write_text('{"目标":"稳定站立"}\n', encoding='utf-8')
    empty.write_bytes(b'')
    provider, page = provider_with_page(monkeypatch)
    provider._upload_text_documents([markdown, data, empty])
    assert page.uploaded == [
        (markdown.name, 'text/markdown', markdown.read_bytes()),
        (data.name, 'application/json', data.read_bytes()),
        (empty.name, 'text/plain', b''),
    ]
    assert page.files[2]['chunks'] == []
    assert page.events.count('register') == 3 and page.events.count('commit') == 1
    assert all(len(script.encode('utf-8')) <= 64 * 1024 for script in page.scripts)
    assert page.cleaned


@pytest.mark.parametrize('stage,acknowledgement,detail', [
    ('initialize', {'ok': False}, '初始化文档上传'),
    ('register', {'ok': True}, '注册文档'),
    ('chunk', {'ok': True}, '第 1 块'),
    ('chunk', {'ok': True, 'chunks': 2}, '第 1 块'),
    ('commit', {'ok': True, 'count': 1, 'names': ['wrong.md']}, '附加文档'),
    ('commit', {'ok': True, 'count': 0, 'names': ['requirements.md']}, '附加文档'),
])
def test_unconfirmed_upload_stops_and_cleans_session(monkeypatch, tmp_path, stage, acknowledgement, detail):
    """缺少确认、错序块及附件名不匹配均中止，不把未确认上传当作成功。"""
    document = tmp_path / 'requirements.md'
    document.write_text('严格安全约束', encoding='utf-8')
    provider, page = provider_with_page(monkeypatch, {stage: acknowledgement})
    with pytest.raises(ProviderError, match=detail):
        provider._upload_text_documents([document])
    assert page.events[-1] == 'cleanup' and page.cleaned
    assert 'preview' not in page.events


@pytest.mark.parametrize('stage,error,detail', [
    ('chunk', OSError(errno.E2BIG, 'Argument list too long'), '第 1 块'),
    ('preview', ProviderTimeout('没有显示全部附件预览'), '等待文档附件预览'),
])
def test_upload_errors_keep_stage_and_original_cause_when_cleanup_fails(
        monkeypatch, tmp_path, stage, error, detail):
    """操作系统或预览异常带上传阶段传播，清理异常不能盖掉首个原因。"""
    document = tmp_path / 'requirements.md'
    document.write_text('完整需求', encoding='utf-8')
    provider, page = provider_with_page(monkeypatch, {
        stage: error, 'cleanup': RuntimeError('cleanup unavailable'),
    })
    with pytest.raises(ProviderError, match=detail) as caught:
        provider._upload_text_documents([document])
    assert caught.value.__cause__ is error
    assert str(error) in str(caught.value) and 'cleanup unavailable' not in str(caught.value)
    assert page.events[-1] == 'cleanup'


def test_cleanup_failure_does_not_fail_confirmed_upload(monkeypatch, tmp_path):
    """附件及预览均已确认时，清理失败不改变上传成功结果。"""
    document = tmp_path / 'requirements.md'
    document.write_bytes(b'requirements')
    provider, page = provider_with_page(monkeypatch, {'cleanup': RuntimeError('cleanup unavailable')})
    provider._upload_text_documents([document])
    assert page.uploaded[0][2] == b'requirements'
    assert page.events[-2:] == ['preview', 'cleanup']
