class ProviderError(RuntimeError):
    """浏览器推理失败的基础异常。"""


class ProviderNeedsHuman(ProviderError):
    """登录、验证码或页面恢复需要人工介入。"""


class ProviderTimeout(ProviderError):
    """网页未在限定时间内产生完整回复。"""


class ProviderResponseError(ProviderError):
    """回复经过修复尝试后仍无法通过校验。"""
