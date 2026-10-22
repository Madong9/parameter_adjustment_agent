"""从当前 Schema 提取字段类型，避免模型仅按字段名称猜测输出表示。"""
from __future__ import annotations

import json
from typing import Type

from pydantic import BaseModel


def _model_registry(schema: Type[BaseModel]):
    """收集真实嵌套模型，补充 JSON Schema 未显式声明的可空信息。"""
    models = {}
    visited = set()

    def visit_field(field):
        """从字段类型和容器子字段收集引用模型。"""
        model = getattr(field, 'type_', None)
        if isinstance(model, type) and issubclass(model, BaseModel):
            visit_model(model)
        for child in getattr(field, 'sub_fields', None) or []:
            visit_field(child)

    def visit_model(model):
        """防止递归模型重复展开，记录可查询的字段定义。"""
        if model in visited:
            return
        visited.add(model)
        models[model.__name__] = model
        for field in model.__fields__.values():
            visit_field(field)

    visit_model(schema)
    return models


def _field_type(definition, field=None):
    """保留引用、容器、枚举及显式可空类型，不生成任何业务数值。"""
    reference = definition.get('$ref')
    alternatives = definition.get('anyOf') or definition.get('oneOf')
    conjunction = definition.get('allOf')
    kind = definition.get('type')
    if reference:
        result = reference.rsplit('/', 1)[-1]
    elif alternatives:
        result = ' | '.join(_field_type(item) for item in alternatives)
    elif conjunction:
        result = ' & '.join(_field_type(item) for item in conjunction)
    elif isinstance(kind, list):
        result = ' | '.join(_field_type(dict(definition, type=item)) for item in kind)
    elif kind == 'array':
        items = definition.get('items', {})
        if isinstance(items, list):
            result = 'tuple<' + ','.join(_field_type(item) for item in items) + '>'
        else:
            children = getattr(field, 'sub_fields', None) or []
            result = 'array<' + _field_type(items, children[0] if len(children) == 1 else None) + '>'
    elif kind == 'object':
        additional = definition.get('additionalProperties')
        result = 'object<string,' + _field_type(additional) + '>' if isinstance(additional, dict) else 'object'
    else:
        result = kind or 'any'
    if 'enum' in definition:
        result += ' enum' + json.dumps(definition['enum'], ensure_ascii=False, separators=(',', ':'))
    if getattr(field, 'allow_none', False) and 'null' not in result.split(' | '):
        result += ' | null'
    return result


def render_schema_contract(schema: Type[BaseModel]) -> str:
    """生成全部字段的紧凑类型摘要，严格区分可省略与可为 null。"""
    document = schema.schema()
    models = _model_registry(schema)
    definitions = dict(document.get('definitions', {}))
    definitions[schema.__name__] = document
    summary = {}
    for name, definition in definitions.items():
        model = models.get(name)
        descriptors = {field.alias: field for field in model.__fields__.values()} if model else {}
        required = set(definition.get('required', []))
        properties = definition.get('properties')
        if not isinstance(properties, dict):
            summary[name] = {'__root__': _field_type(definition, descriptors.get('__root__'))}
            continue
        summary[name] = {
            field_name: ('required ' if field_name in required else 'optional ') +
                        _field_type(field_schema, descriptors.get(field_name))
            for field_name, field_schema in properties.items()
        }
    return 'SCHEMA_FIELD_TYPES: ' + json.dumps(summary, ensure_ascii=False, separators=(',', ':')) + (
        '\nSCHEMA_FIELD_TYPES 从本次 Schema 生成：required 必须给出，optional 可省略；'
        '只有标注 null 的字段可为 null。array<string> 的每项必须是字符串，'
        'array<模型名> 的每项必须是该模型的完整对象。其他约束以完整 JSON Schema 为准。')
