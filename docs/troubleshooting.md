# 故障排查

按"症状 → 根因 → 处理"组织。所有条目都来自实际踩过的坑。

## 症状：422 `extra_forbidden`，`loc` 指向 `assistant.reasoning_content`

**根因**：Mistral 用 `extra="forbid"` 校验请求体，而 QwenPaw 往 assistant 消息里注入了 DeepSeek 风格的 `reasoning_content`。

**两种触发路径**：

| 路径 | 条件 |
|---|---|
| A | 会话历史里存在 thinking 块，且 `relay_reasoning` 为 True（默认） |
| B | 模型 id 含 `deepseek`（只影响 thinking 裁剪，不影响注水） |

**处理**：装这个插件。它把 `reasoning_content` 在传输层翻译成 Mistral 的 chunk 列表，字段本身不再上线。

**验证**：`python3 tests/test_provider_e2e.py` —— 假服务器会对任何携带 `reasoning_content` 的请求返回 422。

---

## 症状：一用工具就 422，纯聊天正常

**根因**：工具调用轮次的 assistant 消息是 `content: null` + `tool_calls`。转换函数若要求 `content` 必须是字符串或列表，就会放弃转换，`reasoning_content` 原样上线。

**处理**：`_chunk_list_from_message` 支持 `content is None`，只发 thinking chunk。无法转换时兜底丢弃字段。

**验证**：`tests/test_provider_e2e.py` 里的 `tool-call turn` 用例。

---

## 症状：`ValidationError: choices.0.message.content Input should be a valid string`

**根因**：带了 `reasoning_effort`，Mistral 返回 chunk 列表，openai SDK 在 pydantic 校验阶段拒绝。

**处理**：装插件（传输层改写）。若临时想绕过，把 agent 的 thinking level 设为 `inherit` 或 `off`，不发 effort 就不会返回 chunk 列表。

**注意**：这不是"丢失思考内容"，是**整个请求失败**。

---

## 症状：界面上没有思考块，但模型明明支持

按顺序查三件事：

**1. agent 的 thinking level 是不是 `inherit`**

```python
# providers/provider.py:957
level = _AGENT_THINKING_LEVEL.get()
if level == "inherit" or not self.supports_agent_thinking(model_id):
    return          # 什么参数都不发
```

`inherit` 等于「不指定」，一个参数都不发。而 DeepSeek 默认**不思考**，所以看不到思考块。

检查：

```bash
python3 -c "import json;print(json.load(open('<workspace>/agent.json'))['thinking_level'])"
```

改成 `high` 或 `max`（用界面，或 `PATCH /api/agents/<id>/model-settings`）。

**2. 模型有没有挂 effort 映射**

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from mistral_provider import MistralProvider
p = MistralProvider(id='m', name='M', base_url='', api_key='k', models=[])
print(p.thinking_control('deepseek-v4-1-flash-1017').kind)"
```

返回 `effort` 才算挂上；`unknown` 说明没挂。

**3. 是不是被 GLM 的默认行为误导了**

`zai-glm-5-3-flash` **不发 effort 也会返回思考内容**（670 字符）。所以"GLM 有思考块"不能证明 effort 生效。

---

## 症状：模型列表空白

**根因**：`ProviderManager._prepare_plugin_registration` 只在 provider 类实现 `get_default_models()` 时才播种目录：

```python
default_models = []
if hasattr(provider_class, "get_default_models"):
    default_models = provider_class.get_default_models()
```

**处理**：实现该 classmethod（本插件已实现，6 个模型）。

**验证**：`python3 tests/test_config.py` 里的 `get_default_models seeds the catalog`。

---

## 症状：`Unsupported cache control: prompt_cache_key`

**根因**：`Provider.cache_modes` 默认是空 frozenset，`cache_policy.py:68` 见到空集合直接抛错。

**处理**：provider 类声明 `cache_modes = frozenset({"openai"})`。

---

## 症状：`Plugin provider ID 'mistral' conflicts with an existing provider`

**根因**：插件 provider 不能顶替同名供应商，只能报错退出（`provider_manager.py:530`）。

**处理**：先删掉手配的同名 provider，再重启或热装。

**注意**：注册只在启动时跑一次，运行中删掉冲突项**不会自动重试**。删完必须重启或 `qwenpaw plugin install <path> --force`。

---

## 症状：热装报 `Plugin loader is not ready yet`

**根因**：启动过程中插件注册抛异常，`app.state.plugin_loader` 的赋值（在注册循环之后）被跳过，整个插件子系统不可用。

**处理**：修掉注册异常后重启一次。

---

## 症状：`session_id mismatch` 警告，工具调用卡片异常

```
WARNING app/routers/tool_calls.py:91 | session_id mismatch:
  url=1794280522564-1ytua22  backend=console:adhoc-06c612aa4280
```

**根因**：前端传的 session id 少了 `console:` 前缀。这是刻意的安全校验（防止用泄漏的 `tool_call_id` 跨会话操作），**不是本插件的问题**，也不影响对话本身。

---

## 通用排查步骤

1. 看 `/tmp/qwenpaw_query_error_*.json`（最新那份），里面是完整请求与异常
2. 看 `qwenpaw.log` 里 `failed before output` 的行 —— 有 fallback 说明主模型挂了
3. 确认实际用的模型：`Usage for session ...` 日志行里的 `provider_id` / `model_name`
4. 跑 `python3 tests/test_config.py`，确认插件自身配置没问题
