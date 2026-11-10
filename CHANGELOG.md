# Changelog

## 1.0.0 — 2026-11-10

首个发布版本。让 Mistral 的 OpenAI 兼容端点在 QwenPaw 里完整支持 reasoning，并启用 prompt 缓存。

### 修复

- **422 `extra_forbidden` on `reasoning_content`**：Mistral 用严格 schema（`extra="forbid"`），QwenPaw 注入的 DeepSeek 风格兄弟字段会被拒。改为在传输层翻译成 Mistral 的 chunk 列表。
- **模型列表空白**：`ProviderManager._prepare_plugin_registration` 只认 `get_default_models()` classmethod，未实现时 `models=[]`。已实现，内置 6 个模型。
- **`prompt_cache_key` 被拒**：`Provider.cache_modes` 默认空集合导致 `Unsupported cache control`。已声明 `{"openai"}` 并由会话前缀派生稳定 key。
- **工具调用轮次 422**：`content: null` + `tool_calls` 的 assistant 消息无法转换，`reasoning_content` 原样上线。已支持该形状，并对不可转换的残留做兜底丢弃。

### 新增

- 传输层双向归一化（`mistral_transport.py`）：非流式 JSON + 流式 SSE 逐帧改写。
- `build_formatter()`：保住被 AgentScope 静默丢弃的 `ThinkingBlock`（`_openai_formatter.py:391`）。
- `reasoning_effort` 映射：DeepSeek V4.1 Flash、GLM 5.3 / 5.3 Flash、Mistral Large 4、Mistral Small 4。
- `probe_deepseek_effort.py`：探测脚本，验证 effort 档位真实生效（不打印 key）。

### 实测数据

| 模型 | effort | thinking 长度 | 说明 |
|---|---|---|---|
| deepseek-v4-1-flash-1017 | 不发 | 无 | 默认不思考 |
| | low / high / max | 161 / 216 / 342 字符 | 逐档变长 |
| zai-glm-5-3-flash | 不发 | 670 字符 | 默认就思考 |
| | low | 无（返回纯字符串） | low 等于关思考 |

### 已知限制

- 依赖 openai SDK 私有属性 `model.client._client._transport`，SDK 大版本升级若改名需跟进（有 warning 兜底）。
- `pricing` 字段留空，避免写错计费。
- Codestral 未探测，未挂 effort 映射。
