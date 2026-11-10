# 架构

这个插件只做一件事：**让 Mistral 的 OpenAI 兼容端点在 QwenPaw 里表现得像 QwenPaw 认识的供应商**。所有改动都落在 provider 层的公开钩子上，不修改 QwenPaw 或 AgentScope 的源码。

## 数据流

```
                    ┌──────────────── QwenPaw agent ────────────────┐
                    │                                               │
  用户消息 ────────▶ │  Msg(ThinkingBlock + TextBlock + ToolCall)     │
                    │                    │                          │
                    │                    ▼                          │
                    │  MistralFormatter.format()   ← 保住 ThinkingBlock
                    │                    │                          │
                    │                    ▼                          │
                    │  wire: {content, reasoning_content}            │
                    └────────────────────┬──────────────────────────┘
                                         │
                                         ▼
                    ┌──── MistralNormalizingTransport ─────┐
                    │  请求：reasoning_content              │
                    │        → content: [ThinkChunk, Text]  │
                    └──────────────────┬───────────────────┘
                                       │  HTTPS
                                       ▼
                              api.mistral.ai/v1
                                       │
                    ┌──────────────────▼───────────────────┐
                    │  响应：content: [ThinkChunk, Text]    │
                    │        → content: str                 │
                    │        + reasoning_content: str       │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────── openai SDK ──────────┐
                    │  pydantic 校验 content: str  ← 关键   │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    AgentScope 解析 → ThinkingBlock + TextBlock
```

关键点：**翻译必须发生在 SDK 之前**。openai SDK 把 `content` 断言为 `str`，chunk 列表在 pydantic 校验阶段就会被拒；formatter 层根本来不及处理。

## 钩子清单

| 钩子 | 位置 | 作用 |
|---|---|---|
| `cache_modes` | 类属性 | 声明 `{"openai"}`，否则 `prompt_cache_key` 被 `cache_policy` 拒绝 |
| `get_default_models()` | classmethod | 注册时播种模型目录（`_prepare_plugin_registration` 只认这个钩子） |
| `_get_relay_reasoning()` | 覆盖 | 返回 `True`，允许 QwenPaw 把思考内容回传到 wire |
| `prepare_request()` | 覆盖 | 派生并注入 `prompt_cache_key` |
| `get_chat_model_instance()` | 覆盖 | 装 formatter + 换 httpx 传输层 |
| `build_formatter()` | 新增 | 保住被 AgentScope 丢弃的 `ThinkingBlock` |

## 为什么需要自定义 formatter

AgentScope 的 `OpenAIChatFormatter` 在拼 assistant 消息时会**静默丢弃** `ThinkingBlock`：

```python
# agentscope/formatter/_openai_formatter.py:391
elif isinstance(block, ThinkingBlock):
    # OpenAI API does not accept reasoning/thinking content
    # in conversation history — skip thinking blocks silently.
    pass
```

QwenPaw 的注水逻辑依赖 formatter 输出里还留着思考内容，但这里已经丢干净了 —— 所以 `reasoning_content` 永远注不进去。Mistral 恰恰要求把思考内容放在 `content` 里回灌，因此必须让这个 block 活下来。

`MistralFormatter` 的做法是在 `super().format()` 之后，按 assistant 消息顺序把思考内容回填成 `reasoning_content`，交给传输层翻译。其余行为（媒体提升、file block、tool call）完全沿用 QwenPaw 自己的 formatter 包装。

## 为什么动传输层

`OpenAIChatModelCompat` 已经有一层 `_SanitizedStream` 处理畸形 tool call，但那是**校验之后**的对象级修补，救不了 pydantic 阶段就失败的 chunk 列表。

所以插件在 httpx 层替换 `model.client._client._transport`：

- 非流式：改写 JSON body（含 `content-length` 重算）
- 流式：逐帧改写 SSE 的 `data:` 行，保留 `[DONE]` 与非 data 行

## 代码地图

| 文件 | 内容 |
|---|---|
| `plugin.py` | 插件入口，调 `api.register_provider(...)` |
| `mistral_provider.py` | `MistralProvider`：钩子实现、模型目录、effort 映射 |
| `mistral_transport.py` | `MistralNormalizingTransport` + 纯函数转换（可单测） |
| `tests/` | 三套离线测试 |
| `tools/probe_deepseek_effort.py` | 真实 API 探测脚本 |

## 依赖的私有接口

| 依赖 | 风险 | 兜底 |
|---|---|---|
| `model.client._client._transport` | openai SDK 大版本改名会失效 | 有 warning 日志，不会静默失败 |
| `OpenAIChatModelCompat.formatter` | 属性名变化 | `build_formatter` 失败时回退到父类 formatter 并告警 |
| AgentScope `ThinkingBlock` 类型 | 类型路径变化 | import 失败会抛错，测试可捕获 |
