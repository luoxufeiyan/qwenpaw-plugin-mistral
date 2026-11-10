# Mistral 与 OpenAI 的线格式差异

这份文档记录实测得到的差异，以及插件如何双向翻译。所有数据来自 2026-11-10 对 `api.mistral.ai/v1` 的真实请求。

## 1. 思考内容的位置

| | OpenAI / DeepSeek 风格 | Mistral |
|---|---|---|
| 思考内容 | 顶层兄弟字段 `reasoning_content` | **`content` 里的 chunk 列表** |
| 纯文本响应 | `"content": "答案"` | `"content": "答案"`（无思考时） |
| 带思考响应 | `"content": "答案"` + `"reasoning_content": "…"` | `"content": [{thinking}, {text}]` |

Mistral 的实际返回（`zai-glm-5-3-flash`，`reasoning_effort: high`）：

```json
{
  "content": [
    {"type": "thinking",
     "thinking": [{"type": "text", "text": "Simple: 17×23 = 391."}],
     "closed": true},
    {"type": "text", "text": "**17 × 23 = 391**\n\n计算过程：…"}
  ]
}
```

## 2. 请求严格度

Mistral 用 pydantic 的 `extra="forbid"` 校验请求体。多一个字段就 422：

```json
{"detail": [{"type": "extra_forbidden",
             "loc": ["body", "messages", 2, "assistant", "reasoning_content"],
             "msg": "Extra inputs are not permitted"}]}
```

DeepSeek 官方 API 对多余字段宽容得多，这也是 QwenPaw 原来那套逻辑能跑通的原因。

## 3. openai SDK 的类型断言

openai SDK 2.33.0 把 `content` 声明为 `str`，两种形状都拒：

```
ChatCompletion:       ValidationError  choices.0.message.content
                      Input should be a valid string, input_type=list
ChatCompletionChunk:  ValidationError  choices.0.delta.content
                      Input should be a valid string, input_type=list
```

**这是解析阶段，早于 QwenPaw 的任何代码。** 所以必须在传输层改写，不能靠 formatter 或响应后处理。

## 4. 双向翻译规则

### 响应方向（`normalize_completion` / `normalize_chunk`）

```
content: [ThinkChunk, TextChunk]
    ↓
content: "答案文本"
reasoning_content: "思考文本"
```

- 未知 chunk 类型丢弃并打 debug 日志，不会把 `{'type': …}` 漏进答案
- `reasoning_content` 已存在时追加而非覆盖

### 请求方向（`normalize_request_body`）

```
assistant: {content: "答案", reasoning_content: "思考"}
    ↓
assistant: {content: [ThinkChunk, TextChunk]}
```

必须处理的三种 `content` 形状：

| 形状 | 场景 | 处理 |
|---|---|---|
| `str` | 普通文本轮次 | 转成 `[thinking, text]` |
| `list[text parts]` | QwenPaw formatter 的输出 | 先拍平成字符串再转 |
| `null` | **只发工具调用、没有文本的轮次** | 只发 `[thinking]`，不带 text chunk |

第三种是最容易漏的：agent 每次调工具都会产生 `content: null + tool_calls`，漏了就是"一用工具就 422"。

无法转换的 `reasoning_content` **一律丢弃**：丢一段思考只是质量下降，留着就是整个请求挂掉。非 assistant 角色上的残留同样清除。

## 5. reasoning_effort

Mistral 用顶层 `reasoning_effort` 控制思考量。实测（`tools/probe_deepseek_effort.py`）：

| 模型 | 不发 | low | high | max |
|---|---|---|---|---|
| `deepseek-v4-1-flash-1017` | 无思考（`content` 是 str） | 161 字符 | 216 字符 | 342 字符 |
| `zai-glm-5-3-flash` | **670 字符**（默认就思考） | 无思考（str） | 22 字符 | 948 字符 |

两个反直觉的点：

1. **GLM 默认就思考**，不发 effort 也有思考内容
2. **GLM 的 `low` 等于关思考**，返回纯字符串

所以"看到思考块"不等于"effort 生效了"——必须看 `reasoning_effort` 是否真的发出。

Mistral 文档的支持列表里没有 DeepSeek，但实测四档全部 200 —— **文档不全，实测为准**。

## 6. Prompt 缓存

| 项 | 值 |
|---|---|
| 缓存输入单价 | 输入价的 **10%**（`deepseek-v4-1-flash-1017`: $0.007/M vs $0.30/M） |
| 命中粒度 | **64 token 块** |
| 启用方式 | 请求带 `prompt_cache_key`（同一会话固定值） |
| 验证 | 响应 `usage.prompt_tokens_details.cached_tokens` |

实测命中率（真实 agent 会话）：

| 模型 | prompt tokens | cache_read | 命中率 |
|---|---|---|---|
| `zai-glm-5-3-flash` | 111,226 | 38,016 | 34% |
| `deepseek-v4-1-flash-1017` | 161,372 | 86,784 | **54%** |

插件的 key 由会话前缀（system/developer 消息 + 首条 user 消息）的 SHA-256 派生，同一会话恒定、不同会话不同。
