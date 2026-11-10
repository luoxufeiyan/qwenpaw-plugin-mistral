# 测试

三套测试，**全部离线**：不需要 API key，不联网，不改动 QwenPaw 配置。

```bash
python3 tests/test_config.py        # 15 项：配置、映射、缓存 key
python3 tests/test_transport.py     # 18 项：归一化单元 + SDK 可解析性
python3 tests/test_provider_e2e.py  # 12 项：对假 Mistral 服务器端到端
```

也支持 pytest（`tests/test_suites.py` 会把三套脚本各跑一遍，断言退出码为 0）：

```bash
python3 -m pytest tests/ -q      # 3 passed
```

## 三套脚本

每个文件都能独立运行，打印 `PASS` / `FAIL` 行并以退出码表示结果；`tests/test_suites.py` 只是把它们包成 pytest 用例。

## test_config.py

验证 provider 类的静态行为，不构造真实请求。

| 检查 | 说明 |
|---|---|
| `cache_modes` 含 `openai` | 否则 `prompt_cache_key` 会被拒 |
| `prompt_cache_key` 过 `cache_policy` | 直接调 `cache_request` 验证 |
| `relay_reasoning` 为 True | 插件依赖回传 + 传输层翻译 |
| 目录内每个模型 `relay_reasoning=True` | 防止漏配 |
| chat 请求带 key | 且同会话稳定、跨会话不同 |
| 非 chat 协议不加 key | `responses` 协议不该被污染 |
| `get_default_models()` 返回 6 个模型 | 模型列表就靠它 |
| effort 只挂在有依据的模型上 | DeepSeek/GLM/Large4/Small4 挂，Codestral 不挂 |
| `thinking_level=high` → `reasoning_effort=high` | 走真实的 `get_agent_thinking_kwargs` |
| reasoning 参数不被拦截 | v0.3 起允许透传，由传输层翻译 |
| model 实例的 formatter relay 为 True | 端到端一致性 |

## test_transport.py

纯函数级测试 + SDK 兼容性验证。

**响应方向**

- chunk 列表 → `content` 字符串 + `reasoning_content`
- 文本内容完整保留
- `content-length` 正确重算
- **改写结果能被 `openai.types.chat.ChatCompletion.model_validate` 接受**

**请求方向**

- 工具调用轮次（`content: null` + `tool_calls`）正确转换
- `content` 的 `str` / `list` / `None` 三种形状全覆盖
- 非 assistant 角色上的游离 `reasoning_content` 被清除
- 无法转换时字段被丢弃而非透传

**SSE**

- 逐帧改写，thinking 与 text 正确累加
- `[DONE]` 与非 `data:` 行原样保留
- **改写后的每个 chunk 能被 `ChatCompletionChunk.model_validate` 接受**

## test_provider_e2e.py

最有说服力的一套：起一个本地 HTTP 服务，**行为与真 Mistral 一致**。

假服务器的行为：

| 请求特征 | 响应 |
|---|---|
| 消息里出现 `reasoning_content` | **422** `extra_forbidden`（与真机同款报错） |
| 带 `reasoning_effort` | chunk 列表 |
| 不带 | 纯字符串 |
| `stream: true` | SSE 逐帧 |

然后驱动真实的调用栈：

```
MistralProvider
  → OpenAIChatModelCompat
  → openai SDK（pydantic 校验）
  → httpx
  → MistralNormalizingTransport
  → 假服务器
```

断言：

- `reasoning_effort` 真的到了 wire
- 非流式：答案文本 + thinking block 都正确
- 流式：同上
- 多轮历史回灌：请求被接受（无 422）、思考内容保留在 `content` 里
- 工具调用轮次：请求被接受、无 `reasoning_content`

## tools/probe_deepseek_effort.py

**唯一需要真实 key 的脚本**，用来验证 effort 档位是否真的生效。

```bash
python3 tools/probe_deepseek_effort.py
```

它从 `SECRET_DIR/providers/plugin/mistral.json` 读取已保存的 key（解密后仅用于发请求，**不打印**），对每个模型 × 每个档位发一次最小请求，只输出：

- HTTP 状态码
- `content` 的**结构**（`str(409 chars)` / `list[thinking(161 chars), text(126 chars)]`）
- `completion_tokens`

输出示例：

```
=== deepseek-v4-1-flash-1017 ===
  {"effort": "(none)", "status": 200, "content": "str(409 chars)", "completion_tokens": 154}
  {"effort": "low",    "status": 200, "content": "list[thinking(161 chars), text(126 chars)]"}
  {"effort": "high",   "status": 200, "content": "list[thinking(216 chars), text(126 chars)]"}
  {"effort": "max",    "status": 200, "content": "list[thinking(342 chars), text(127 chars)]"}
```

这是"文档说支持 low/high/max，实测确认"的证据来源。

## 回归清单

改动 provider 或 transport 后，至少跑：

```bash
python3 tests/test_config.py && python3 tests/test_transport.py && python3 tests/test_provider_e2e.py
```

三套全绿再热装：

```bash
qwenpaw plugin install . --force
```
