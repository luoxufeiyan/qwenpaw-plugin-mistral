# Mistral AI provider plugin for QwenPaw

让 Mistral 的 OpenAI 兼容端点在 QwenPaw 里**完整支持 reasoning**，同时打开 prompt 缓存。

## 为什么需要它

Mistral 的 reasoning 表达方式和 OpenAI 完全不同：

| | OpenAI / DeepSeek 风格 | Mistral |
|---|---|---|
| 思考内容放哪 | 顶层兄弟字段 `reasoning_content` | **`content` 里的 chunk 列表** |
| 形状 | `"content": "答案"` | `[{"type":"thinking","thinking":[{"type":"text","text":"…"}],"closed":true},{"type":"text","text":"答案"}]` |
| 多轮回灌 | 回传 `reasoning_content` 字符串 | 把整条 assistant 消息（含 ThinkChunk）原样回灌 |
| 请求严格度 | 宽松 | **`extra="forbid"`，多一个字段就 422** |

而 openai SDK 2.33 把 `content` 断言为 `str`：

```
ChatCompletion:       ValidationError  choices.0.message.content
                      Input should be a valid string, input_type=list
ChatCompletionChunk:  ValidationError  choices.0.delta.content
                      Input should be a valid string, input_type=list
```

也就是说：**带上 `reasoning_effort` 就直接失败**，不是丢思考内容，是请求整个挂掉。这个插件在 httpx 传输层做双向翻译，把两种格式接起来。

## 做了什么

### 1. 传输层双向归一化（`mistral_transport.py`）

```
响应方向：content 的 chunk 列表  ->  content 字符串 + reasoning_content
请求方向：reasoning_content      ->  content 的 chunk 列表
```

- 非流式：改写 JSON body
- 流式：逐帧改写 SSE（`data:` 行），保留 `[DONE]` 与非 data 行
- 未知 chunk 类型丢弃并 debug 日志，不会把 `{'type':...}` 漏进答案

### 2. 保住 ThinkingBlock（`build_formatter`）

AgentScope 的 `OpenAIChatFormatter` 会**静默丢弃** ThinkingBlock（源码注释："OpenAI API does not accept reasoning/thinking content in conversation history"）。Mistral 恰恰要把它放进 `content`，所以 formatter 必须让它活下来——插件在 format 之后按 assistant 消息顺序把思考内容回填成 `reasoning_content`，交给传输层翻译。

### 3. `reasoning_effort` 映射

> **前提：agent 的 thinking level 不能是 `inherit`。**
> `provider.py:957` 里 `level == "inherit"` 会直接 return，什么参数都不发。
> 而 DeepSeek 默认**不思考**（探针实测：不发 effort 时 content 是纯字符串），
> 所以档位必须是 `low` / `high` / `max` 才看得到思考块。
>
> 注意 GLM 5.3 Flash 默认就思考（不发 effort 也返回 thinking），
> 容易让人误以为"effort 已经生效了"——其实没有。


只给 Mistral 文档明确支持的模型挂 `ThinkingControl`：

| 模型 | effort |
|---|---|
| `zai-glm-5-3-flash` / `zai-glm-5-3` | low / high / max |
| `mistral-large-4-0` | low / medium / high |
| `mistral-small-2603` | low / medium / high（+ off） |
| `deepseek-v4-1-flash-1017` | low / high / max（**实测确认**，见下） |
| `codestral-25-08` | 不挂（未探测） |

### 实测：DeepSeek 也支持 reasoning_effort

Mistral 文档的支持列表里没有 DeepSeek，但 2026-11-10 用真实 key 探测（`probe_deepseek_effort.py`）的结果是它能用，而且**思考深度确实随档位变化**：

| effort | 状态 | content 形状 | thinking 长度 | completion tokens |
|---|---|---|---|---|
| （不发） | 200 | `str` | 无 | 154 |
| low | 200 | `list[thinking, text]` | 161 字符 | 100 |
| high | 200 | `list[thinking, text]` | 216 字符 | 101 |
| max | 200 | `list[thinking, text]` | **342 字符** | 158 |

对照组 `zai-glm-5-3-flash` 同样能跑，但它**不发 effort 时也会返回 thinking**（670 字符），说明这两个模型的默认档行为不同：DeepSeek 默认不思考，GLM 默认就思考。

结论：**文档不全，实测为准**。Codestral 没测，所以仍然不挂。

### 4. prompt 缓存

`cache_modes = {"openai"}` 让 `prompt_cache_key` 过得了校验，并由会话前缀（system/developer + 首条 user）派生稳定 key。Mistral 缓存输入按 **10% 计价**：`deepseek-v4-1-flash-1017` 输入 `$0.30/M`、缓存命中 `$0.007/M`。

### 5. 模型目录

实现 `get_default_models()` 钩子（`ProviderManager._prepare_plugin_registration` 只认这个 classmethod），否则注册出来的 provider 模型列表是空的。

## 安装

```bash
# 从 GitHub 仓库安装
qwenpaw plugin install https://github.com/luoxufeiyan/qwenpaw-plugin-mistral

# 或从本地目录
qwenpaw plugin install /path/to/qwenpaw-plugin-mistral --force
```

热加载，无需重启。日志应为：

```
Unloaded plugin 'mistral'
Unregistered plugin provider 'mistral' from memory
Registered provider 'mistral' from plugin 'mistral'
✓ Loaded plugin 'mistral' successfully
```

api_key 会从 `SECRET_DIR/providers/plugin/mistral.json` 自动恢复。

> 热装会**整目录覆盖**目标路径，源里没带的文件会消失。重新安装时把所有文件一起带上。

## 测试

三套，全部离线（不需要真 key、不联网）：

```bash
python3 selftest.py           # 14/14  配置与映射
python3 test_transport.py     # 15/15  归一化单元 + openai SDK 可解析性
python3 test_provider_e2e.py  # 10/10  对假 Mistral 服务器的端到端
```

`test_provider_e2e.py` 起一个本地 HTTP 服务，行为与真 Mistral 一致：

- 请求里出现 `reasoning_content` → 返回 422（和真机同款报错）
- 带 `reasoning_effort` → 返回 chunk 列表
- 不带 → 返回普通字符串
- 支持流式与非流式

然后驱动真实的 `MistralProvider → OpenAIChatModelCompat → openai SDK → httpx → 归一化传输层`，断言：

```
PASS  reasoning_effort reaches the wire
PASS  non-stream: answer text / thinking block
PASS  stream: answer text / thinking block
PASS  history: request accepted (no 422)
PASS  history: reasoning_content translated away
PASS  history: thinking preserved in content
```

## 已知限制

- **依赖 openai SDK 内部结构**：通过 `model.client._client._transport` 换传输层。SDK 大版本升级若改了私有属性名，需要跟着改（有 warning 兜底，不会静默失败）。
- **`content` 是列表**：Mistral 返回 chunk 列表时，请求侧同样要用 chunk 列表回灌。若某天 Mistral 改为只收字符串，请求方向也要跟着关掉。
- **价格表留空**：`pricing` 字典的键名语义没有文档，写错比不写糟，靠官方价目表人工核对。
- **未做**：`ThinkChunk` 的 `closed` / 签名类字段透传（Mistral 目前不需要）。

## 相关代码位置（QwenPaw 2.2.2 / agentscope 2.0）

| 位置 | 作用 |
|---|---|
| `providers/provider.py:344` | `cache_modes` 默认空集合 |
| `providers/provider.py:1236` | `_get_relay_reasoning` 默认 `True` |
| `providers/adapters/cache_policy.py:68` | `prompt_cache_key` 放行条件 |
| `providers/provider_manager_persistence.py:534` | `get_default_models()` 调用点 |
| `agents/model_factory.py:1813-1816` | 是否注水 `reasoning_content` |
| `agentscope/formatter/_openai_formatter.py:391` | **静默丢弃 ThinkingBlock** |
| `agentscope/model/_openai_chat/_model.py:493` | `TextBlock(text=choice.message.content)`，list 会炸 |
