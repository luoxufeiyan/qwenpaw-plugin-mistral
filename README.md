# qwenpaw-plugin-mistral

让 [Mistral AI](https://mistral.ai) 的 OpenAI 兼容端点在 [QwenPaw](https://github.com/agentscope-ai/QwenPaw) 里**完整支持 reasoning**，并启用 prompt 缓存。

Mistral 把思考内容放在 `content` 的 chunk 列表里，而 openai SDK 把 `content` 断言为字符串 —— 两者不兼容，结果是**带思考的请求直接失败**。这个插件在传输层做双向翻译，把两种格式接起来。

## 能力

| 能力 | 说明 |
|---|---|
| **Reasoning** | 双向翻译 `reasoning_content` ↔ `ThinkChunk`，含 SSE 流式逐帧改写 |
| **ThinkingBlock 保活** | AgentScope 的 OpenAI formatter 会静默丢弃思考块，插件在 format 后回填 |
| **思考档位** | agent 的 thinking level 映射到 `reasoning_effort`（low / high / max） |
| **Prompt 缓存** | 自动派生稳定 `prompt_cache_key`，缓存输入按 10% 计价 |
| **模型目录** | 内置 6 个模型，注册即可见 |

## 安装

```bash
qwenpaw plugin install https://github.com/luoxufeiyan/qwenpaw-plugin-mistral
```

或从本地目录：

```bash
git clone https://github.com/luoxufeiyan/qwenpaw-plugin-mistral
qwenpaw plugin install ./qwenpaw-plugin-mistral --force
```

安装后热加载，无需重启。在「模型 / 供应商」里会多出 **Mistral AI**，base URL 已预填 `https://api.mistral.ai/v1`，填入 API key 即可。

> 若已有同名 provider，先删掉它 —— 插件 provider 不能顶替同名供应商。

## 使用

1. 选模型：`deepseek-v4-1-flash-1017` 或 `zai-glm-5-3-flash`
2. **把 agent 的 thinking level 设为 `high` 或 `max`**
3. 问一个需要推理的问题，界面上会出现思考块

第 2 步不能省：`inherit` 表示"不指定"，一个参数都不发，而 DeepSeek 默认**不思考**。

## 验证

```bash
python3 tests/test_config.py        # 15 项
python3 tests/test_transport.py     # 18 项
python3 tests/test_provider_e2e.py  # 12 项
```

全部离线，不需要 API key。`test_provider_e2e.py` 会起一个行为与真 Mistral 一致的假服务器（带 `reasoning_content` 就回 422），驱动真实的 provider → SDK → transport 全链路。

## 目录结构

```
.
├── plugin.json              # 插件清单
├── plugin.py                # 入口：注册 provider
├── mistral_provider.py      # Provider 钩子：缓存、effort、formatter、transport 装配
├── mistral_transport.py     # httpx 双向归一化（JSON + SSE）
├── tests/                   # 三套离线测试
├── tools/                   # 真实 API 探测脚本
└── docs/
    ├── architecture.md      # 数据流、钩子清单、代码地图
    ├── mistral-wire-format.md  # 线格式差异与实测数据
    ├── troubleshooting.md   # 症状 → 根因 → 处理
    └── testing.md           # 三套测试分别覆盖什么
```

## 兼容性

- QwenPaw **≥ 2.2.2**
- Python 3.11+
- 依赖 openai SDK 私有属性 `model.client._client._transport`；SDK 大版本升级若改名需跟进（有 warning 兜底，不会静默失败）

## 已知限制

- `pricing` 字段留空 —— 避免写错计费
- Codestral 未做 effort 探测，故未挂思考档位
- 未处理 `ThinkChunk` 的 `closed` / 签名类字段（Mistral 目前不需要）

## 许可

MIT
