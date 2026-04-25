# FoodCooker - 个性化食谱推荐助手

基于 LangChain + RAG 的智能食谱推荐系统。根据用户的饮食偏好（过敏、菜系、口味）和现有食材，通过向量检索和 LLM 适配，生成个性化食谱并计算营养成分。

## 功能特性

- **个性化推荐**：根据用户食材和偏好推荐菜谱
- **智能适配**：LLM 动态调整食谱（口味、烹饪方式）
- **营养计算**：自动计算卡路里、蛋白质、碳水、脂肪
- **购物清单**：对比食材库存，生成所需购买清单
- **多轮对话**：支持上下文记忆，持续优化推荐
- **会话管理**：Chainlit 内置历史记录，支持恢复对话

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | DashScope (Qwen) / OpenAI |
| Embedding | DashScope TextEmbedding / HuggingFace |
| 向量数据库 | Chroma |
| UI | Chainlit |
| 数据库 | PostgreSQL (Chainlit 历史) |
| 框架 | LangChain 1.2.x |

## 项目结构

```
FoodCooker/
├── src/food_cooker/
│   ├── agent/
│   │   ├── executor.py          # ReAct 执行器
│   │   ├── prompts.py           # Agent 提示词
│   │   └── tools/               # 6 个 Tool 函数
│   │       ├── user_profile_tool.py
│   │       ├── recipe_retriever_tool.py
│   │       ├── recipe_adaptor_tool.py
│   │       ├── nutrition_calculator_tool.py
│   │       ├── shopping_list_tool.py
│   │       └── feedback_tool.py
│   ├── models/                  # Pydantic 数据模型
│   ├── memory/                   # Token 缓冲区
│   ├── ui/
│   │   └── chainlit_app.py      # Chainlit 主应用
│   ├── vectorstore/
│   │   └── chroma_client.py     # Chroma + Embeddings
│   ├── llm.py                   # LLM 工厂 (dashscope/openai)
│   └── settings.py              # 配置管理
├── data/
│   ├── recipes_raw.json         # 食谱数据
│   └── chroma_db/              # 向量数据库
├── docker-compose.yml           # PostgreSQL
├── chainlit.md                 # 欢迎页
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 配置环境

复制 `.env.example` 为 `.env`，填入 API Key：

```bash
cp .env.example .env
```

必需配置：
- `DASHSCOPE_API_KEY` — 阿里云 DashScope API Key
- `DASHSCOPE_MODEL_NAME` — 模型名称（如 `qwen-plus`）

可选：
- `LLM_PROVIDER=openai` 切换到 OpenAI
- `EMBEDDING_PROVIDER=huggingface` 切换 HuggingFace Embedding

### 3. 初始化向量数据库

```bash
python scripts/ingest_recipes.py
```

### 4. 启动 PostgreSQL（Chainlit 历史功能）

```bash
docker-compose up -d
```

### 5. 启动应用

```bash
chainlit run src/food_cooker/ui/chainlit_app.py --port 8000
```

访问 http://localhost:8000，使用 `admin` / `admin123` 登录。

## 使用示例

```
用户：我有鸡蛋和西红柿，我喜欢清淡的
AI：  为你推荐「番茄炒蛋」，根据你的清淡偏好调整了烹饪方式...
用户：热量高吗？
AI：  约 150 kcal，蛋白质 12g...
用户：生成购物清单
AI：  你还需要购买：番茄 2个、葱花少许...
```

## 开发命令

```bash
# 运行所有测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_tools/test_nutrition_calculator_tool.py -v

# 启动 Chainlit（开发模式，支持热重载）
chainlit run src/food_cooker/ui/chainlit_app.py --port 8000 --watch
```

## Agent Tools

| Tool | 功能 |
|------|------|
| `user_profile_tool` | 读取/写入用户偏好（过敏、饮食类型、器材等） |
| `recipe_retriever_tool` | Chroma 向量检索食谱 |
| `recipe_adaptor_tool` | LLM 动态适配食谱到用户约束 |
| `nutrition_calculator_tool` | 查询营养成分数据 |
| `shopping_list_tool` | 对比库存 vs 食谱，生成购物清单 |
| `feedback_tool` | 解析用户反馈，更新偏好 |

## 架构说明

```
用户输入 → [session_id=xxx] → ReAct Loop (max 20 turns)
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
      user_profile_tool       recipe_retriever_tool    nutrition_calculator_tool
              │                       │                       │
              ▼                       ▼                       ▼
      recipe_adaptor_tool      适配结果 ────────────────────┘
              │
              ▼
      shopping_list_tool / 反馈解析
```

- **ReAct Loop**：使用 `llm.bind_tools(ALL_TOOLS)` 实现，而非 `create_agent`
- **Provider 切换**：修改 `.env` 中的 `LLM_PROVIDER` 和 `EMBEDDING_PROVIDER`
- **Token 控制**：`MAX_TOKEN_BUFFER=2000` 限制 context 长度

## 数据文件

| 文件 | 说明 |
|------|------|
| `data/recipes_raw.json` | 原始食谱数据（10 道示例菜谱） |
| `data/chroma_db/` | Chroma 向量数据库（git-ignored） |
| `data/user_profiles.json` | 用户偏好快照（git-ignored） |
| `data/history/` | 会话历史文件（git-ignored） |

## License

MIT
