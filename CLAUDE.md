# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FoodCooker** — A LangGraph-based personalized recipe recommendation agent using Supervisor-Worker multi-agent architecture. Given a user's dietary constraints and available ingredients, it retrieves recipes via RAG (hybrid BM25+vector search), adapts them with an LLM, calculates nutrition, and generates shopping lists. Multi-turn conversation with long-term memory is served through two UIs: a FastAPI REST API with SSE streaming, and a Chainlit chat UI.

## Commands

```bash
# ── Backend ──
pip install -e .

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_tools/test_nutrition_calculator_tool.py -v

# Ingest recipes into Chroma vector store (run once after setup)
python scripts/ingest_recipes.py

# Run RAG retrieval evaluation (hit@K, MRR, precision@K)
python scripts/evaluate_retriever.py

# Start the FastAPI backend (production)
uvicorn food_cooker.api.app:create_app --factory --port 8000

# Start the Chainlit chat UI
chainlit run src/food_cooker/ui/chainlit_app.py --port 8000

# Start infrastructure (Postgres + Redis)
docker-compose up -d
```

## Architecture

```
User (Chainlit UI / FastAPI)
        │  streaming via astream_events v2 / SSE
        ▼
Supervisor-Worker LangGraph (supervisor.py)
        │
        ├── Supervisor LLM — routes to 4 domain workers via transfer_to_* tools
        │
        ├── recipe_worker   ←→ recipe_tools (recipe_retriever, recipe_adaptor, image_generation)
        ├── nutrition_worker ←→ nutrition_tools (nutrition_calculator)
        ├── shopping_worker  ←→ shopping_tools (shopping_list)
        └── general_worker   ←→ general_tools (user_profile, feedback, vision)
                │
                ▼
        RAG: BM25 + Chroma hybrid → RRF fusion
        Cache: Redis (graceful degradation)
```

- **Backend**: FastAPI with CORS, JWT auth (bcrypt + python-jose), SQLite async auth DB, SSE streaming via `StreamingResponse`.
- **Chainlit UI**: Legacy chat interface built with Chainlit. Compiles agent with `MemorySaver`, handles streaming loop and session lifecycle.
- **Agent graph**: `build_supervisor_graph()` in [supervisor.py](src/food_cooker/agent/supervisor.py) returns a `StateGraph(MessagesState)`. Compiled with `MemorySaver` for per-session checkpointing.
- **Streaming**: Uses `agent.astream_events(version="v2")` to capture `on_chat_model_stream` events. SSE format: `data: {"type": "token", "content": "..."}\n\n`.
- **Hybrid search**: BM25 (rank_bm25) + Chroma vector search → RRF fusion (k=60) → optional cross-encoder reranking.
- **Long-term memory**: After each turn, an LLM generates a 2-4 sentence summary of user facts, persisted via `user_profile_tool` under the key `__global_user__`. Injected into the system prompt on subsequent conversations. Capped at 2000 chars.

## Provider Configuration

All settings in `.env`:

| Setting | Values | Default |
| ------ | ------ | ------ |
| `LLM_PROVIDER` | `dashscope`, `openai` | `dashscope` |
| `EMBEDDING_PROVIDER` | `dashscope`, `huggingface` | `dashscope` |

Both LLM providers go through `ChatOpenAI` — DashScope uses `base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`.

## Key Files

| File | Role |
| ------ | ------ |
| [chainlit_app.py](src/food_cooker/ui/chainlit_app.py) | Chainlit UI entry point. Compiles agent with `MemorySaver`, handling streaming loop, message conversion, memory injection, and session lifecycle. |
| [supervisor.py](src/food_cooker/agent/supervisor.py) | Builds the Supervisor-Worker LangGraph. Supervisor routes via 4 transfer tools; each worker is an LLM + ToolNode with domain-specific tools. |
| [llm.py](src/food_cooker/llm.py) | Single factory `get_llm(temperature)` — `@lru_cache(maxsize=8)` by temperature. Returns `ChatOpenAI`. |
| [hybrid_retriever.py](src/food_cooker/vectorstore/hybrid_retriever.py) | BM25 + Chroma hybrid search with RRF fusion (k=60) and optional cross-encoder reranking. |
| [chroma_client.py](src/food_cooker/vectorstore/chroma_client.py) | `get_chroma_client()` returns a Chroma wrapper. Contains `DashScopeEmbeddings`. |
| [settings.py](src/food_cooker/settings.py) | Pydantic-settings with `.env` loading. JWT, Redis, DB, and API keys configured here. |
| [cache.py](src/food_cooker/cache.py) | Redis cache with connection pool and `@cached(prefix, ttl)` decorator. Graceful degradation when Redis unavailable. |

## Eight Agent Tools

All tools are `@tool`-decorated functions in [src/food_cooker/agent/tools/](src/food_cooker/agent/tools/):

| Tool | Worker | Purpose |
| ------ | ------ | ------ |
| `recipe_retriever_tool` | recipe | BM25 + Chroma hybrid search with `tags_filter` (list) and `cuisine_filter` (str) |
| `recipe_adaptor_tool` | recipe | Calls LLM via `JsonOutputParser` to adapt recipes to user constraints |
| `image_generation_tool` | recipe | DALL-E 3 food image generation for recipes (1024x1024) |
| `nutrition_calculator_tool` | nutrition | Lookup nutrition data from `data/nutrition_db.json` (external JSON, 50+ ingredients) |
| `shopping_list_tool` | shopping | Diff recipe ingredients vs user inventory, categorize, generate buy list |
| `user_profile_tool` | general | Read/update user profiles in `data/user_profiles.json` with `FileLock` concurrency protection |
| `feedback_tool` | general | Parse user feedback and update profile preferences |
| `vision_identify_ingredients_tool` | general | GPT-4V image analysis to identify ingredients from food photos |

## Key Implementation Notes

- **LangGraph, not ReAct**: The old `llm.bind_tools(ALL_TOOLS)` direct-call approach was replaced by a Supervisor-Worker `StateGraph`. `build_supervisor_graph()` returns the uncompiled graph; `build_agent()` compiles it with `MemorySaver`.
- **MemorySaver, not PostgresSaver**: `astream_events` creates an `AsyncPregelLoop` internally, which requires `checkpointer.aget_tuple()`. `PostgresSaver` with sync `Connection` doesn't implement this — use `MemorySaver` instead. Chainlit's `SQLAlchemyDataLayer` handles Postgres chat history separately.
- **`recipe_retriever_tool`**: `tags_filter` must be a Python list, not a JSON string. `cuisine_filter` is a plain string.
- **LangChain version**: 1.2.x — APIs differ significantly from 0.x. `create_agent` doesn't properly handle Chinese text, hence the explicit graph approach.
- **Message format**: Chainlit stores dict messages; LangGraph uses LangChain message objects. `_convert_to_langchain_messages()` and `_convert_to_dict_messages()` handle conversion, including ToolCall reconstruction for resume scenarios.
- **Session isolation**: Each Chainlit session gets a UUID-based `session_id`, passed as `config={"configurable": {"thread_id": session_id}}` to the agent. The human input text also carries `[session_id=xxx]` prefix for tool visibility.

## Data Files

- `data/recipes_raw.json` — 10 sample recipes (source for Chroma ingestion)
- `data/nutrition_db.json` — Nutrition data for 50+ common ingredients
- `data/eval_ground_truth.json` — 12 QA pairs for RAG evaluation
- `data/categories.json` — Ingredient category mapping used by shopping_list_tool
- `data/chroma_db/` — Persistent Chroma vector store (git-ignored)
- `data/user_profiles.json` — Per-session user profiles + conversation memory (git-ignored)

## Tests

18 tests across 6 files in `tests/test_tools/`, testing each tool independently. `pytest-asyncio` in auto mode.
