# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FoodCooker** — A LangChain-based personalized recipe generation agent. Given a user's dietary constraints (allergies, diet type, equipment, cuisine preference), it retrieves recipes from a RAG vector store, adapts them using an LLM, calculates nutrition, and generates shopping lists. Supports multi-turn conversation.

## Commands

```bash
# Install dependencies
pip install -e .

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_tools/test_nutrition_calculator_tool.py -v

# Ingest recipes into Chroma vector store (run once after setup)
python scripts/ingest_recipes.py

# Start the Chainlit chat UI
chainlit run src/food_cooker/ui/chainlit_app.py --port 8000
```

## Architecture

```
User (Chainlit UI)
        │
        ▼
ReAct Loop (llm.bind_tools) in chainlit_app.py
   ├── LLM: get_llm() — dual-provider (dashscope / openai)
   └── Tools (6 @tool-decorated functions):
         ├── user_profile_tool     → reads/writes user prefs to JSON
         ├── recipe_retriever_tool  → Chroma similarity_search (RAG)
         ├── recipe_adaptor_tool    → LLM chain adapts recipe to constraints
         ├── nutrition_calculator_tool → hardcoded nutrition DB lookup
         ├── shopping_list_tool     → diff recipe ingredients vs user inventory
         └── feedback_tool          → parses feedback → updates profile

RAG Layer: Chroma (langchain_chroma) + DashScopeEmbeddings (custom Embeddings class)
LLM: All LLM calls routed through get_llm() in llm.py — swap provider via .env
```

## Provider Configuration

All settings are in `.env` (see `.env.example`):

| Setting | Values | Default |
|---------|--------|---------|
| `LLM_PROVIDER` | `dashscope`, `openai` | `dashscope` |
| `EMBEDDING_PROVIDER` | `dashscope`, `huggingface` | `dashscope` |

Switching providers only requires changing `.env` — no code changes needed.

## Key Implementation Notes

- **`chainlit_app.py`** — Implements a manual ReAct loop using `llm.bind_tools(ALL_TOOLS)`. Session ID is auto-generated per Chainlit session and embedded as `[session_id=xxx]` prefix in the input text. No `create_agent` needed.
- **`llm.py`** — Single factory `get_llm()` for all LLM instances. DashScope uses OpenAI-compatible API (`base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`).
- **`DashScopeEmbeddings`** — Custom `Embeddings` subclass in `chroma_client.py` that calls `dashscope.TextEmbedding.call()` directly. Raises `RuntimeError` on API failure.
- **`recipe_adaptor_tool`** — Internally calls LLM again to produce structured JSON output via `JsonOutputParser`. This is intentional: it keeps adaptation logic isolated and testable.
- **`recipe_retriever_tool`** — Chroma similarity search with optional `tags_filter` (list) and `cuisine_filter` (str). `tags_filter` must be a Python list, not a JSON string.
- **LangChain version**: 1.2.15 — APIs differ significantly from 0.x. Do not assume other LangChain examples will work directly. In particular, `create_agent` from LangChain 1.2.x does not properly merge `input` dict field into messages for Chinese text — use the `bind_tools` approach instead.

## Data Files

- `data/recipes_raw.json` — Source recipe data (10 sample recipes)
- `data/chroma_db/` — Persistent Chroma vector store (git-ignored)
- `data/user_profiles.json` — Per-session user preference snapshots (git-ignored)

## Models

Pydantic schemas in `src/food_cooker/models/`:
- `UserProfile` — session_id, allergies, diet, cuisine_preference, dislikes, equipment_constraints, user_inventory
- `AdaptedRecipe` — name, ingredients (name+amount), steps (step_number+instruction), nutrition, adaptation_notes
