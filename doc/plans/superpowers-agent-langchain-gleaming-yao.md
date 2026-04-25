# Personalized Recipe Generation Agent - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangChain-powered AI agent that generates personalized recipes from user constraints (diet, allergies, equipment), retrieves食谱 from RAG, adapts them, calculates nutrition, and generates shopping lists with memory across sessions.

**Architecture:** OpenAI Tools Agent with 6 custom tools + ConversationTokenBufferMemory, Chroma vector store for RAG retrieval, Chainlit for chat UI. Recipe adaptation done via a dedicated LLM call inside the `recipe_adaptor_tool`.

**Tech Stack:** Python 3.10+, LangChain (latest), Chroma, HuggingFaceEmbeddings (all-MiniLM-L6-v2), Pydantic, Chainlit, pytest, `openai` / `langchain-openai`.

---

## 1. Project Structure

```
d:\workspaceForTrae\FoodCooker\
├── src/food_cooker/
│   ├── __init__.py
│   ├── settings.py              # All env / config (API keys, paths, model names)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user_profile.py       # Pydantic schema: allergies, diet, cuisine_preference, etc.
│   │   └── recipe.py             # Pydantic schema: adapted recipe output
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── chroma_client.py      # Chroma PersistentClient + embedding setup
│   │   └── ingest.py            # CLI script: load dataset → embed → upsert to Chroma
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── executor.py           # build_agent() — wires tools + memory + LLM into AgentExecutor
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── user_profile_tool.py
│   │   │   ├── recipe_retriever_tool.py
│   │   │   ├── recipe_adaptor_tool.py
│   │   │   ├── nutrition_calculator_tool.py
│   │   │   ├── shopping_list_tool.py
│   │   │   └── feedback_tool.py
│   │   └── prompts.py           # System prompt string used by build_agent()
│   ├── memory/
│   │   ├── __init__.py
│   │   └── token_buffer.py       # ConversationTokenBufferMemory factory
│   └── ui/
│       ├── __init__.py
│       └── chainlit_app.py      # @cl.on_message handler, streaming, tool卡片rendering
├── data/
│   ├── recipes_raw.json         # Downloaded / cleaned Kaggle dataset
│   └── chroma_db/                # Chroma persistent store (git-ignored)
├── tests/
│   ├── conftest.py              # Shared fixtures (mock_chroma, mock_llm)
│   ├── test_tools/
│   │   ├── test_user_profile_tool.py
│   │   ├── test_recipe_retriever_tool.py
│   │   ├── test_recipe_adaptor_tool.py
│   │   ├── test_nutrition_calculator_tool.py
│   │   ├── test_shopping_list_tool.py
│   │   └── test_feedback_tool.py
│   ├── test_agent/
│   │   └── test_executor.py     # Integration: tool + agent wiring
│   └── test_memory/
│       └── test_token_buffer.py
├── scripts/
│   └── ingest_recipes.py        # Standalone: dataset → embedding → Chroma upsert
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 2. Task Map

| Task | Description |
|------|-------------|
| 1 | Project scaffold: `pyproject.toml`, `.env.example`, `src/` packages, `tests/` layout |
| 2 | Settings: `settings.py` with `pydantic-settings`, env var loading, model name constants |
| 3 | Data models: `user_profile.py` and `recipe.py` Pydantic schemas |
| 4 | Vector store: `chroma_client.py` (ChromaPersistentClient + HuggingFaceEmbeddings) + `ingest.py` |
| 5 | Dataset ingestion script: `scripts/ingest_recipes.py` + sample `data/recipes_raw.json` |
| 6 | Tool: `user_profile_tool.py` (get/update, JSON file persistence keyed by session_id) |
| 7 | Tool: `recipe_retriever_tool.py` (Chroma similarity_search with metadata filtering) |
| 8 | Tool: `recipe_adaptor_tool.py` (internal LLM call with structured output parser) |
| 9 | Tool: `nutrition_calculator_tool.py` (ingredient → calorie/macro estimation) |
| 10 | Tool: `shopping_list_tool.py` (recipe ingredients vs user inventory diff) |
| 11 | Tool: `feedback_tool.py` (parse feedback text → update user_profile via tool call) |
| 12 | Memory: `token_buffer.py` (ConversationTokenBufferMemory, save/load to JSON) |
| 13 | Prompt: `prompts.py` (system message for agent role + decision flow) |
| 14 | Agent executor: `executor.py` (build_agent() → AgentExecutor with all tools + memory) |
| 15 | Chainlit UI: `chainlit_app.py` (streaming, tool result cards, session init) |
| 16 | Unit tests for all 6 tools |
| 17 | Integration test for agent executor |
| 18 | README + demo video note |

---

## 3. Task Details

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/food_cooker/__init__.py`
- Create: `src/food_cooker/models/__init__.py`
- Create: `src/food_cooker/vectorstore/__init__.py`
- Create: `src/food_cooker/agent/__init__.py`
- Create: `src/food_cooker/agent/tools/__init__.py`
- Create: `src/food_cooker/memory/__init__.py`
- Create: `src/food_cooker/ui/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_tools/__init__.py`
- Create: `tests/test_agent/__init__.py`
- Create: `tests/test_memory/__init__.py`
- Create: `scripts/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "food-cooker"
version = "0.1.0"
description = "Personalized recipe generation agent with LangChain + RAG"
requires-python = ">=3.10"
dependencies = [
    "langchain>=0.3.0",
    "langchain-openai",
    "langchain-community",
    "chromadb",
    "huggingface-huggingface-embeddings>=0.0.0",
    "pydantic>=2.0",
    "pydantic-settings",
    "chainlit>=0.7.0",
    "python-dotenv",
    "pytest>=8.0",
    "pytest-asyncio",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Write `.env.example`**

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o-mini
CHROMA_DB_PATH=data/chroma_db
USER_PROFILES_PATH=data/user_profiles.json
RECIPES_DATA_PATH=data/recipes_raw.json
HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_TOKEN_BUFFER=2000
```

- [ ] **Step 3: Write all `__init__.py` stub files**

Each `__init__.py` just exposes the public interface — e.g., `from .user_profile import UserProfile` in `models/__init__.py`.

---

### Task 2: Settings

**Files:**
- Create: `src/food_cooker/settings.py`

- [ ] **Step 1: Write `settings.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_model_name: str = "gpt-4o-mini"

    chroma_db_path: Path = BASE_DIR / "data" / "chroma_db"
    user_profiles_path: Path = BASE_DIR / "data" / "user_profiles.json"
    recipes_data_path: Path = BASE_DIR / "data" / "recipes_raw.json"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_token_buffer: int = 2000

settings = Settings()
```

---

### Task 3: Data Models

**Files:**
- Create: `src/food_cooker/models/user_profile.py`
- Create: `src/food_cooker/models/recipe.py`

- [ ] **Step 1: Write `user_profile.py`**

```python
from pydantic import BaseModel, Field
from typing import Optional

class UserProfile(BaseModel):
    session_id: str
    allergies: list[str] = Field(default_factory=list)
    diet: Optional[str] = None  # e.g., "low_carb", "keto", "vegetarian"
    cuisine_preference: Optional[str] = None  # e.g., "Chinese", "Italian"
    dislikes: list[str] = Field(default_factory=list)
    calorie_target: Optional[int] = None  # per meal
    spice_tolerance: str = "medium"  # "mild", "medium", "hot"
    equipment_constraints: list[str] = Field(default_factory=list)  # e.g., ["no_oven"]
    user_inventory: list[str] = Field(default_factory=list)  # ingredients user already has
    feedback_history: list[str] = Field(default_factory=list)  # recent feedback strings
```

- [ ] **Step 2: Write `recipe.py`**

```python
from pydantic import BaseModel, Field
from typing import Optional

class RecipeStep(BaseModel):
    step_number: int
    instruction: str
    duration_minutes: Optional[int] = None

class Ingredient(BaseModel):
    name: str
    amount: str  # e.g., "150g", "2 tbsp"
    unit: Optional[str] = None

class AdaptedRecipe(BaseModel):
    name: str
    cuisine: str
    tags: list[str]
    ingredients: list[Ingredient]
    steps: list[RecipeStep]
    estimated_calories: int
    protein_grams: float
    carbs_grams: float
    fat_grams: float
    prep_time_minutes: int
    cook_time_minutes: int
    adaptation_notes: list[str] = Field(default_factory=list)  # e.g., "replaced peanuts with sunflower seeds"
```

- [ ] **Step 3: Run test to verify models are importable**

```bash
cd d:\workspaceForTrae\FoodCooker && python -c "from food_cooker.models import UserProfile, AdaptedRecipe; print('models OK')"
```
Expected: `models OK`

---

### Task 4: Chroma Client

**Files:**
- Create: `src/food_cooker/vectorstore/chroma_client.py`

- [ ] **Step 1: Write `chroma_client.py`**

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from food_cooker.settings import settings

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_chroma_client(collection_name: str = "recipes") -> Chroma:
    return Chroma(
        client=chromadb.PersistentClient(path=str(settings.chroma_db_path)),
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
    )
```

Add `import chromadb` to dependencies in pyproject.toml.

---

### Task 5: Dataset Ingestion Script

**Files:**
- Create: `scripts/ingest_recipes.py`
- Create: `data/recipes_raw.json` (sample 10 recipes as starter dataset)

- [ ] **Step 1: Write `scripts/ingest_recipes.py`**

```python
"""
Standalone script: python scripts/ingest_recipes.py
Loads recipes from data/recipes_raw.json, embeds them, upserts to Chroma.
"""
import json
from langchain_core.documents import Document
from food_cooker.vectorstore.chroma_client import get_chroma_client, get_embedding_model

def load_recipes(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_documents(recipes: list[dict]) -> list[Document]:
    docs = []
    for r in recipes:
        text = f"Dish: {r['name']}, Tags: {', '.join(r.get('tags', []))}, Ingredients: {', '.join(r.get('ingredients', []))}"
        docs.append(Document(page_content=text, metadata={
            "name": r["name"],
            "ingredients": r.get("ingredients", []),
            "steps": r.get("steps", []),
            "tags": r.get("tags", []),
            "cuisine": r.get("cuisine", "unknown"),
            "nutrition": r.get("nutrition", {}),
        }))
    return docs

def ingest():
    recipes = load_recipes("data/recipes_raw.json")
    docs = build_documents(recipes)
    db = get_chroma_client()
    db.add_documents(docs)
    print(f"Ingested {len(docs)} recipes into Chroma.")

if __name__ == "__main__":
    ingest()
```

- [ ] **Step 2: Write `data/recipes_raw.json` with 10 sample Chinese/low-carb recipes** — include fields: `name`, `ingredients` (list of strings with amount), `steps` (list of strings), `tags` (list), `cuisine` (string), `nutrition` ({calories, protein, carbs, fat}).

- [ ] **Step 3: Run ingestion**

```bash
cd d:\workspaceForTrae\FoodCooker && python scripts/ingest_recipes.py
```
Expected: "Ingested 10 recipes into Chroma."

---

### Task 6: user_profile_tool

**Files:**
- Create: `src/food_cooker/agent/tools/user_profile_tool.py`

- [ ] **Step 1: Write `user_profile_tool.py`**

```python
import json
from pathlib import Path
from typing import Literal
from langchain_core.tools import tool
from food_cooker.models.user_profile import UserProfile
from food_cooker.settings import settings

def _load_profiles() -> dict:
    path = settings.user_profiles_path
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_profiles(profiles: dict) -> None:
    path = settings.user_profiles_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

@tool
def user_profile_tool(
    action: Literal["get", "update"],
    session_id: str,
    preferences: dict | None = None,
) -> dict:
    """Read or update the user profile for the current session.
    Use 'get' to retrieve the profile, 'update' to modify it.
    Returns the full profile after any update."""
    profiles = _load_profiles()
    if action == "get":
        profile_data = profiles.get(session_id, {"session_id": session_id})
        return profile_data
    elif action == "update":
        if preferences is None:
            return {"error": "preferences required for update action"}
        current = profiles.get(session_id, {"session_id": session_id})
        current.update(preferences)
        profiles[session_id] = current
        _save_profiles(profiles)
        return current
    return {"error": "invalid action"}
```

---

### Task 7: recipe_retriever_tool

**Files:**
- Create: `src/food_cooker/agent/tools/recipe_retriever_tool.py`

- [ ] **Step 1: Write `recipe_retriever_tool.py`**

```python
from typing import Optional
from langchain_core.tools import tool
from food_cooker.vectorstore.chroma_client import get_chroma_client

@tool
def recipe_retriever_tool(
    query: str,
    tags_filter: Optional[list[str]] = None,
    cuisine_filter: Optional[str] = None,
    k: int = 3,
) -> dict:
    """Retrieve the top-k most relevant recipes from the vector store.
    Supports optional metadata filtering by tags and cuisine."""
    db = get_chroma_client()
    filter_dict = {}
    if tags_filter:
        filter_dict["tags"] = {"$in": tags_filter}
    if cuisine_filter:
        filter_dict["cuisine"] = cuisine_filter

    results = db.similarity_search(query, k=k, filter=filter_dict if filter_dict else None)
    recipes = [
        {
            "name": r.metadata.get("name", ""),
            "cuisine": r.metadata.get("cuisine", ""),
            "tags": r.metadata.get("tags", []),
            "ingredients": r.metadata.get("ingredients", []),
            "steps": r.metadata.get("steps", []),
            "nutrition": r.metadata.get("nutrition", {}),
        }
        for r in results
    ]
    return {"recipes": recipes, "count": len(recipes)}
```

---

### Task 8: recipe_adaptor_tool

**Files:**
- Create: `src/food_cooker/agent/tools/recipe_adaptor_tool.py`

- [ ] **Step 1: Write `recipe_adaptor_tool.py`**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import HumanMessage
from langchain_core.tools import tool
from food_cooker.models.recipe import AdaptedRecipe
from food_cooker.settings import settings

ADAPTOR_PROMPT = """You are a professional chef adapting a recipe for a user with specific constraints.
Base recipe:
{name}
Ingredients: {ingredients}
Steps: {steps}
Tags: {tags}

User profile constraints:
- Allergies: {allergies}
- Diet: {diet}
- Cuisine preference: {cuisine_preference}
- Dislikes: {dislikes}
- Equipment constraints: {equipment_constraints}
- Spice tolerance: {spice_tolerance}

Adapt the recipe to satisfy all constraints. Replace or remove allergenic ingredients.
Return a JSON object with the adapted recipe following this schema:
{{
  "name": "...",
  "cuisine": "...",
  "tags": [...],
  "ingredients": [{{"name": "...", "amount": "..."}}],
  "steps": [{{"step_number": 1, "instruction": "..."}}],
  "estimated_calories": ...,
  "protein_grams": ...,
  "carbs_grams": ...,
  "fat_grams": ...,
  "prep_time_minutes": ...,
  "cook_time_minutes": ...,
  "adaptation_notes": [...]
}}"""

@tool
def recipe_adaptor_tool(base_recipe: str, user_profile: str) -> dict:
    """Adapt a base recipe according to user preferences and constraints.
    base_recipe: JSON string of retrieved recipe metadata.
    user_profile: JSON string of the user profile."""
    import json
    recipe_data = json.loads(base_recipe)
    profile_data = json.loads(user_profile)

    prompt = ChatPromptTemplate.from_template(ADAPTOR_PROMPT)
    chain = prompt | ChatOpenAI(
        model=settings.openai_model_name,
        api_key=settings.openai_api_key,
        temperature=0.7,
    ) | JsonOutputParser()

    try:
        adapted = chain.invoke({
            "name": recipe_data.get("name", ""),
            "ingredients": ", ".join(recipe_data.get("ingredients", [])),
            "steps": " | ".join(str(s) for s in recipe_data.get("steps", [])),
            "tags": ", ".join(recipe_data.get("tags", [])),
            "allergies": ", ".join(profile_data.get("allergies", [])) or "none",
            "diet": profile_data.get("diet", "none"),
            "cuisine_preference": profile_data.get("cuisine_preference", "any"),
            "dislikes": ", ".join(profile_data.get("dislikes", [])) or "none",
            "equipment_constraints": ", ".join(profile_data.get("equipment_constraints", [])) or "none",
            "spice_tolerance": profile_data.get("spice_tolerance", "medium"),
        })
        return adapted
    except Exception as e:
        return {"error": str(e), "adapted_recipe": None}
```

---

### Task 9: nutrition_calculator_tool

**Files:**
- Create: `src/food_cooker/agent/tools/nutrition_calculator_tool.py`

- [ ] **Step 1: Write `nutrition_calculator_tool.py`**

```python
from langchain_core.tools import tool

# Hardcoded per 100g / common unit for POC — extend with USDA CSV or Nutritionix API
NUTRITION_DB: dict[str, dict] = {
    "chicken breast": {"cal": 165, "protein": 31, "carbs": 0, "fat": 3.6},
    "broccoli": {"cal": 34, "protein": 2.8, "carbs": 7, "fat": 0.4},
    "soy sauce": {"cal": 53, "protein": 8, "carbs": 4, "fat": 0.6},
    "olive oil": {"cal": 884, "protein": 0, "carbs": 0, "fat": 100},
    "egg": {"cal": 155, "protein": 13, "carbs": 1.1, "fat": 11},
    # ... extend as needed
}

@tool
def nutrition_calculator_tool(ingredients: list[dict]) -> dict:
    """Calculate estimated nutrition for a list of ingredients.
    Each ingredient: {{"name": "...", "amount": "..."}}.
    Returns total calories, protein, carbs, fat."""
    totals = {"calories": 0, "protein_grams": 0, "carbs_grams": 0, "fat_grams": 0}
    for item in ingredients:
        name = item.get("name", "").lower()
        amount_str = item.get("amount", "100g")
        # Simple: assume amount is in grams if numeric, else parse "150g"
        try:
            grams = int("".join(filter(str.isdigit, amount_str))) if any(c.isdigit() for c in amount_str) else 100
        except ValueError:
            grams = 100

        nutrient = NUTRITION_DB.get(name, {"cal": 50, "protein": 2, "carbs": 5, "fat": 2})
        scale = grams / 100.0
        totals["calories"] += nutrient["cal"] * scale
        totals["protein_grams"] += nutrient["protein"] * scale
        totals["carbs_grams"] += nutrient["carbs"] * scale
        totals["fat_grams"] += nutrient["fat"] * scale

    return {k: round(v, 1) for k, v in totals.items()}
```

---

### Task 10: shopping_list_tool

**Files:**
- Create: `src/food_cooker/agent/tools/shopping_list_tool.py`

- [ ] **Step 1: Write `shopping_list_tool.py`**

```python
from langchain_core.tools import tool

@tool
def shopping_list_tool(
    recipe_ingredients: list[dict],
    user_inventory: list[str],
) -> dict:
    """Compare recipe ingredients against user's existing inventory.
    Return categorized shopping list of missing items."""
    inventory_lower = {i.lower().strip() for i in user_inventory}
    missing = []
    for ing in recipe_ingredients:
        name = ing.get("name", "").lower().strip()
        amount = ing.get("amount", "")
        # Check if user has any variant of this ingredient
        if not any(name in inv or inv in name for inv in inventory_lower):
            missing.append({"name": ing.get("name", ""), "amount": amount})

    # Categorize by type
    categories = {"调味品": [], "蛋白质": [], "蔬菜": [], "其他": []}
    for item in missing:
        n = item["name"].lower()
        if any(k in n for k in ["酱油", "盐", "糖", "醋", "料酒", "酱"]):
            categories["调味品"].append(item)
        elif any(k in n for k in ["鸡", "肉", "鱼", "虾", "蛋", "豆腐"]):
            categories["蛋白质"].append(item)
        elif any(k in n for k in ["菜", "西兰花", "白菜", "萝卜", "葱", "姜", "蒜"]):
            categories["蔬菜"].append(item)
        else:
            categories["其他"].append(item)

    return {"shopping_list": categories, "total_items": len(missing)}
```

---

### Task 11: feedback_tool

**Files:**
- Create: `src/food_cooker/agent/tools/feedback_tool.py`

- [ ] **Step 1: Write `feedback_tool.py`**

```python
import json
from langchain_core.tools import tool
from food_cooker.agent.tools.user_profile_tool import user_profile_tool

@tool
def feedback_tool(session_id: str, feedback_text: str) -> dict:
    """Parse user feedback text and update user profile accordingly.
    Examples: 'too oily' → add 'oily' to dislikes; 'too spicy' → adjust spice_tolerance.
    Returns updated profile and instructs agent to regenerate."""
    feedback_lower = feedback_text.lower()

    updates = {}
    if any(w in feedback_lower for w in ["太油", "oily", "greasy"]):
        updates.setdefault("dislikes", []).append("oily_food")
    if any(w in feedback_lower for w in ["太辣", "spicy", "hot"]):
        old_tolerance = "hot"
        updates["spice_tolerance"] = "mild"
    if any(w in feedback_lower for w in ["太淡", "bland", "not flavorful"]):
        updates.setdefault("dislikes", []).append("bland_food")

    if not updates:
        updates["feedback_history"] = [feedback_text]
    else:
        updates["feedback_history"] = [f"User feedback: {feedback_text}"]

    updated_profile = user_profile_tool.invoke({
        "action": "update",
        "session_id": session_id,
        "preferences": updates,
    })

    return {
        "status": "profile_updated",
        "updated_profile": updated_profile,
        "regenerate": True,
        "message": "Preferences updated. Please regenerate the recipe considering the new feedback.",
    }
```

---

### Task 12: Memory Module

**Files:**
- Create: `src/food_cooker/memory/token_buffer.py`

- [ ] **Step 1: Write `token_buffer.py`**

```python
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI
from food_cooker.settings import settings

def create_memory(session_id: str) -> ConversationTokenBufferMemory:
    llm = ChatOpenAI(model=settings.openai_model_name, api_key=settings.openai_api_key)
    memory = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=settings.max_token_buffer,
        memory_key="chat_history",
        return_messages=True,
    )
    return memory

def save_memory(memory: ConversationTokenBufferMemory, path: str) -> None:
    # Serialise chat_history to JSON for cross-session persistence
    import json
    messages = memory.chat_memory.messages
    data = [{"type": m.type, "content": m.content} for m in messages]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_memory(memory: ConversationTokenBufferMemory, path: str) -> None:
    import json
    from langchain.schema import HumanMessage, AIMessage
    if not Path(path).exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if item["type"] == "human":
            memory.chat_memory.add_message(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            memory.chat_memory.add_message(AIMessage(content=item["content"]))
```

---

### Task 13: Agent Prompts

**Files:**
- Create: `src/food_cooker/agent/prompts.py`

- [ ] **Step 1: Write `prompts.py`**

```python
SYSTEM_PROMPT = """You are a personalized recipe assistant. When a user describes what they want to eat or cook, you must:

1. Call user_profile_tool with action='get' and session_id to retrieve user preferences.
2. Understand the user's request including: cuisine, diet, allergies, available ingredients, equipment, time constraints.
3. Call recipe_retriever_tool with the user's request and relevant filters (tags, cuisine) to get candidate recipes.
4. Call recipe_adaptor_tool with the best candidate recipe and user profile to produce a tailored recipe.
5. Call nutrition_calculator_tool with the adapted recipe ingredients to get nutrition estimates.
6. Call shopping_list_tool with recipe ingredients and user's inventory to generate a shopping list.
7. Present the complete response with: recipe name, ingredients, steps, nutrition, and shopping list in a structured format.
8. If the user gives feedback (e.g., 'too spicy', 'too oily'), call feedback_tool to update preferences.
   After feedback update, re-run from step 3 to regenerate the recipe.

Always use the tools in order. If a step fails, report the error clearly to the user.
When adapting recipes, ALWAYS respect allergy and diet constraints.
"""

USER_PROFILE_REMINDER = """Key user constraints to always respect:
- Allergies: {allergies}
- Diet: {diet}
- Equipment: {equipment_constraints}
- Spice tolerance: {spice_tolerance}
"""
```

---

### Task 14: Agent Executor

**Files:**
- Create: `src/food_cooker/agent/executor.py`

- [ ] **Step 1: Write `executor.py`**

```python
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from food_cooker.agent.tools import (
    user_profile_tool,
    recipe_retriever_tool,
    recipe_adaptor_tool,
    nutrition_calculator_tool,
    shopping_list_tool,
    feedback_tool,
)
from food_cooker.memory.token_buffer import create_memory
from food_cooker.agent.prompts import SYSTEM_PROMPT
from food_cooker.settings import settings

def build_agent():
    llm = ChatOpenAI(
        model=settings.openai_model_name,
        api_key=settings.openai_api_key,
        temperature=0.7,
    )

    tools = [
        user_profile_tool,
        recipe_retriever_tool,
        recipe_adaptor_tool,
        nutrition_calculator_tool,
        shopping_list_tool,
        feedback_tool,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("placeholder", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    memory = create_memory(session_id="default")

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )
    return executor

def run_agent(query: str, session_id: str = "default"):
    executor = build_agent()
    result = executor.invoke({"input": query})
    return result
```

---

### Task 15: Chainlit UI

**Files:**
- Create: `src/food_cooker/ui/chainlit_app.py`

- [ ] **Step 1: Write `chainlit_app.py`**

```python
import chainlit as cl
from food_cooker.agent.executor import build_agent

@cl.on_message
async def main(message: cl.Message):
    agent = build_agent()
    msg = cl.Message(content="")
    await msg.send()

    # Stream the response
    await cl.sleep(0.1)  # allow UI to render

    result = await cl.make_async(agent.invoke)({"input": message.content})

    await msg.update(content=result["output"])
```

---

### Task 16: Unit Tests for All 6 Tools

**Files:**
- Create: `tests/test_tools/test_user_profile_tool.py`
- Create: `tests/test_tools/test_recipe_retriever_tool.py`
- Create: `tests/test_tools/test_recipe_adaptor_tool.py`
- Create: `tests/test_tools/test_nutrition_calculator_tool.py`
- Create: `tests/test_tools/test_shopping_list_tool.py`
- Create: `tests/test_tools/test_feedback_tool.py`

Example for `test_nutrition_calculator_tool.py`:

```python
from food_cooker.agent.tools.nutrition_calculator_tool import nutrition_calculator_tool

def test_calculates_chicken_broccoli():
    ingredients = [
        {"name": "chicken breast", "amount": "150g"},
        {"name": "broccoli", "amount": "100g"},
    ]
    result = nutrition_calculator_tool.invoke({"ingredients": ingredients})
    assert result["calories"] > 0
    assert result["protein_grams"] > 20
```

All 6 test files follow the same pattern — call `tool.invoke(...)` with fixture inputs and assert structured output shapes/values.

---

### Task 17: Integration Test for Agent Executor

**Files:**
- Create: `tests/test_agent/test_executor.py`

- [ ] **Step 1: Write `test_executor.py`**

```python
import pytest
from unittest.mock import patch

@patch("food_cooker.agent.tools.user_profile_tool._load_profiles", return_value={})
@patch("food_cooker.agent.executor.ChatOpenAI")
def test_agent_runs_full_flow(mock_llm, mock_profiles):
    # Set up mock to return a simple response
    mock_instance = mock_llm.return_value
    mock_instance.invoke.return_value = {"output": "Mocked agent response"}

    from food_cooker.agent.executor import build_agent
    agent = build_agent()
    result = agent.invoke({"input": "I want a low-carb Chinese dinner with chicken"})
    assert "output" in result
```

---

## 4. Verification

1. **Install dependencies:** `pip install -e .` then `pip install -e ".[dev]"`)
2. **Run unit tests:** `pytest tests/test_tools/ -v` — all 6 tools should pass
3. **Run ingestion script:** `python scripts/ingest_recipes.py` — verify "Ingested N recipes"
4. **Start Chainlit UI:** `chainlit run src/food_cooker/ui/chainlit_app.py --port 8000` and send a test message
5. **Test end-to-end:** Send "I want high-protein low-carb Chinese dinner, no oven, I'm allergic to peanuts" → expect adapted recipe + shopping list + nutrition
6. **Test feedback loop:** Send "too spicy" → expect preference update + regeneration

---

**Plan complete.** Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?