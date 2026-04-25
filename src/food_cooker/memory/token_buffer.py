import json
from pathlib import Path
from langchain_classic.memory import ConversationTokenBufferMemory
from food_cooker.llm import get_llm


def create_memory(session_id: str) -> ConversationTokenBufferMemory:
    llm = get_llm(temperature=0.0)
    memory = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=2000,
        memory_key="chat_history",
        return_messages=True,
    )
    return memory


def save_memory(memory: ConversationTokenBufferMemory, path: str | Path) -> None:
    messages = memory.chat_memory.messages
    data = [{"type": m.type, "content": m.content} for m in messages]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_memory(memory: ConversationTokenBufferMemory, path: str | Path) -> None:
    from langchain.schema import HumanMessage, AIMessage

    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if item["type"] == "human":
            memory.chat_memory.add_message(HumanMessage(content=item["content"]))
        elif item["type"] == "ai":
            memory.chat_memory.add_message(AIMessage(content=item["content"]))