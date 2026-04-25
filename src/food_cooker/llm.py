from langchain_openai import ChatOpenAI
from food_cooker.settings import settings

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_llm(temperature: float = 0.7):
    """Get the configured LLM instance based on settings.llm_provider."""
    if settings.llm_provider == "dashscope":
        return ChatOpenAI(
            model=settings.dashscope_model_name,
            api_key=settings.dashscope_api_key,
            base_url=DASHSCOPE_BASE_URL,
            temperature=temperature,
        )
    else:
        return ChatOpenAI(
            model=settings.openai_model_name,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )
