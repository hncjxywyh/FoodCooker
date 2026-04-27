import logging
from functools import lru_cache
from langchain_openai import ChatOpenAI
from food_cooker.settings import settings

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.7):
    """Get the configured LLM instance based on settings.llm_provider.
    Instances are cached per temperature value."""
    provider = settings.llm_provider
    model = settings.dashscope_model_name if provider == "dashscope" else settings.openai_model_name
    logger.debug(f"get_llm provider={provider} model={model} temperature={temperature}")
    try:
        if provider == "dashscope":
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
    except Exception as e:
        logger.error(f"get_llm failed provider={provider} model={model}: {e}")
        raise
