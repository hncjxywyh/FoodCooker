import logging
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from dashscope import TextEmbedding
from food_cooker.settings import settings

logger = logging.getLogger(__name__)


class DashScopeEmbeddings(Embeddings):
    """Custom DashScope embeddings implementation for LangChain."""

    def __init__(self, model: str = "text-embedding-v3", api_key: str = "", base_url: str = ""):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        responses = TextEmbedding.call(
            model=self.model,
            input=texts,
            api_key=self.api_key,
        )
        if responses.output is None:
            raise RuntimeError(f"DashScope API error: {responses.code} - {responses.message}")
        embeddings = []
        for resp in responses.output["embeddings"]:
            embeddings.append(resp["embedding"])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = TextEmbedding.call(
            model=self.model,
            input=text,
            api_key=self.api_key,
        )
        if response.output is None:
            raise RuntimeError(f"DashScope API error: {response.code} - {response.message}")
        return response.output["embeddings"][0]["embedding"]


_chroma_client: Chroma | None = None

_embedding_model: Embeddings | None = None


def get_embedding_model() -> Embeddings:
    global _embedding_model
    if _embedding_model is None:
        provider = settings.embedding_provider
        logger.info(f"Initializing embedding model provider={provider}")
        if provider == "huggingface":
            _embedding_model = HuggingFaceEmbeddings(
                model_name=settings.huggingface_embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        elif provider == "dashscope":
            _embedding_model = DashScopeEmbeddings(
                model=settings.dashscope_embedding_model,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
    return _embedding_model


def get_chroma_client(collection_name: str = "recipes") -> Chroma:
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing Chroma client collection={collection_name} path={settings.chroma_db_path}")
        _chroma_client = Chroma(
            client=chromadb.PersistentClient(path=str(settings.chroma_db_path)),
            collection_name=collection_name,
            embedding_function=get_embedding_model(),
        )
    return _chroma_client