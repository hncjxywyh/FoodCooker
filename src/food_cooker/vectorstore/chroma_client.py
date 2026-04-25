import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from dashscope import TextEmbedding
from food_cooker.settings import settings


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


def get_embedding_model():
    if settings.embedding_provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=settings.huggingface_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif settings.embedding_provider == "dashscope":
        return DashScopeEmbeddings(
            model=settings.dashscope_embedding_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def get_chroma_client(collection_name: str = "recipes") -> Chroma:
    return Chroma(
        client=chromadb.PersistentClient(path=str(settings.chroma_db_path)),
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
    )