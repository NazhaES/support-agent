import hashlib
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from pinecone import Pinecone , ServerlessSpec
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder
from app.core.config import get_settings

settings = get_settings()


_embeddings = None
_index = None
_vectorstore = None
_reranker = None
_bm25 = None
_bm25_built_at = 0.0


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}



def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "text-embedding-3-small" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized:
        return 3072
    if "text-embedding-ada-002" in normalized:
        return 1536
    if "all-minilm" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model '{model_name or settings.embedding_model}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )



MODEL_FILES = ["*.json", "*.txt", "*.safetensors"]



def local_model(repo_id: str) -> str:
    path = Path(settings.models_dir) / repo_id.split("/")[-1]
    if not path.exists():
        tmp = path.with_name(path.name + ".downloading")
        snapshot_download(repo_id=repo_id, local_dir=tmp, allow_patterns=MODEL_FILES)
        tmp.rename(path)
    return str(path)



def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=local_model(settings.embedding_model))
    return _embeddings




def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    names = [x["name"] for x in pc.list_indexes()]

    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(settings.pinecone_index_name)
        current_dimension = getattr(index_info, "dimension", None)
        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension is not None and current_dimension != desired_dimension:
            pc.delete_index(name=settings.pinecone_index_name)
            while settings.pinecone_index_name in [x["name"] for x in pc.list_indexes()]:
                time.sleep(1)

    if settings.pinecone_index_name not in [x["name"] for x in pc.list_indexes()]:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(settings.pinecone_index_name)





def get_vectorstore():
    global _index, _vectorstore
    if _vectorstore is None:
        _index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=_index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore



def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})



def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(local_model(settings.rerank_model))
    return _reranker



def load_pinecone_chunks() -> list[Document]:
    get_vectorstore()
    chunks = []
    for ids in _index.list(namespace=settings.pinecone_namespace):
        fetched = _index.fetch(ids=ids, namespace=settings.pinecone_namespace)
        for vector in fetched.vectors.values():
            metadata = dict(vector.metadata or {})
            text = metadata.pop("text", "")
            if text:
                chunks.append(Document(page_content=text, metadata=metadata))
    return chunks



def get_bm25():
    global _bm25, _bm25_built_at
    if _bm25 is None or time.time() - _bm25_built_at > settings.bm25_refresh_seconds:
        chunks = load_pinecone_chunks()
        _bm25 = BM25Retriever.from_documents(chunks, k=settings.rerank_candidates) if chunks else None
        _bm25_built_at = time.time()
    return _bm25



def warm_up():
    get_embeddings().embed_query("warm up")
    get_reranker().predict([("warm up", "warm up")])
    get_bm25()



def search_kb(query: str):
    candidates = get_vectorstore().similarity_search(query, k=settings.rerank_candidates)
    bm25 = get_bm25()
    if bm25:
        candidates += bm25.invoke(query)
    candidates = list({d.page_content: d for d in candidates}.values())
    if not candidates:
        return []
    scores = get_reranker().predict([(query, d.page_content) for d in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[: settings.top_k]]



def file_name(doc: Document) -> str:
    return Path(str(doc.metadata.get("source", ""))).name



def delete_files(names: set[str]):
    get_vectorstore()
    stale = []
    for ids in _index.list(namespace=settings.pinecone_namespace):
        fetched = _index.fetch(ids=ids, namespace=settings.pinecone_namespace)
        for vector_id, vector in fetched.vectors.items():
            if Path(str((vector.metadata or {}).get("source", ""))).name in names:
                stale.append(vector_id)
    for i in range(0, len(stale), 1000):
        _index.delete(ids=stale[i:i + 1000], namespace=settings.pinecone_namespace)
    return len(stale)



def add_documents(chunks):
    global _bm25
    store = get_vectorstore()
    delete_files({file_name(c) for c in chunks})
    counters = {}
    ids = []
    for chunk in chunks:
        name = file_name(chunk)
        counters[name] = counters.get(name, 0) + 1
        ids.append(hashlib.sha1(f"{name}:{counters[name]}".encode()).hexdigest())
    result = store.add_documents(chunks, ids=ids)
    _bm25 = None
    return result
