from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.rag.chains.hybrid import create_hybrid_rag_chain
from app.rag.chains.hyde import create_hyde_rag_chain
from app.rag.chains.multi_query import create_multi_query_rag_chain
from app.rag.chains.naive import create_naive_rag_chain
from app.rag.chains.rag_fusion import create_rag_fusion_chain
from app.rag.chains.rerank import create_rerank_rag_chain
from app.rag.chains.route import create_route_rag_chain

ChainConstructorType = Callable[[BaseChatModel], Runnable[str, dict[str, Any]]]

chain_constructor_by_name: dict[str, ChainConstructorType] = {
    "naive": create_naive_rag_chain,
    "hyde": create_hyde_rag_chain,
    "multi_query": create_multi_query_rag_chain,
    "rag_fusion": create_rag_fusion_chain,
    "rerank": create_rerank_rag_chain,
    "route": create_route_rag_chain,
    "hybrid": create_hybrid_rag_chain,
}


def create_rag_chain(
    chain_name: str,
    model: BaseChatModel,
) -> Runnable[str, dict[str, Any]]:
    if chain_name not in chain_constructor_by_name:
        raise ValueError(f"Unknown chain name: {chain_name}")

    chain_constructor = chain_constructor_by_name[chain_name]
    chain = chain_constructor(model)

    return chain.with_config({"run_name": chain_name})
