from langchain_core.documents import Document

from .config import settings
from .embeddings import get_embeddings
from .pinecone_store import index


def retrieve(query, top_k=None, metadata_filter=None):
    """
    Retrieve recipe Documents that are semantically similar to the query.

    An optional metadata filter can enforce exact constraints before
    Pinecone ranks the remaining recipes by semantic similarity.
    """

    number_of_results = top_k if top_k is not None else settings.top_k

    embedding_model = get_embeddings()
    pinecone_index = index()

    # Embed the query using the same model used for the recipe corpus.
    query_vector = embedding_model.embed_query(query)

    search_arguments = {
        "vector": query_vector,
        "top_k": number_of_results,
        "include_metadata": True,
    }

    if metadata_filter:
        search_arguments["filter"] = metadata_filter

    # Filter eligible recipes first, then rank them by vector similarity.
    search_results = pinecone_index.query(**search_arguments)

    retrieved_documents = []

    for match in search_results.matches:
        metadata = dict(match.metadata or {})
        recipe_text = metadata.pop("text", "")

        document = Document(
            page_content=recipe_text,
            metadata=metadata,
        )

        retrieved_documents.append(
            (document, float(match.score))
        )

    return retrieved_documents