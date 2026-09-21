from langchain_core.documents import Document

from .config import settings
from .embeddings import get_embeddings
from .pinecone_store import index


def retrieve(
    query,
    top_k=None,
):
    """
    Retrieve the recipe Documents that are most semantically similar
    to the user's query.
    """

    # Use the configured Top-K value unless the caller provides another value.
    number_of_results = (
        top_k
        if top_k is not None
        else settings.top_k
    )

    embedding_model = get_embeddings()
    pinecone_index = index()

    # Embed the query with the same model used to embed the recipe corpus.
    query_vector = embedding_model.embed_query(
        query
    )

    # Retrieve the Top-K recipe vectors ranked by cosine similarity.
    search_results = pinecone_index.query(
        vector=query_vector,
        top_k=number_of_results,
        include_metadata=True,
    )

    retrieved_documents = []

    # Reconstruct LangChain Documents from the Pinecone search results.
    for match in search_results.matches:
        metadata = dict(
            match.metadata or {}
        )

        recipe_text = metadata.pop(
            "text",
            ""
        )

        document = Document(
            page_content=recipe_text,
            metadata=metadata,
        )

        similarity_score = float(
            match.score
        )

        retrieved_documents.append(
            (
                document,
                similarity_score,
            )
        )

    return retrieved_documents