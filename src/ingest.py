import argparse

import pandas as pd

from .documents import make_documents
from .embeddings import get_embeddings
from .pinecone_store import index


def ingest(
    csv_path,
    limit=None,
    batch_size=64,
):
    """
    Load the recipe corpus, create embeddings, and upload them to Pinecone.

    Current chunking strategy:
        1 recipe row -> 1 LangChain Document -> 1 embedding vector
    """

    # Load the processed recipe corpus and create one Document per recipe.
    recipe_dataframe = pd.read_csv(csv_path)

    recipe_documents = make_documents(
        recipe_dataframe,
        limit,
    )

    embedding_model = get_embeddings()
    pinecone_index = index()

    total_documents = len(recipe_documents)

    print(f"Recipes to ingest: {total_documents}")

    # Process recipes in batches so embedding and upload stay memory-efficient.
    for batch_start in range(
        0,
        total_documents,
        batch_size,
    ):
        batch_end = batch_start + batch_size

        batch_documents = recipe_documents[
            batch_start:batch_end
        ]

        # No text splitter is used here, so each recipe produces one embedding.
        recipe_texts = [
            document.page_content
            for document in batch_documents
        ]

        embedding_vectors = embedding_model.embed_documents(
            recipe_texts
        )

        pinecone_records = []

        # Pair each recipe Document with its corresponding embedding vector.
        for document, embedding_vector in zip(
            batch_documents,
            embedding_vectors,
        ):
            metadata = dict(document.metadata)

            # Store the recipe text so it can be returned as RAG context later.
            metadata["text"] = document.page_content

            pinecone_record = {
                "id": f"recipe-{metadata['row_id']}",
                "values": embedding_vector,
                "metadata": metadata,
            }

            pinecone_records.append(
                pinecone_record
            )

        # Upload this batch of recipe vectors and metadata to Pinecone.
        pinecone_index.upsert(
            vectors=pinecone_records
        )

        processed_count = min(
            batch_end,
            total_documents,
        )

        print(
            f"Upserted "
            f"{processed_count}/{total_documents}"
        )

    print(
        f"Ingestion complete: "
        f"{total_documents} recipes indexed."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Embed the processed recipe corpus "
            "and upload it to Pinecone."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the processed recipe CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of recipes "
            "to ingest. Useful for testing."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Number of recipes to embed and upload "
            "in each batch."
        ),
    )

    arguments = parser.parse_args()

    ingest(
        csv_path=arguments.csv,
        limit=arguments.limit,
        batch_size=arguments.batch_size,
    )


if __name__ == "__main__":
    main()