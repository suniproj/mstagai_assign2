import pandas as pd
from langchain_core.documents import Document

from .parsing import as_list, missing, nutrient


REQUIRED_COLUMNS = {
    "recipe_name",
    "source",
    "url",
    "servings",
    "calories",
    "diet_labels",
    "health_labels",
    "cautions",
    "cuisine_type",
    "meal_type",
    "dish_type",
    "ingredient_lines",
    "ingredients",
    "total_nutrients",
}


def to_number(value):
    """Convert a value to float when possible."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_per_serving(total_value, servings):
    """Calculate a per-serving value from a whole-recipe total."""

    if total_value is None or not servings:
        return None

    return round(total_value / servings, 2)


def validate_columns(dataframe):
    """Verify that the recipe dataset contains the fields required by the RAG app."""

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"Missing required CSV columns: {missing_names}"
        )


def make_documents(dataframe, limit=None):
    """
    Convert recipe rows into LangChain Documents.

    Current chunking strategy:
        1 recipe row -> 1 LangChain Document -> 1 embedding vector
    """

    validate_columns(dataframe)

    if limit is not None:
        dataframe = dataframe.head(limit)

    documents = []

    for row_id, recipe in dataframe.iterrows():

        # Extract the recipe's basic numeric fields.
        servings = to_number(recipe.get("servings")) or 1.0
        total_calories = to_number(recipe.get("calories"))

        calories_per_serving = calculate_per_serving(
            total_calories,
            servings,
        )

        # Extract whole-recipe nutrition values from total_nutrients.
        total_protein, protein_unit = nutrient(
            recipe.get("total_nutrients"),
            "PROCNT",
        )

        total_fat, fat_unit = nutrient(
            recipe.get("total_nutrients"),
            "FAT",
        )

        total_carbohydrates, carbohydrate_unit = nutrient(
            recipe.get("total_nutrients"),
            "CHOCDF",
        )

        protein_per_serving = calculate_per_serving(
            total_protein,
            servings,
        )

        fat_per_serving = calculate_per_serving(
            total_fat,
            servings,
        )

        carbohydrates_per_serving = calculate_per_serving(
            total_carbohydrates,
            servings,
        )

        # Parse multi-value recipe fields from the CSV into Python lists.
        ingredient_lines = as_list(
            recipe.get("ingredient_lines")
        )

        if not ingredient_lines:
            ingredient_lines = as_list(
                recipe.get("ingredients")
            )

        diet_labels = as_list(
            recipe.get("diet_labels")
        )

        health_labels = as_list(
            recipe.get("health_labels")
        )

        cautions = as_list(
            recipe.get("cautions")
        )

        cuisine_types = as_list(
            recipe.get("cuisine_type")
        )

        meal_types = as_list(
            recipe.get("meal_type")
        )

        dish_types = as_list(
            recipe.get("dish_type")
        )

        recipe_name = str(
            recipe.get(
                "recipe_name",
                f"Recipe {row_id}",
            )
        )

        source = str(
            recipe.get(
                "source",
                "Unknown",
            )
        )

        recipe_url = (
            ""
            if missing(recipe.get("url"))
            else str(recipe.get("url"))
        )

        # Build the readable recipe text that will be embedded and sent to the LLM.
        content_lines = [
            f"Recipe: {recipe_name}",
            f"Source: {source}",
            f"URL: {recipe_url or 'unavailable'}",
            f"Servings: {servings:g}",
            f"Meal type: {', '.join(map(str, meal_types)) or 'unknown'}",
            f"Cuisine: {', '.join(map(str, cuisine_types)) or 'unknown'}",
            f"Dish type: {', '.join(map(str, dish_types)) or 'unknown'}",
            f"Diet labels: {', '.join(map(str, diet_labels)) or 'none listed'}",
            f"Health labels: {', '.join(map(str, health_labels)) or 'none listed'}",
            f"Cautions: {', '.join(map(str, cautions)) or 'none listed'}",
            "",
            "Ingredients:",
        ]

        content_lines.extend(
            f"- {ingredient}"
            for ingredient in ingredient_lines
        )

        content_lines.extend(
            [
                "",
                "Nutrition:",
            ]
        )

        if total_calories is not None:
            content_lines.append(
                f"- Calories, whole recipe: "
                f"{total_calories:.2f} kcal"
            )

            content_lines.append(
                f"- Calories per serving: "
                f"{calories_per_serving:.2f} kcal"
            )

        if total_protein is not None:
            content_lines.append(
                f"- Protein, whole recipe: "
                f"{total_protein:.2f} {protein_unit or ''}"
            )

            content_lines.append(
                f"- Protein per serving: "
                f"{protein_per_serving:.2f} {protein_unit or ''}"
            )

        if total_fat is not None:
            content_lines.append(
                f"- Fat per serving: "
                f"{fat_per_serving:.2f} {fat_unit or ''}"
            )

        if total_carbohydrates is not None:
            content_lines.append(
                f"- Carbohydrates per serving: "
                f"{carbohydrates_per_serving:.2f} "
                f"{carbohydrate_unit or ''}"
            )

        page_content = "\n".join(content_lines)

        # Keep multi-value attributes as lists so Pinecone can filter them exactly.
        metadata = {
            "row_id": int(row_id),
            "recipe_name": recipe_name[:250],
            "source": source[:250],
            "url": recipe_url[:1000],
            "meal_type": meal_types,
            "diet_labels": diet_labels,
            "health_labels": health_labels,
            "cautions": cautions,
            "servings": float(servings),
        }

        if calories_per_serving is not None:
            metadata["calories_per_serving"] = float(
                calories_per_serving
            )

        if protein_per_serving is not None:
            metadata["protein_per_serving"] = float(
                protein_per_serving
            )

        document = Document(
            page_content=page_content,
            metadata=metadata,
        )

        documents.append(document)

    return documents