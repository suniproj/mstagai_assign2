import argparse
import json
from pathlib import Path

import pandas as pd


# Define the balanced corpus: 100 Vegan, 100 Vegetarian-only, 100 Non-Vegetarian.
SAMPLING_GROUPS = [
    ("Vegan Breakfast", "Vegan", ["breakfast"], 30),
    ("Vegan Lunch/Dinner", "Vegan", ["lunch/dinner"], 40),
    ("Vegan Snack", "Vegan", ["snack"], 30),

    ("Vegetarian Breakfast", "Vegetarian", ["breakfast"], 30),
    ("Vegetarian Lunch/Dinner", "Vegetarian", ["lunch/dinner"], 40),
    ("Vegetarian Snack", "Vegetarian", ["snack"], 30),

    ("Non-Vegetarian Breakfast", "Non-Vegetarian", ["breakfast"], 30),
    ("Non-Vegetarian Lunch/Dinner", "Non-Vegetarian", ["lunch/dinner"], 40),
    ("Non-Vegetarian Snack", "Non-Vegetarian", ["snack"], 30),
]


def parse_json_list(value):
    """Convert a JSON list stored in a CSV cell into a Python list."""

    if pd.isna(value):
        return []

    try:
        parsed_value = json.loads(value)

        if isinstance(parsed_value, list):
            return parsed_value

    except (json.JSONDecodeError, TypeError):
        pass

    return []


def has_required_nutrition(value):
    """Check that the recipe contains the nutrition fields used by the RAG app."""

    if pd.isna(value):
        return False

    try:
        nutrients = json.loads(value)

        required_nutrients = {
            "PROCNT",   # Protein
            "FAT",
            "CHOCDF",   # Carbohydrates
        }

        return required_nutrients.issubset(nutrients.keys())

    except (json.JSONDecodeError, TypeError):
        return False


def create_quality_mask(dataframe):
    """Keep recipes with the core fields needed for retrieval and per-serving nutrition."""

    return (
        dataframe["recipe_name"].notna()
        & dataframe["source"].notna()
        & dataframe["servings"].notna()
        & (dataframe["servings"] > 0)
        & dataframe["ingredient_lines"].notna()
        & dataframe["total_nutrients"].notna()
        & dataframe["health_labels"].notna()
        & dataframe["meal_type"].notna()
        & dataframe["total_nutrients"].apply(has_required_nutrition)
    )


def create_diet_mask(dataframe, diet_type):
    """Select Vegan, Vegetarian-only, or Non-Vegetarian recipes."""

    if diet_type == "Vegan":
        return dataframe["health_labels_parsed"].apply(
            lambda labels: "Vegan" in labels
        )

    if diet_type == "Vegetarian":
        return dataframe["health_labels_parsed"].apply(
            lambda labels:
                "Vegetarian" in labels
                and "Vegan" not in labels
        )

    if diet_type == "Non-Vegetarian":
        return dataframe["health_labels_parsed"].apply(
            lambda labels:
                "Vegetarian" not in labels
                and "Vegan" not in labels
        )

    raise ValueError(f"Unknown diet type: {diet_type}")


def prepare_corpus(
    input_path,
    output_path,
    random_seed=42,
):
    """Create the balanced 300-recipe corpus used by the RAG application."""

    dataframe = pd.read_csv(input_path)

    print(f"Raw recipes: {len(dataframe):,}")

    # Parse the multi-value JSON fields once so filtering is easier to read.
    dataframe["health_labels_parsed"] = dataframe[
        "health_labels"
    ].apply(parse_json_list)

    dataframe["meal_type_parsed"] = dataframe[
        "meal_type"
    ].apply(parse_json_list)

    quality_mask = create_quality_mask(dataframe)

    print(
        f"Recipes passing quality checks: "
        f"{quality_mask.sum():,}"
    )

    sampled_groups = []

    for (
        group_name,
        diet_type,
        meal_type,
        sample_size,
    ) in SAMPLING_GROUPS:

        diet_mask = create_diet_mask(
            dataframe,
            diet_type,
        )

        meal_type_mask = dataframe[
            "meal_type_parsed"
        ].apply(
            lambda meal_types:
                meal_types == meal_type
        )

        candidate_recipes = dataframe[
            quality_mask
            & diet_mask
            & meal_type_mask
        ]

        print(
            f"{group_name}: "
            f"{len(candidate_recipes):,} candidates"
        )

        if len(candidate_recipes) < sample_size:
            raise ValueError(
                f"Not enough recipes for {group_name}. "
                f"Need {sample_size}, "
                f"found {len(candidate_recipes)}."
            )

        # Pick a reproducible sample that satisfies this diet and meal-type group.
        sampled_recipes = candidate_recipes.sample(
            n=sample_size,
            random_state=random_seed,
        ).copy()

        sampled_recipes["sample_group"] = group_name

        sampled_groups.append(sampled_recipes)

    processed_corpus = pd.concat(
        sampled_groups,
        ignore_index=True,
    )

    # Helper columns were only needed during preprocessing, not by the RAG app.
    processed_corpus = processed_corpus.drop(
        columns=[
            "health_labels_parsed",
            "meal_type_parsed",
        ]
    )

    # Create the output directory if it does not already exist.
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_corpus.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Saved corpus: {processed_corpus.shape}")
    print(f"Output: {output_path}")

    print()
    print("Sample groups:")
    print(
        processed_corpus[
            "sample_group"
        ].value_counts()
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced recipe corpus "
            "for the Meal Plan RAG application."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw recipe CSV.",
    )

    parser.add_argument(
        "--output",
        default="data/processed/recipes_clean.csv",
        help="Path for the processed recipe corpus.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling.",
    )

    arguments = parser.parse_args()

    prepare_corpus(
        input_path=arguments.input,
        output_path=arguments.output,
        random_seed=arguments.seed,
    )


if __name__ == "__main__":
    main()