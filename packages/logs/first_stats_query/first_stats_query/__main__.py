#!/usr/bin/env python

import argparse

# Import to trigger recipe registration
import first_stats_query.visualizations.cluster_six_month_stats  # noqa: F401
import first_stats_query.visualizations.cluster_token_breakdown  # noqa: F401
import first_stats_query.visualizations.inference_stats  # noqa: F401
import first_stats_query.visualizations.minerva_aug6_stats  # noqa: F401
import first_stats_query.visualizations.requests_by_institution  # noqa: F401
import first_stats_query.visualizations.status_requests_stats  # noqa: F401

from .visualizations import get_recipes


def load_data(dataset_dir: str, name: str):
    """Load a dataset by name from the given directory."""
    import polars as pl

    match name:
        case "metrics":
            return pl.scan_parquet(
                f"{dataset_dir}/*.request_metrics.parquet",
                missing_columns="insert",
                extra_columns="ignore",
            )
        case "request_log":
            return pl.scan_parquet(
                f"{dataset_dir}/*.request_log.parquet",
                missing_columns="insert",
                extra_columns="ignore",
            )
        case "access_log":
            return pl.scan_parquet(
                f"{dataset_dir}/*.access_log.parquet",
                missing_columns="insert",
                extra_columns="ignore",
            )
        case "user":
            return pl.scan_parquet(f"{dataset_dir}/*.user.parquet")
        case _:
            raise ValueError(f"unknown data source: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stats query CLI")
    parser.add_argument("dataset_dir", help="Path to the dataset directory")
    parser.add_argument(
        "--help-categories",
        action="store_true",
        help="List available categories and recipes",
    )

    recipes = get_recipes()
    subparsers = parser.add_subparsers(dest="category")

    for cat, recipe_list in recipes.items():
        cat_parser = subparsers.add_parser(cat)
        cat_parser.add_argument(
            "recipe",
            nargs="?",
            choices=[r[0] for r in recipe_list] + ["all"],
            default="all",
        )

    args = parser.parse_args()

    if args.help_categories:
        _print_recipes(recipes)
        return

    if args.category is None:
        parser.print_help()
        return

    recipe_list = recipes[args.category]

    for name, func in recipe_list:
        if args.recipe == "all" or args.recipe == name:
            load_data_func = lambda n: load_data(args.dataset_dir, n)
            func(load_data_func)
        if args.recipe != "all":
            break


def _print_recipes(recipes):
    print("Available categories and recipes:")
    for category_name, recipe_list in recipes.items():
        print(f"\n  {category_name}:")
        for name, _ in recipe_list:
            print(f"    - {name}")
        print("    - all")


if __name__ == "__main__":
    main()
