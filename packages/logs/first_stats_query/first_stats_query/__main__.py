#!/usr/bin/env python

import argparse
import datetime

import polars as pl

# Import to trigger recipe registration
import first_stats_query.visualizations.cluster_stats  # noqa: F401
import first_stats_query.visualizations.inference_overview  # noqa: F401
import first_stats_query.visualizations.institution_stats  # noqa: F401
import first_stats_query.visualizations.status_code_stats  # noqa: F401
import first_stats_query.visualizations.token_breakdown  # noqa: F401

from .tables import STREAMS, generate_table
from .visualizations import get_recipes


def load_data(dataset_dir: str, name: str) -> pl.LazyFrame:
    """Load a dataset by name from the given directory."""
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


def _parse_datetime(s: str) -> datetime.datetime:
    """Parse a UTC timestamp; accepts date-only or date + time."""
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            pass
    raise ValueError(
        f"invalid time: {s!r} (expected YYYY-MM-DD or YYYY-MM-DD HH:MM, UTC)"
    )


def _today_utc() -> datetime.datetime:
    """Start of today in UTC, used as the default window anchor."""
    return datetime.datetime.now(datetime.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def _period_span(period: str) -> datetime.timedelta:
    """Fixed span back from today for a relative period."""
    match period:
        case "6m":
            return datetime.timedelta(days=183)
        case "1m":
            return datetime.timedelta(days=31)
        case _:
            raise ValueError(f"unknown period: {period!r}")


def _resolve_window(
    period: str | None, start: str | None, end: str | None
) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """Resolve an optional period preset to a concrete window, overridable with --start/--end."""
    if period is None or period == "all":
        return (None, None)
    today = _today_utc()
    delta = _period_span(period)
    start_dt = _parse_datetime(start) if start else today - delta
    end_dt = _parse_datetime(end) if end else today
    return (start_dt, end_dt)


def _resolve_window_start_end(
    args: argparse.Namespace,
) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """Resolve the time window from a subcommand's --period/--start/--end options."""
    if args.period:
        return _resolve_window(args.period, args.start, args.end)
    if args.start and args.end:
        return (_parse_datetime(args.start), _parse_datetime(args.end))
    return (None, None)


def _add_window_args(
    parser: argparse.ArgumentParser, *, granularity: bool = False
) -> None:
    """Add shared window selection options to a subparser."""
    parser.add_argument(
        "--period",
        type=str,
        choices=["6m", "1m", "all"],
        default=None,
        help="Relative window preset from today (6m=last 6 months, 1m=last month, "
        "all=no filter); --start/--end override either end of the window",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start of time window (YYYY-MM-DD or YYYY-MM-DD HH:MM, UTC); "
        "overrides the preset start for --period, or pairs with --end for a custom range",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End of time window (YYYY-MM-DD or YYYY-MM-DD HH:MM, UTC); "
        "overrides the preset end for --period, or pairs with --start for a custom range",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=None,
        help="Filter to a single cluster (e.g. minerva)",
    )
    if granularity:
        parser.add_argument(
            "--granularity",
            type=str,
            choices=["daily", "monthly", "yearly", "auto"],
            default=None,
            help="Time bucket for chart (auto uses period default)",
        )


def _run_visualize(args: argparse.Namespace, recipes: dict[str, list]) -> None:
    """Run graph recipes under the given category/recipe choice."""
    if args.help_categories:
        _print_recipes(recipes)
        return

    start, end = _resolve_window_start_end(args)
    recipe_list = recipes[args.category]

    for name, func in recipe_list:
        if args.recipe == "all" or args.recipe == name:
            load_data_func = _make_load_data(args, load_data, start, end)
            granularity = args.granularity or _infer_granularity(start, end)
            func(load_data_func, start, end, granularity, args.cluster)
        if args.recipe == name:
            break


def _run_table(args: argparse.Namespace) -> None:
    """Print rows or column summations from an ingested stream."""
    start, end = _resolve_window_start_end(args)
    columns = args.columns.split(",") if args.columns else None
    aggregate = args.aggregate.split(",") if args.aggregate else None

    def base_load(name: str) -> pl.LazyFrame:
        return load_data(args.dataset_dir, name)

    generate_table(
        base_load,
        args.stream,
        start,
        end,
        args.cluster,
        columns=columns,
        limit=args.limit,
        sort=args.sort,
        aggregate=aggregate,
        group=args.group,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stats query CLI")
    parser.add_argument("dataset_dir", help="Path to the dataset directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    visualize = subparsers.add_parser(
        "visualize", help="Generate graph recipes from a dataset"
    )
    visualize.add_argument(
        "--help-categories",
        action="store_true",
        help="List available categories and recipes",
    )

    recipes = get_recipes()
    viz_cats = visualize.add_subparsers(dest="category")
    for cat, recipe_list in recipes.items():
        cat_parser = viz_cats.add_parser(cat)
        _add_window_args(cat_parser, granularity=True)
        cat_parser.add_argument(
            "recipe",
            nargs="?",
            choices=[r[0] for r in recipe_list] + ["all"],
            default="all",
        )

    table = subparsers.add_parser("table", help="Print rows from an ingested stream")
    _add_window_args(table)
    table.add_argument(
        "stream",
        choices=list(STREAMS),
        help="Stream to tabulate (metrics, request_log, access_log, user)",
    )
    table.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Comma-separated columns to include in the table",
    )
    table.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of rows shown",
    )
    table.add_argument(
        "--sort",
        type=str,
        default=None,
        help="Sort by column; append ':desc' for descending order",
    )
    table.add_argument(
        "--aggregate",
        type=str,
        default=None,
        help="Comma-separated numeric columns to sum instead of listing rows",
    )
    table.add_argument(
        "--group",
        type=str,
        default=None,
        help="Group aggregation rows by this column (requires --aggregate)",
    )

    args = parser.parse_args()

    if args.command == "visualize":
        _run_visualize(args, recipes)
    elif args.command == "table":
        _run_table(args)


def _infer_granularity(
    start: datetime.datetime | None, end: datetime.datetime | None
) -> str:
    """Infer plot granularity from the resolved window span."""
    if start is None or end is None:
        return "auto"  # no time window (e.g. all)
    days = (end - start).days
    if days <= 31:  # up to ~1 month
        return "daily"
    if days <= 366:  # up to ~1 year
        return "monthly"
    return "yearly"


def _make_load_data(args, base_load_data, start, end):
    """Return a load_data function that optionally filters metrics by time/cluster."""
    if start and end:
        time_filter = lambda c: (
            c.str.head(19).str.to_datetime(time_zone="UTC").is_between(start, end)
        )

        if args.cluster:
            cluster_filter = pl.col("cluster").eq(args.cluster)

            def load_data_func(name):
                lf = base_load_data(args.dataset_dir, name)
                if name == "metrics":
                    return lf.filter(
                        time_filter(pl.col("timestamp_compute_request"))
                    ).filter(cluster_filter)
                return lf
        else:

            def load_data_func(name):
                lf = base_load_data(args.dataset_dir, name)
                if name == "metrics":
                    return lf.filter(time_filter(pl.col("timestamp_compute_request")))
                return lf

        return load_data_func
    elif args.cluster:
        cluster_filter = pl.col("cluster").eq(args.cluster)

        def load_data_func(name):
            lf = base_load_data(args.dataset_dir, name)
            if name == "metrics":
                return lf.filter(cluster_filter)
            return lf

        return load_data_func
    else:

        def load_data_func(name):
            return base_load_data(args.dataset_dir, name)

        return load_data_func


def _print_recipes(recipes):
    print("Available categories and recipes:")
    for category_name, recipe_list in recipes.items():
        print(f"\n  {category_name}:")
        for name, _ in recipe_list:
            print(f"    - {name}")
        print("    - all")


if __name__ == "__main__":
    main()
