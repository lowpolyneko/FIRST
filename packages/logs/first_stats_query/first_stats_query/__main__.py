#!/usr/bin/env python

import argparse
import datetime

# Import to trigger recipe registration
import first_stats_query.visualizations.cluster_stats  # noqa: F401
import first_stats_query.visualizations.inference_overview  # noqa: F401
import first_stats_query.visualizations.institution_stats  # noqa: F401
import first_stats_query.visualizations.status_code_stats  # noqa: F401
import first_stats_query.visualizations.token_breakdown  # noqa: F401

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stats query CLI")
    parser.add_argument("dataset_dir", help="Path to the dataset directory")
    parser.add_argument(
        "--help-categories",
        action="store_true",
        help="List available categories and recipes",
    )
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
        "--granularity",
        type=str,
        choices=["daily", "monthly", "yearly", "auto"],
        default=None,
        help="Time bucket for chart (auto uses period default)",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=None,
        help="Filter to a single cluster (e.g. minerva)",
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

    # Resolve a span preset to a concrete window (or parse custom dates)
    start: datetime.datetime | None = None
    end: datetime.datetime | None = None

    if args.period:
        start, end = _resolve_window(args.period, args.start, args.end)
    elif args.start and args.end:
        start = _parse_datetime(args.start)
        end = _parse_datetime(args.end)

    recipe_list = recipes[args.category]

    for name, func in recipe_list:
        if args.recipe == "all" or args.recipe == name:
            load_data_func = _make_load_data(args, load_data, start, end)
            granularity = args.granularity or _infer_granularity(start, end)
            func(load_data_func, start, end, granularity, args.cluster)
        if args.recipe == name:
            break


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
    import polars as pl

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
