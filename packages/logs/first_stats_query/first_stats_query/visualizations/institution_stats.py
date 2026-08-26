"""Requests by institution over time."""

import polars as pl

from . import recipe
from .helpers import (
    group_day,
    group_month_year,
    join_request_log,
    join_user,
    plot_stacked_pct,
    window_slug,
)


@recipe("institution")
def institution(load_data, start, end, granularity="monthly", _cluster=None):
    """Requests by institution, adapted to granularity."""
    df = (
        load_data("metrics")
        .pipe(join_request_log, load_data)
        .pipe(join_user, load_data)
        .select("timestamp_compute_request", "username")
    )
    if start is not None and end is not None:
        from .helpers import filter_period

        df = filter_period(df, "timestamp_compute_request", start, end)

    df = df.with_columns(
        pl.col("username")
        .str.split("@")
        .list.get(1, null_on_oob=True)
        .alias("institution"),
    ).filter(pl.col("institution").is_not_null())

    match granularity:
        case "monthly":
            df = df.pipe(group_month_year, "timestamp_compute_request")
            aggregated = (
                df.group_by("institution", "year", "month")
                .agg(pl.col("institution").count().alias("request_count"))
                .sort("year", "month", "institution")
            )
            aggregated = aggregated.with_columns(
                (
                    pl.col("request_count")
                    / pl.col("request_count").sum().over(["year", "month"])
                ).alias("pct_request"),
            )
            aggregated = aggregated.collect(engine="streaming")
            plot_stacked_pct(
                aggregated,
                start,
                end,
                "monthly",
                "institution",
                "request_count",
                "Requests By Institution",
                "Month",
                "% of Requests",
                f"institution_monthly_{window_slug(start, end)}.svg",
            )
        case "daily":
            df = df.pipe(group_day, "timestamp_compute_request")
            aggregated = (
                df.group_by("institution", "day")
                .agg(pl.col("institution").count().alias("request_count"))
                .sort("day", "institution")
            )
            aggregated = aggregated.with_columns(
                (
                    pl.col("request_count")
                    / pl.col("request_count").sum().over(["day"])
                ).alias("pct_request"),
            )
            aggregated = aggregated.collect(engine="streaming")
            plot_stacked_pct(
                aggregated,
                start,
                end,
                "daily",
                "institution",
                "request_count",
                "Requests By Institution",
                "Day of Period",
                "% of Requests",
                f"institution_daily_{window_slug(start, end)}.svg",
            )
        case _:
            aggregated = (
                df.group_by("institution")
                .agg(pl.col("institution").count().alias("request_count"))
                .sort("request_count", descending=True)
            )
            aggregated = aggregated.with_columns(
                (
                    pl.col("request_count") / pl.col("request_count").sum().over([])
                ).alias("pct_request"),
            )
            aggregated = aggregated.collect(engine="streaming")
            plot_stacked_pct(
                aggregated,
                start,
                end,
                "auto",
                "institution",
                "request_count",
                "Requests By Institution",
                "Institution",
                "% of Requests",
                f"institution_auto_{window_slug(start, end)}.svg",
            )
