"""Requests by status code over time."""

import polars as pl

from . import recipe
from .helpers import group_day, group_month_year, plot_stacked_pct, window_slug


@recipe("status")
def status(load_data, start, end, granularity="monthly", _cluster=None):
    """Requests by status code, adapted to granularity."""
    df = load_data("metrics").select("status_code", "timestamp_compute_request")
    if start is not None and end is not None:
        from .helpers import filter_period

        df = filter_period(df, "timestamp_compute_request", start, end)

    df = df.with_columns(
        pl.col("status_code").cast(pl.Utf8).fill_null("null").alias("status_code"),
    ).filter(pl.col("status_code").is_not_null())

    match granularity:
        case "monthly":
            df = df.pipe(group_month_year, "timestamp_compute_request")
            aggregated = (
                df.group_by("status_code", "year", "month")
                .agg(pl.col("status_code").count().alias("request_count"))
                .sort("year", "month", "status_code")
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
                "status_code",
                "request_count",
                "Requests By Status",
                "Month",
                "% of Requests",
                f"status_monthly_{window_slug(start, end)}.svg",
            )
        case "daily":
            df = df.pipe(group_day, "timestamp_compute_request")
            aggregated = (
                df.group_by("status_code", "day")
                .agg(pl.col("status_code").count().alias("request_count"))
                .sort("day", "status_code")
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
                "status_code",
                "request_count",
                "Requests By Status",
                "Day of Period",
                "% of Requests",
                f"status_daily_{window_slug(start, end)}.svg",
            )
        case _:
            aggregated = (
                df.group_by("status_code")
                .agg(pl.col("status_code").count().alias("request_count"))
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
                "status_code",
                "request_count",
                "Requests By Status",
                "Status Code",
                "% of Requests",
                f"status_auto_{window_slug(start, end)}.svg",
            )
