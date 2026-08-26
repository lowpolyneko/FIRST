"""Token usage and request counts by cluster over time."""

import polars as pl

from . import recipe
from .helpers import (
    group_day,
    group_month_year,
    join_request_log,
    join_user,
    plot_bar,
    window_slug,
)


@recipe("cluster")
def cluster_tokens(load_data, start, end, granularity="monthly", _cluster=None):
    """Total tokens by cluster, adapted to granularity."""
    df = load_data("metrics").select(
        "cluster", "total_tokens", "timestamp_compute_request"
    )
    if start is not None and end is not None:
        from .helpers import filter_period

        df = filter_period(df, "timestamp_compute_request", start, end)
    match granularity:
        case "monthly":
            df = df.pipe(group_month_year, "timestamp_compute_request")
            result = (
                df.group_by("cluster", "year", "month")
                .agg(pl.col("total_tokens").sum())
                .sort("year", "month", "cluster")
                .collect(engine="streaming")
            )
            plot_bar(
                result,
                start,
                end,
                "monthly",
                "total_tokens",
                "cluster",
                "Total Token Usage By Cluster",
                "Month",
                "# of Total Tokens",
                f"tokens_monthly_{window_slug(start, end)}.svg",
            )
        case "daily":
            df = df.pipe(group_day, "timestamp_compute_request")
            result = (
                df.group_by("cluster", "day")
                .agg(pl.col("total_tokens").sum())
                .sort("day", "cluster")
                .collect(engine="streaming")
            )
            plot_bar(
                result,
                start,
                end,
                "daily",
                "total_tokens",
                "cluster",
                "Total Token Usage By Cluster",
                "Day of Period",
                "# of Total Tokens",
                f"tokens_daily_{window_slug(start, end)}.svg",
            )
        case _:
            result = (
                df.group_by("cluster")
                .agg(pl.col("total_tokens").sum())
                .sort("total_tokens", descending=True)
                .collect(engine="streaming")
            )
            plot_bar(
                result,
                start,
                end,
                "auto",
                "total_tokens",
                "cluster",
                "Total Token Usage By Cluster",
                "Cluster",
                "# of Total Tokens",
                f"tokens_auto_{window_slug(start, end)}.svg",
            )


@recipe("cluster")
def cluster_requests(load_data, start, end, granularity="monthly", _cluster=None):
    """Request count by cluster, adapted to granularity."""
    df = load_data("metrics").select("cluster", "timestamp_compute_request")
    if start is not None and end is not None:
        from .helpers import filter_period

        df = filter_period(df, "timestamp_compute_request", start, end)
    match granularity:
        case "monthly":
            df = df.pipe(group_month_year, "timestamp_compute_request")
            result = (
                df.group_by("cluster", "year", "month")
                .agg(pl.col("cluster").count().alias("request_count"))
                .sort("year", "month", "cluster")
                .collect(engine="streaming")
            )
            plot_bar(
                result,
                start,
                end,
                "monthly",
                "request_count",
                "cluster",
                "Request Count By Cluster",
                "Month",
                "# of Requests",
                f"requests_monthly_{window_slug(start, end)}.svg",
            )
        case "daily":
            df = df.pipe(group_day, "timestamp_compute_request")
            result = (
                df.group_by("cluster", "day")
                .agg(pl.col("cluster").count().alias("request_count"))
                .sort("day", "cluster")
                .collect(engine="streaming")
            )
            plot_bar(
                result,
                start,
                end,
                "daily",
                "request_count",
                "cluster",
                "Request Count By Cluster",
                "Day of Period",
                "# of Requests",
                f"requests_daily_{window_slug(start, end)}.svg",
            )
        case _:
            result = (
                df.group_by("cluster")
                .agg(pl.col("cluster").count().alias("request_count"))
                .sort("request_count", descending=True)
                .collect(engine="streaming")
            )
            plot_bar(
                result,
                start,
                end,
                "auto",
                "request_count",
                "cluster",
                "Request Count By Cluster",
                "Cluster",
                "# of Requests",
                f"requests_auto_{window_slug(start, end)}.svg",
            )


@recipe("cluster")
def cluster_users(load_data, start, end, granularity="monthly", _cluster=None):
    """Unique users by cluster, adapted to granularity."""
    df = (
        load_data("metrics")
        .pipe(join_request_log, load_data)
        .pipe(join_user, load_data)
        .select("cluster", "user.name", "timestamp_compute_request")
    )
    if start is not None and end is not None:
        from .helpers import filter_period

        df = filter_period(df, "timestamp_compute_request", start, end)
    match granularity:
        case "monthly":
            df = df.pipe(group_month_year, "timestamp_compute_request").unique(
                subset=["cluster", "user.name", "year", "month"]
            )
            result = (
                df.group_by("cluster", "year", "month")
                .agg(pl.col("user.name").count().alias("user_count"))
                .sort("year", "month", "cluster")
                .collect(engine="streaming")
            )
            plot_bar(
                result,
                start,
                end,
                "monthly",
                "user_count",
                "cluster",
                "Unique Users By Cluster",
                "Month",
                "# of Users",
                f"users_monthly_{window_slug(start, end)}.svg",
            )
        case "daily":
            df = df.pipe(group_day, "timestamp_compute_request").unique(
                subset=["cluster", "user.name", "day"]
            )
            result = (
                df.group_by("cluster", "day")
                .agg(pl.col("user.name").count().alias("user_count"))
                .sort("day", "cluster")
                .collect(engine="streaming")
            )
            plot_bar(
                result,
                start,
                end,
                "daily",
                "user_count",
                "cluster",
                "Unique Users By Cluster",
                "Day of Period",
                "# of Users",
                f"users_daily_{window_slug(start, end)}.svg",
            )
        case _:
            result = (
                df.unique(subset=["cluster", "user.name"])
                .group_by("cluster")
                .agg(pl.col("user.name").count().alias("user_count"))
                .sort("user_count", descending=True)
                .collect(engine="streaming")
            )
            plot_bar(
                result,
                start,
                end,
                "auto",
                "user_count",
                "cluster",
                "Unique Users By Cluster",
                "Cluster",
                "# of Users",
                f"users_auto_{window_slug(start, end)}.svg",
            )
