"""Token usage breakdown by cluster over time."""

import polars as pl

from . import recipe
from .helpers import group_day, group_month_year, window_label, window_slug


@recipe("token-breakdown")
def token_breakdown(load_data, start, end, granularity="monthly", _cluster=None):
    """Prompt vs completion tokens by cluster, adapted to granularity."""
    df = load_data("metrics").select(
        "cluster", "prompt_tokens", "completion_tokens", "timestamp_compute_request"
    )
    if start is not None and end is not None:
        from .helpers import filter_period

        df = filter_period(df, "timestamp_compute_request", start, end)

    df = df.with_columns(
        pl.col("prompt_tokens").fill_null(0),
        pl.col("completion_tokens").fill_null(0),
    )

    match granularity:
        case "monthly":
            df = df.pipe(group_month_year, "timestamp_compute_request")
            aggregated = (
                df.group_by("cluster", "year", "month")
                .agg(
                    pl.col("prompt_tokens").sum().alias("prompt_tokens"),
                    pl.col("completion_tokens").sum().alias("completion_tokens"),
                )
                .sort("year", "month", "cluster")
                .collect(engine="streaming")
            )
            _plot_token_monthly(
                aggregated, start, end, f"monthly_{window_slug(start, end)}"
            )
        case "daily":
            df = df.pipe(group_day, "timestamp_compute_request")
            aggregated = (
                df.group_by("cluster", "day")
                .agg(
                    pl.col("prompt_tokens").sum().alias("prompt_tokens"),
                    pl.col("completion_tokens").sum().alias("completion_tokens"),
                )
                .sort("day", "cluster")
                .collect(engine="streaming")
            )
            _plot_token_daily(
                aggregated, start, end, f"daily_{window_slug(start, end)}"
            )
        case _:
            aggregated = (
                df.group_by("cluster")
                .agg(
                    pl.col("prompt_tokens").sum().alias("prompt_tokens"),
                    pl.col("completion_tokens").sum().alias("completion_tokens"),
                )
                .sort("cluster")
                .collect(engine="streaming")
            )
            _plot_token_auto(aggregated, start, end, f"auto_{window_slug(start, end)}")


def _plot_token_monthly(df: pl.DataFrame, start, end, suffix: str):
    """Unpivot and plot per-cluster stacked token bar charts (monthly)."""
    import holoviews as hv

    melted = df.unpivot(
        index=["cluster", "year", "month"],
        on=["prompt_tokens", "completion_tokens"],
        variable_name="token_type",
        value_name="tokens",
    )
    for cluster in melted.select("cluster").unique().to_series().to_list():
        plot_df = melted.filter(pl.col("cluster") == cluster)
        plot = plot_df.hvplot.bar(
            x="month",
            y="tokens",
            by="token_type",
            stacked=True,
            title=f"Token Usage Breakdown By Type - {cluster} ({window_label(start, end)})",
            xlabel="Month",
            ylabel="# of Tokens",
            aspect="square",
            width=800,
        )
        renderer = hv.renderer("matplotlib")
        plot_state = renderer.get_plot(plot)
        fig = plot_state.state
        fig.tight_layout()
        fig.savefig(f"token_breakdown_{cluster}_monthly_{suffix}.svg")


def _plot_token_daily(df: pl.DataFrame, start, end, suffix: str):
    """Unpivot and plot per-cluster stacked token bar charts (daily)."""
    import holoviews as hv

    melted = df.unpivot(
        index=["cluster", "day"],
        on=["prompt_tokens", "completion_tokens"],
        variable_name="token_type",
        value_name="tokens",
    )
    for cluster in melted.select("cluster").unique().to_series().to_list():
        plot_df = melted.filter(pl.col("cluster") == cluster)
        plot = plot_df.hvplot.bar(
            x="day",
            y="tokens",
            by="token_type",
            stacked=True,
            title=f"Token Usage Breakdown By Type - {cluster} ({window_label(start, end)})",
            xlabel="Day of Period",
            ylabel="# of Tokens",
            aspect="square",
            width=800,
        )
        renderer = hv.renderer("matplotlib")
        plot_state = renderer.get_plot(plot)
        fig = plot_state.state
        fig.tight_layout()
        fig.savefig(f"token_breakdown_{cluster}_daily_{suffix}.svg")


def _plot_token_auto(df: pl.DataFrame, start, end, suffix: str):
    """Unpivot and plot per-cluster stacked token bar charts (auto)."""
    import holoviews as hv

    melted = df.unpivot(
        index=["cluster"],
        on=["prompt_tokens", "completion_tokens"],
        variable_name="token_type",
        value_name="tokens",
    )
    for cluster in melted.select("cluster").unique().to_series().to_list():
        plot_df = melted.filter(pl.col("cluster") == cluster)
        plot = plot_df.hvplot.bar(
            x="cluster",
            y="tokens",
            by="token_type",
            stacked=True,
            title=f"Token Usage Breakdown By Type - {cluster} ({window_label(start, end)})",
            xlabel="Cluster",
            ylabel="# of Tokens",
            aspect="square",
            width=800,
        )
        renderer = hv.renderer("matplotlib")
        plot_state = renderer.get_plot(plot)
        fig = plot_state.state
        fig.tight_layout()
        fig.savefig(f"token_breakdown_{cluster}_auto_{suffix}.svg")
