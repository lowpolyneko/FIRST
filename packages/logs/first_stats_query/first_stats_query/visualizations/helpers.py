"""Scope-aware helpers for visualization recipes.

Encapsulates granularity-aware plotting so recipe bodies stay simple.
Granularity (daily/monthly/yearly/auto) controls the time bucket and
axis labels; the period (6m/1m/all) controls the data filter.
"""

import datetime

import holoviews as hv
import polars as pl

# ── Data helpers ────────────────────────────────────────────────────────


def filter_period(
    df: pl.LazyFrame, time_col: str, start: datetime.datetime, end: datetime.datetime
) -> pl.LazyFrame:
    """Parse timestamp and filter to a date range."""
    return df.with_columns(
        pl.col(time_col).str.head(19).str.to_datetime(time_zone="UTC"),
    ).filter(pl.col(time_col).is_between(start, end))


def join_request_log(load_data, df: pl.LazyFrame) -> pl.LazyFrame:
    """Join metrics with request_log on request_id."""
    return df.join(
        load_data("request_log"),
        left_on="request_id",
        right_on="id",
        suffix="_rl",
    )


def join_user(load_data, df: pl.LazyFrame) -> pl.LazyFrame:
    """Join with user table (deduplicated by id)."""
    return df.join(
        load_data("user").unique(subset="id"),
        left_on="user_id",
        right_on="id",
        suffix="_u",
    )


def group_day(df: pl.LazyFrame, time_col: str) -> pl.LazyFrame:
    return df.with_columns(pl.col(time_col).dt.day().alias("day"))


def group_month_year(df: pl.LazyFrame, time_col: str) -> pl.LazyFrame:
    return df.with_columns(
        pl.col(time_col).dt.month().alias("month"),
        pl.col(time_col).dt.year().alias("year"),
    )


# ── Granularity-aware plotting helpers ──────────────────────────────────
# Granularity options: "daily", "monthly", "yearly", "auto"
#   - auto = no time dimension (single aggregated bar chart)


def window_label(start: datetime.datetime | None, end: datetime.datetime | None) -> str:
    """Human-readable window label, computed from the window itself."""
    if start is None or end is None:
        return "All-time"
    return f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"


def window_slug(start: datetime.datetime | None, end: datetime.datetime | None) -> str:
    """Filesystem-safe identifier derived from the window (for output filenames)."""
    if start is None or end is None:
        return "all"
    return f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"


def plot_bar(
    df, start, end, granularity, y_col, by_col, title, xlabel, ylabel, save_path
):
    """Grouped bar chart, adapted to granularity."""
    plot, ax_setup = _plot_info(granularity)
    agg_fn = pl.col(y_col).sum()

    if granularity == "auto":
        result = df.group_by(by_col).agg(agg_fn).sort(y_col, descending=True)
    else:
        result = (
            df.group_by(by_col, plot["group_col"])
            .agg(agg_fn)
            .sort([plot["group_col"], by_col])
        )

    plot_obj = result.hvplot.bar(
        x=plot["x_col"],
        y=y_col,
        by=by_col,
        title=f"{title} ({window_label(start, end)})",
        xlabel=plot["xlabel"] or xlabel,
        ylabel=ylabel,
        aspect="square",
        stacked=False,
        width=800,
    )
    _save_plot(plot_obj, save_path, plot["xticks"], plot["xtick_labels"])


def plot_stacked_pct(
    df, start, end, granularity, by_col, y_col, title, xlabel, ylabel, save_path
):
    """Stacked bar chart with percentage, adapted to granularity."""
    plot, ax_setup = _plot_info(granularity)
    group_cols = [by_col, plot["group_col"]] if granularity != "auto" else [by_col]

    result = (
        df.group_by(*group_cols)
        .agg(pl.col(y_col).count().alias("count"))
        .sort(group_cols)
    )
    result = result.with_columns(
        (pl.col("count") / pl.col("count").sum().over(group_cols)).alias("pct"),
    )

    plot_obj = result.hvplot.bar(
        x=plot["x_col"],
        y="pct",
        by=by_col,
        title=f"{title} ({window_label(start, end)})",
        xlabel=plot["xlabel"] or xlabel,
        ylabel=ylabel,
        aspect="square",
        stacked=True,
    )
    _save_plot(plot_obj, save_path, plot["xticks"], plot["xtick_labels"])


def _plot_info(granularity):
    """Return x-axis config for a granularity."""
    if granularity == "daily":
        return {
            "group_col": "day",
            "x_col": "day",
            "xlabel": "Day of Period",
            "xticks": list(range(1, 32)),
            "xtick_labels": [str(d) for d in range(1, 32)],
        }
    elif granularity == "monthly":
        return {
            "group_col": "month_year",
            "x_col": "month",
            "xlabel": "Month",
            "xticks": list(range(6)),
            "xtick_labels": ["Mar", "Apr", "May", "Jun", "Jul", "Aug"],
        }
    elif granularity == "yearly":
        return {
            "group_col": "year",
            "x_col": "year",
            "xlabel": "Year",
            "xticks": [],
            "xtick_labels": [],
        }
    else:  # auto
        return {
            "group_col": None,
            "x_col": None,
            "xlabel": None,
            "xticks": [],
            "xtick_labels": [],
        }


def _save_plot(plot, save_path, xticks, xtick_labels):
    renderer = hv.renderer("matplotlib")
    plot_state = renderer.get_plot(plot)
    fig = plot_state.state
    ax = fig.gca()
    if xticks:
        ax.set_xticks(xticks)
        if xtick_labels:
            ax.set_xticklabels(xtick_labels)
    fig.tight_layout()
    fig.savefig(save_path)
