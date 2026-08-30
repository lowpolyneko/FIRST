"""Scope-aware helpers for visualization recipes.

Encapsulates granularity-aware plotting so recipe bodies stay simple.
Granularity (daily/monthly/yearly/auto) controls the time bucket and
axis labels; the period (6m/1m/all) controls the data filter.
"""

import datetime
from collections.abc import Callable
from typing import Any

import holoviews as hv  # type: ignore[import-untyped]
import hvplot.polars  # type: ignore  # noqa: F401 (registers DataFrame.hvplot accessor)
import polars as pl

# Loads one ingested stream; the signature every recipe is called with.
LoadData = Callable[[str], pl.LazyFrame]
Recipe = Callable[
    [LoadData, datetime.datetime | None, datetime.datetime | None, str, str | None],
    None,
]

# ── Data helpers ────────────────────────────────────────────────────────


def parse_time(df: pl.LazyFrame, time_col: str) -> pl.LazyFrame:
    """Return ``df`` with ``time_col`` converted to UTC datetimes.

    Ingested logs carry timestamps as ISO-8601 strings, so they must be
    parsed before any date part or axis is taken from them.
    """
    if df.collect_schema()[time_col] == pl.String:
        df = df.with_columns(
            pl.col(time_col).str.head(19).str.to_datetime(time_zone="UTC")
        )
    return df


def filter_period(
    df: pl.LazyFrame,
    time_col: str,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
) -> pl.LazyFrame:
    """Parse ``time_col`` and keep rows no earlier than ``start``, no later than ``end``."""
    df = parse_time(df, time_col)
    if start is not None:
        df = df.filter(pl.col(time_col) >= start)
    if end is not None:
        df = df.filter(pl.col(time_col) <= end)
    return df


def join_request_log(df: pl.LazyFrame, load_data: LoadData) -> pl.LazyFrame:
    """Join metrics with request_log on request_id, keeping unmatched requests.

    A metrics row whose request_log row is missing from this dataset still
    counts toward request totals, so it must not be dropped by the join.
    """
    return df.join(
        load_data("request_log"),
        left_on="request_id",
        right_on="id",
        how="left",
        suffix="_rl",
    )


def join_user(df: pl.LazyFrame, load_data: LoadData) -> pl.LazyFrame:
    """Join with user table (deduplicated by id), keeping requests with no user row.

    Users authenticate once per session, so a dataset can be missing the user
    row of a request it does hold metrics for.
    """
    return df.join(
        load_data("user").unique(subset="id"),
        left_on="user_id",
        right_on="id",
        how="left",
        suffix="_u",
    )


def group_day(df: pl.LazyFrame, time_col: str) -> pl.LazyFrame:
    """Bucket rows by calendar date in the ``day`` column."""
    return parse_time(df, time_col).with_columns(
        pl.col(time_col).dt.strftime("%Y-%m-%d").alias("day")
    )


def group_month_year(df: pl.LazyFrame, time_col: str) -> pl.LazyFrame:
    """Bucket rows by calendar month, labelled with its year in ``month_year``."""
    return parse_time(df, time_col).with_columns(
        pl.col(time_col).dt.month().alias("month"),
        pl.col(time_col).dt.year().alias("year"),
        pl.col(time_col).dt.strftime("%Y-%m").alias("month_year"),
    )


def group_year(df: pl.LazyFrame, time_col: str) -> pl.LazyFrame:
    """Bucket rows by calendar year in the ``year`` column."""
    return parse_time(df, time_col).with_columns(
        pl.col(time_col).dt.strftime("%Y").alias("year")
    )


def group_time(
    df: pl.LazyFrame, time_col: str, granularity: str
) -> tuple[pl.LazyFrame, list[str]]:
    """Add the time bucket columns for ``granularity`` and name them.

    The bucket list is empty for ``auto``, which plots without a time axis.
    """
    match granularity:
        case "daily":
            return group_day(df, time_col), ["day"]
        case "monthly":
            return group_month_year(df, time_col), ["month_year"]
        case "yearly":
            return group_year(df, time_col), ["year"]
        case _:
            return df, []


def add_percent(
    aggregated: pl.LazyFrame, buckets: list[str], value: str
) -> pl.LazyFrame:
    """Add a ``pct_request`` share-of-total column, per time bucket when bucketed."""
    total = pl.col(value).sum().over(buckets) if buckets else pl.col(value).sum()
    return aggregated.with_columns((pl.col(value) / total).alias("pct_request"))


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
    df: pl.DataFrame,
    start: datetime.datetime | None,
    end: datetime.datetime | None,
    granularity: str,
    y_col: str,
    by_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
) -> None:
    """Grouped bar chart, adapted to granularity."""
    if no_rows(df, save_path):
        return
    axis = plot_info(granularity)
    title_text = f"{title} ({window_label(start, end)})"

    if granularity == "auto":
        plot_obj = _hvplot(df).bar(
            x=by_col,
            y=y_col,
            title=title_text,
            xlabel=axis["xlabel"] or xlabel,
            ylabel=ylabel,
            aspect="square",
        )
    else:
        plot_obj = _hvplot(df).bar(
            x=axis["x_col"],
            y=y_col,
            by=by_col,
            title=title_text,
            xlabel=axis["xlabel"] or xlabel,
            ylabel=ylabel,
            aspect="square",
            stacked=False,
            width=800,
        )
    save_plot(plot_obj, save_path)


def plot_stacked_pct(
    df: pl.DataFrame,
    start: datetime.datetime | None,
    end: datetime.datetime | None,
    granularity: str,
    by_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
) -> None:
    """Stacked bar chart with percentage, adapted to granularity."""
    if no_rows(df, save_path):
        return
    axis = plot_info(granularity)
    title_text = f"{title} ({window_label(start, end)})"

    if granularity == "auto":
        plot_obj = _hvplot(df).bar(
            x=by_col,
            y="pct_request",
            title=title_text,
            xlabel=axis["xlabel"] or xlabel,
            ylabel=ylabel,
            aspect="square",
            stacked=True,
        )
    else:
        plot_obj = _hvplot(df).bar(
            x=axis["x_col"],
            y="pct_request",
            by=by_col,
            title=title_text,
            xlabel=axis["xlabel"] or xlabel,
            ylabel=ylabel,
            aspect="square",
            stacked=True,
        )
    save_plot(plot_obj, save_path)


def plot_barh(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
) -> None:
    """Horizontal bar chart, one bar per value of ``x_col``."""
    if no_rows(df, save_path):
        return
    plot_obj = _hvplot(df).barh(
        x=x_col,
        y=y_col,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        aspect="square",
    )
    save_plot(plot_obj, save_path)


def plot_hist(
    df: pl.DataFrame,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
    **kwargs: Any,
) -> None:
    """Histogram of the values in ``y_col``, binned and labelled by ``kwargs``."""
    if no_rows(df, save_path):
        return
    plot_obj = _hvplot(df).hist(
        y=y_col,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        aspect="square",
        **kwargs,
    )
    save_plot(plot_obj, save_path)


def plot_stacked(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    by_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
) -> None:
    """Stacked bar chart of ``y_col`` split by ``by_col`` over ``x_col``."""
    if no_rows(df, save_path):
        return
    plot_obj = _hvplot(df).bar(
        x=x_col,
        y=y_col,
        by=by_col,
        stacked=True,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        aspect="square",
        width=800,
    )
    save_plot(plot_obj, save_path)


def no_rows(df: pl.DataFrame, save_path: str) -> bool:
    """Report a selection with no rows to plot, so callers can skip it."""
    if df.is_empty():
        print(f"{save_path}: no rows in the selected window, skipped")
        return True
    return False


def save_plot(plot: Any, save_path: str) -> None:
    """Render ``plot`` to ``save_path``."""
    fig = hv.renderer("matplotlib").get_plot(plot).state
    fig.tight_layout()
    fig.savefig(save_path)


def _hvplot(df: pl.DataFrame) -> Any:
    """The hvplot accessor, which mypy cannot see on ``pl.DataFrame``."""
    return df.hvplot  # type: ignore[attr-defined]


def plot_info(granularity: str) -> dict[str, Any]:
    """Return the x-axis config for a granularity.

    Bucket columns are ISO-formatted strings, so the categorical axis labels
    them in order and needs no explicit ticks.
    """
    match granularity:
        case "daily":
            return {"x_col": "day", "xlabel": "Date"}
        case "monthly":
            return {"x_col": "month_year", "xlabel": "Month"}
        case "yearly":
            return {"x_col": "year", "xlabel": "Year"}
        case _:  # auto
            return {"x_col": None, "xlabel": None}
