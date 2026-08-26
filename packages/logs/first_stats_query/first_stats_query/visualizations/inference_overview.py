"""Inference stats: top users, models, user distributions, request histograms."""

import datetime

import holoviews as hv
import polars as pl
from matplotlib.dates import DateFormatter
from matplotlib.ticker import StrMethodFormatter

from . import recipe
from .helpers import join_request_log, join_user, window_label, window_slug


@recipe("inference")
def top_users(load_data, start, end, _granularity="auto", _cluster=None):  # noqa: ARG001
    """Top 20 users by total tokens for the period."""
    result = (
        load_data("metrics")
        .select("user.name", "total_tokens")
        .group_by("user.name")
        .agg(pl.col("total_tokens").sum())
        .sort("total_tokens", descending=True)
        .limit(20)
        .collect(engine="streaming")
    )
    plot = result.hvplot.barh(
        x="user.name",
        y="total_tokens",
        title=f"Total Token Usage By User (Top 20, {window_label(start, end)})",
        xlabel="Name",
        ylabel="# of Total Tokens",
        aspect="square",
    )
    renderer = hv.renderer("matplotlib")
    plot_state = renderer.get_plot(plot)
    fig = plot_state.state
    fig.tight_layout()
    fig.savefig(f"top_users_{window_slug(start, end)}.svg")


@recipe("inference")
def top_models(load_data, start, end, _granularity="auto", _cluster=None):  # noqa: ARG001
    """Top 20 models by total tokens for the period."""
    result = (
        load_data("metrics")
        .select("model", "total_tokens")
        .group_by("model")
        .agg(pl.col("total_tokens").sum())
        .sort("total_tokens", descending=True)
        .limit(20)
        .collect(engine="streaming")
    )
    plot = result.hvplot.barh(
        x="model",
        y="total_tokens",
        title=f"Total Token Usage By Model (Top 20, {window_label(start, end)})",
        xlabel="Model Name",
        ylabel="# of Total Tokens",
        aspect="square",
    )
    renderer = hv.renderer("matplotlib")
    plot_state = renderer.get_plot(plot)
    fig = plot_state.state
    fig.tight_layout()
    fig.savefig(f"top_models_{window_slug(start, end)}.svg")


@recipe("inference")
def top_models_requests(load_data, start, end, _granularity="auto", _cluster=None):  # noqa: ARG001
    """Top 20 models by request count for the period."""
    result = (
        load_data("metrics")
        .group_by("model")
        .agg(pl.col("model").count())
        .sort("model", descending=True)
        .limit(20)
        .collect(engine="streaming")
    )
    plot = result.hvplot.barh(
        x="model",
        y="model_count",
        title=f"Model Usage By Request (Top 20, {window_label(start, end)})",
        xlabel="Model Name",
        ylabel="# of Requests",
        aspect="square",
    )
    renderer = hv.renderer("matplotlib")
    plot_state = renderer.get_plot(plot)
    fig = plot_state.state
    fig.tight_layout()
    fig.savefig(f"top_models_requests_{window_slug(start, end)}.svg")


@recipe("inference")
def top_models_users(load_data, start, end, _granularity="auto", _cluster=None):  # noqa: ARG001
    """Top 20 models by unique users for the period."""
    result = (
        load_data("metrics")
        .pipe(join_request_log, load_data)
        .pipe(join_user, load_data)
        .select("user.name", "model")
        .unique(subset=["model", "user.name"])
        .group_by("model")
        .agg(pl.col("user.name").count().alias("user_count"))
        .sort("user_count", descending=True)
        .limit(20)
        .collect(engine="streaming")
    )
    plot = result.hvplot.barh(
        x="model",
        y="user_count",
        title=f"Model Usage By Unique Users (Top 20, {window_label(start, end)})",
        xlabel="Model",
        ylabel="# of Users",
        aspect="square",
    )
    renderer = hv.renderer("matplotlib")
    plot_state = renderer.get_plot(plot)
    fig = plot_state.state
    fig.tight_layout()
    fig.savefig(f"top_models_users_{window_slug(start, end)}.svg")


@recipe("inference")
def hist_users(load_data, start, end, _granularity="auto", _cluster=None):  # noqa: ARG001
    """User distribution histogram for the period."""
    df = (
        load_data("access_log")
        .pipe(join_user, load_data)
        .select("user.name", "timestamp_request")
        .collect(engine="streaming")
    )
    kwargs = {
        "y": "timestamp_request",
        "title": f"User Distribution ({window_label(start, end)})",
        "xlabel": "Timestamp",
        "ylabel": "Number of Users",
        "aspect": "square",
    }
    kwargs.update(_hist_setup(start, end))
    plot = df.hvplot.hist(**kwargs)
    renderer = hv.renderer("matplotlib")
    plot_state = renderer.get_plot(plot)
    fig = plot_state.state
    fig.tight_layout()
    fig.savefig(f"hist_users_{window_slug(start, end)}.svg")


@recipe("inference")
def hist_requests(load_data, start, end, _granularity="auto", _cluster=None):  # noqa: ARG001
    """Request distribution histogram for the period."""
    df = (
        load_data("metrics")
        .select("timestamp_compute_request")
        .collect(engine="streaming")
    )
    kwargs = {
        "y": "timestamp_compute_request",
        "title": f"Request Distribution ({window_label(start, end)})",
        "xlabel": "Timestamp",
        "ylabel": "Number of Requests",
        "aspect": "square",
    }
    kwargs.update(_hist_setup(start, end))
    plot = df.hvplot.hist(**kwargs)
    renderer = hv.renderer("matplotlib")
    plot_state = renderer.get_plot(plot)
    fig = plot_state.state
    fig.tight_layout()
    fig.savefig(f"hist_requests_{window_slug(start, end)}.svg")


def _hist_setup(start: datetime.datetime | None, end: datetime.datetime | None) -> dict:
    """Histogram binning/x-axis config derived from the window span."""
    if start is None or end is None:
        return {
            "bins": 12,
            "xformatter": DateFormatter("%m/%y"),
            "yformatter": StrMethodFormatter("{x:,.0f}"),
        }
    span_days = (end - start).days
    if span_days <= 31:  # ~1 month -> daily bins
        return {
            "bins": 31,
            "xformatter": DateFormatter("%d"),
            "xticks": _hist_xticks_daily(start, end),
            "yformatter": StrMethodFormatter("{x:,.0f}"),
        }
    if span_days <= 183:  # ~6 months -> monthly bins
        return {
            "bins": 6,
            "xformatter": DateFormatter("%b"),
            "xticks": _hist_xticks_monthly(start, end),
        }
    return {
        "bins": 12,
        "xformatter": DateFormatter("%m/%y"),
        "yformatter": StrMethodFormatter("{x:,.0f}"),
    }


def _hist_xticks_monthly(
    start: datetime.datetime, end: datetime.datetime
) -> list[datetime.datetime]:
    """Mid-month tick dates spanning the window."""
    ticks = []
    day = start.replace(day=15)
    while day <= end:
        ticks.append(day)
        day = (day + datetime.timedelta(days=30)).replace(day=15)
    return ticks


def _hist_xticks_daily(
    start: datetime.datetime, end: datetime.datetime
) -> list[datetime.datetime]:
    """Tick dates every ~5 days spanning the window."""
    return [
        start + datetime.timedelta(days=i) for i in range(0, (end - start).days + 1, 5)
    ]
