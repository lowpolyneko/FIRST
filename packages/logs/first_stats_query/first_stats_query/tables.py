"""Generate tables for the streams ingested by the CLI.

Each stream (metrics, request_log, access_log, user) can be dumped as a
raw table or reduced to summations. Rows are filtered by time range
(--period/--start/--end) and cluster; columns are selectable. With
``--aggregate``, numeric columns are summed instead of listed, optionally
grouped by a dimension column (``--group``). Output is printed as a polars
table or CSV.
"""

import datetime
from typing import Callable

import polars as pl

STREAMS = ("metrics", "request_log", "access_log", "user")

# Stream -> timestamp column used for time-range filtering (None = no time).
TIMESTAMP_COL: dict[str, str | None] = {
    "metrics": "timestamp_compute_request",
    "request_log": "timestamp_compute_request",
    "access_log": "timestamp_request",
    "user": None,
}

# Stream -> cluster column used for --cluster filtering (None = no cluster).
CLUSTER_COL: dict[str, str | None] = {
    "metrics": "cluster",
    "request_log": "cluster",
    "access_log": None,
    "user": None,
}


def generate_table(
    load_data: Callable[[str], pl.LazyFrame],
    stream: str,
    start: datetime.datetime | None,
    end: datetime.datetime | None,
    cluster: str | None,
    columns: list[str] | None = None,
    limit: int | None = None,
    sort: str | None = None,
    fmt: str = "table",
    aggregate: list[str] | None = None,
    group: str | None = None,
) -> None:
    """Print rows from ``stream``, optionally as column summations.

    Applies time-range and cluster selection, then either lists rows
    (column/sort/limit selection) or sums the ``aggregate`` columns,
    grouped by ``group`` when given.
    """
    lf = load_data(stream)

    timestamp_col = TIMESTAMP_COL[stream]
    if start is not None and end is not None and timestamp_col is not None:
        lf = lf.filter(
            pl.col(timestamp_col)
            .str.head(19)
            .str.to_datetime(time_zone="UTC")
            .is_between(start, end)
        )

    cluster_col = CLUSTER_COL[stream]
    if cluster is not None and cluster_col is not None:
        lf = lf.filter(pl.col(cluster_col).eq(cluster))

    if aggregate is not None:
        agg_exprs = [pl.col(name).sum().alias(name) for name in aggregate]
        if group is not None:
            lf = lf.group_by(group).agg(*agg_exprs)
        else:
            lf = lf.select(*agg_exprs)
        if sort is not None:
            column, sep, direction = sort.partition(":")
            lf = lf.sort(column, descending=sep == ":" and direction == "desc")
    else:
        if sort is not None:
            column, sep, direction = sort.partition(":")
            lf = lf.sort(column, descending=sep == ":" and direction == "desc")
        if columns is not None:
            lf = lf.select(*columns)

    if limit is not None:
        lf = lf.head(limit)

    df = lf.collect(engine="streaming")
    if fmt == "csv":
        print(df.write_csv())
    else:
        print(df)
