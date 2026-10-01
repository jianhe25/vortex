# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os
from datetime import datetime, timezone

import polars as pl
import pyarrow as pa
import pytest

import vortex as vx
import vortex.expr as ve
from vortex.polars_ import polars_to_vortex


@pytest.mark.parametrize(
    "polars, vortex",
    [
        (pl.col("AdvEngineID") != 0, ve.column("AdvEngineID") != 0),
        (pl.col("MobilePhoneModel") != "", ve.column("MobilePhoneModel") != ""),
        (pl.col("UserID") == 435090932899640449, ve.column("UserID") == 435090932899640449),
        # (pl.col("URL").str.contains("google"), ve.column("URL").str.contains("google")),
        # (
        #     (
        #         (pl.col("Title").str.contains("Google"))
        #         & (~pl.col("URL").str.contains(".google."))
        #         & (pl.col("SearchPhrase") != "")
        #     ),
        #     (
        #         (ve.column("Title").str.contains("Google"))
        #         & (~ve.column("URL").str.contains(".google."))
        #         & (ve.column("SearchPhrase") != "")
        #     ),
        # ),
        (pl.col("c") > 10000, ve.column("c") > 10000),
        #        (pl.col("EventDate") >= date(2013, 7, 1), ve.column("EventDate") >= date(2013, 7, 1)),
    ],
)
def test_exprs(polars: pl.Expr, vortex: ve.Expr) -> None:
    assert polars_to_vortex(polars).serialize() == vortex.serialize()


@pytest.fixture(scope="module")
def vxf(tmpdir_factory) -> vx.VortexFile:
    fname = tmpdir_factory.mktemp("data") / "polars_test.vortex"

    if not os.path.exists(fname):
        a = pa.array([{"index": x, "value": math.sqrt(x)} for x in range(1_000_000)])
        vx.io.write(vx.compress(vx.array(a)), str(fname))
    return vx.open(str(fname), without_segment_cache=True)


def test_to_polars_with_limit(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().limit(100).collect()
    assert len(df) == 100


def test_to_polars_with_filter(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().filter(pl.col("index") < 500).collect()
    assert len(df) == 500
    assert df["index"].to_list() == list(range(500))


def test_to_polars_with_projection(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().select("index").limit(10).collect()
    assert df.columns == ["index"]
    assert len(df) == 10


def test_to_polars_with_projection_and_filter(vxf: vx.VortexFile) -> None:
    df = vxf.to_polars().select("index", "value").filter(pl.col("index") < 100).collect()
    assert df.columns == ["index", "value"]
    assert len(df) == 100


def test_datetime_predicate_pushdown(tmp_path):
    table = pa.table(
        {
            "id": [0, 1, 2],
            "value": pa.array(
                [datetime(2026, 9, day, tzinfo=timezone.utc) for day in [16, 17, 18]],
                type=pa.timestamp("us", tz="UTC"),
            ),
        }
    )
    path = tmp_path / "datetimes.vortex"
    vx.io.write(vx.array(table), str(path))
    predicate = pl.col("value") >= datetime(2026, 9, 17, tzinfo=timezone.utc)
    result = vx.open(str(path)).to_polars().filter(predicate).collect()
    assert result["id"].to_list() == [1, 2]


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("source_zone", [None, "UTC", "Europe/London"])
@pytest.mark.parametrize("target_zone", [None, "UTC", "America/New_York"])
def test_replace_time_zone_columns(unit, source_zone, target_zone):
    frame = pl.DataFrame(
        {"dt": [datetime(2024, 1, 15, 12, 0), datetime(2024, 7, 15, 12, 0), None]}
    ).with_columns(pl.col("dt").cast(pl.Datetime(unit)).dt.replace_time_zone(source_zone))
    expression = pl.col("dt").dt.replace_time_zone(target_zone)
    converted = polars_to_vortex(expression)
    expected = frame.select(expression).to_series().to_arrow()
    actual = vx.array(frame.to_arrow()).apply(converted).to_arrow_array()
    assert actual.equals(expected)
    restored = ve.deserialize(converted.serialize())
    assert vx.array(frame.to_arrow()).apply(restored).to_arrow_array().equals(expected)


@pytest.mark.parametrize("ambiguous", ["earliest", "latest", "null"])
def test_replace_time_zone_ambiguous(ambiguous):
    frame = pl.DataFrame({"dt": [datetime(2024, 11, 3, 1, 30), None]})
    expression = pl.col("dt").dt.replace_time_zone("America/New_York", ambiguous=ambiguous)
    expected = frame.select(expression).to_series().to_arrow()
    actual = vx.array(frame.to_arrow()).apply(polars_to_vortex(expression)).to_arrow_array()
    assert actual.equals(expected)


def test_replace_time_zone_policy_column():
    frame = pl.DataFrame(
        {
            "dt": [datetime(2024, 11, 3, 1, 30)] * 4,
            "policy": ["earliest", "latest", "null", None],
        }
    )
    expression = pl.col("dt").dt.replace_time_zone("America/New_York", ambiguous=pl.col("policy"))
    expected = frame.select(expression).to_series().to_arrow()
    actual = vx.array(frame.to_arrow()).apply(polars_to_vortex(expression)).to_arrow_array()
    assert actual.equals(expected)


def test_replace_time_zone_non_existent_null():
    frame = pl.DataFrame({"dt": [datetime(2024, 3, 10, 2, 30), datetime(2024, 3, 10, 3, 30)]})
    expression = pl.col("dt").dt.replace_time_zone("America/New_York", non_existent="null")
    expected = frame.select(expression).to_series().to_arrow()
    actual = vx.array(frame.to_arrow()).apply(polars_to_vortex(expression)).to_arrow_array()
    assert actual.equals(expected)


@pytest.mark.parametrize("value", [datetime(2024, 11, 3, 1, 30), datetime(2024, 3, 10, 2, 30)])
def test_replace_time_zone_raises(value):
    frame = pl.DataFrame({"dt": [value]})
    expression = pl.col("dt").dt.replace_time_zone("America/New_York")
    with pytest.raises(pl.exceptions.ComputeError):
        frame.select(expression)
    with pytest.raises(RuntimeError, match="ambiguous|gap|fold"):
        vx.array(frame.to_arrow()).apply(polars_to_vortex(expression)).to_arrow_array()


def test_replace_time_zone_maps_to_native_expression():
    expression = pl.col("dt").dt.replace_time_zone(
        "America/New_York", ambiguous=pl.col("policy"), non_existent="null"
    )
    expected = ve.replace_time_zone(
        ve.column("dt"), "America/New_York", ambiguous=ve.column("policy"), non_existent="null"
    )
    assert polars_to_vortex(expression).serialize() == expected.serialize()


def test_replace_time_zone_same_zone_during_fold():
    frame = pl.DataFrame({"dt": [datetime(2024, 11, 3, 1, 30)]}).with_columns(
        pl.col("dt").dt.replace_time_zone("America/New_York", ambiguous="latest")
    )
    expression = pl.col("dt").dt.replace_time_zone("America/New_York")
    expected = frame.select(expression).to_series().to_arrow()
    actual = vx.array(frame.to_arrow()).apply(polars_to_vortex(expression)).to_arrow_array()
    assert actual.equals(expected)
