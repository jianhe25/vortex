# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import os

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

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
    schema = pl.Schema({"AdvEngineID": pl.Int64, "MobilePhoneModel": pl.String, "UserID": pl.Int64, "c": pl.Int64})
    assert polars_to_vortex(polars, schema=schema).serialize() == vortex.serialize()


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


def test_polars_boolean_xor(tmp_path):
    frame = pl.DataFrame(
        {
            "id": list(range(9)),
            "x": [False, False, True, True, None, None, None, True, False],
            "y": [False, True, False, True, False, True, None, None, None],
        }
    )
    expr = pl.col("x") ^ pl.col("y")
    path = tmp_path / "boolean_xor.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [1, 2]


def test_polars_boolean_xor_maps_to_not_equal():
    lhs = pl.col("x") > 0
    rhs = pl.col("y") < 5
    schema = pl.Schema({"x": pl.Int64, "y": pl.Int64})
    expected = (ve.column("x") > 0) != (ve.column("y") < 5)
    assert polars_to_vortex(lhs ^ rhs, schema=schema).serialize() == expected.serialize()


@pytest.mark.parametrize(
    "schema",
    [
        pl.Schema({"x": pl.Int64, "y": pl.Int64}),
        pl.Schema({"x": pl.Boolean, "y": pl.Int64}),
        pl.Schema({"x": pl.Int64, "y": pl.Boolean}),
    ],
)
def test_polars_xor_rejects_non_boolean_operands(schema):
    with pytest.raises(NotImplementedError, match="requires Boolean operands"):
        polars_to_vortex(pl.col("x") ^ pl.col("y"), schema=schema)


def test_polars_xor_rejects_integer_literals():
    with pytest.raises(NotImplementedError, match="requires Boolean operands"):
        polars_to_vortex(pl.lit(1) ^ pl.lit(2), schema=pl.Schema())
