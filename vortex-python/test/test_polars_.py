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


def test_polars_struct_field(tmp_path):
    frame = pl.DataFrame({"id": [0, 1, 2, 3], "x": [{"a": 1}, None, {"a": 3}, {"a": None}]})
    expr = pl.col("x").struct.field("a") >= 2
    path = tmp_path / "struct_field.vortex"
    vx.io.write(vx.array(frame.to_arrow()), str(path))
    expected_frame = frame.lazy().filter(expr).collect()
    actual = vx.open(str(path)).to_polars().filter(expr).collect()
    assert_frame_equal(actual, expected_frame)
    assert actual["id"].to_list() == [2]


def _assert_struct_field_scan(tmp_path, parents, field, expected_values, expected_ids):
    table = pa.table({"id": range(len(parents)), "x": parents})
    frame = pl.from_arrow(table)
    assert frame.select(field).to_series().to_list() == expected_values
    path = tmp_path / "struct_parent_nulls.vortex"
    vx.io.write(vx.array(table), str(path))
    predicate = field >= 0
    actual = vx.open(str(path)).to_polars().filter(predicate).collect()
    assert_frame_equal(actual, frame.lazy().filter(predicate).collect())
    assert actual["id"].to_list() == expected_ids


def test_polars_struct_field_null_parent(tmp_path):
    leaf = pa.array([10, 20])
    parent = pa.StructArray.from_arrays([leaf], names=["value"], mask=pa.array([True, False]))
    assert parent.field("value").to_pylist() == [10, 20]
    field = pl.col("x").struct.field("value")
    _assert_struct_field_scan(tmp_path, parent, field, [None, 20], [1])


def test_polars_struct_field_null_inner_parent(tmp_path):
    leaf = pa.array([10, 20])
    inner = pa.StructArray.from_arrays([leaf], names=["value"], mask=pa.array([True, False]))
    outer = pa.StructArray.from_arrays([inner], names=["child"])
    assert outer.field("child").field("value").to_pylist() == [10, 20]
    field = pl.col("x").struct.field("child").struct.field("value")
    _assert_struct_field_scan(tmp_path, outer, field, [None, 20], [1])


def test_polars_struct_field_null_outer_parent(tmp_path):
    leaf = pa.array([10, 20])
    inner = pa.StructArray.from_arrays([leaf], names=["value"])
    outer = pa.StructArray.from_arrays([inner], names=["child"], mask=pa.array([True, False]))
    assert outer.field("child").field("value").to_pylist() == [10, 20]
    field = pl.col("x").struct.field("child").struct.field("value")
    _assert_struct_field_scan(tmp_path, outer, field, [None, 20], [1])


def test_polars_struct_field_null_middle_parent(tmp_path):
    leaf = pa.array([10, 20])
    inner = pa.StructArray.from_arrays([leaf], names=["value"])
    middle = pa.StructArray.from_arrays([inner], names=["child"], mask=pa.array([True, False]))
    outer = pa.StructArray.from_arrays([middle], names=["child"])
    assert outer.field("child").field("child").field("value").to_pylist() == [10, 20]
    field = pl.col("x").struct.field("child").struct.field("child").struct.field("value")
    _assert_struct_field_scan(tmp_path, outer, field, [None, 20], [1])


def test_polars_struct_field_null_parents_and_leaf(tmp_path):
    leaf = pa.array([10, 20, 30, None, 50])
    inner = pa.StructArray.from_arrays(
        [leaf], names=["value"], mask=pa.array([True, False, True, False, False])
    )
    outer = pa.StructArray.from_arrays(
        [inner], names=["child"], mask=pa.array([False, True, True, False, False])
    )
    assert outer.field("child").field("value").to_pylist() == [10, 20, 30, None, 50]
    field = pl.col("x").struct.field("child").struct.field("value")
    _assert_struct_field_scan(tmp_path, outer, field, [None, None, None, None, 50], [4])
