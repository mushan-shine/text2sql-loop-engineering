import datetime as dt
from decimal import Decimal

import pyarrow.parquet as pq

from benchmark.beaver.replicate import (
    MySqlColumn, compare_profiles, create_table_sql, insert_from_parquet_sql, map_column, write_parquet,
)


def col(name, data_type, column_type=None, p=None, s=None, coll=None):
    return MySqlColumn(name, data_type, column_type or data_type, p, s, coll)


class TestTypeMapping:
    def test_numeric(self):
        assert map_column(col("a", "int")).databricks_type == "INT"
        assert map_column(col("a", "int", "int unsigned")).databricks_type == "BIGINT"
        assert map_column(col("a", "bigint", "bigint unsigned")).databricks_type == "DECIMAL(20,0)"
        assert map_column(col("a", "decimal", "decimal(12,2)", 12, 2)).databricks_type == "DECIMAL(12,2)"
        assert map_column(col("a", "double")).databricks_type == "DOUBLE"

    def test_temporal(self):
        m = map_column(col("a", "datetime"))
        assert m.databricks_type == "TIMESTAMP_NTZ" and m.staged_as_string
        assert map_column(col("a", "date")).databricks_type == "DATE"

    def test_collation_policy(self):
        ci = col("a", "varchar", "varchar(20)", coll="utf8mb4_0900_ai_ci")
        assert map_column(ci, "mirror_mysql_ci").databricks_type == "STRING COLLATE UTF8_LCASE"
        assert map_column(ci, "binary").databricks_type == "STRING"
        cs = col("a", "varchar", "varchar(20)", coll="utf8mb4_bin")
        assert map_column(cs, "mirror_mysql_ci").databricks_type == "STRING"


def test_parquet_staging_roundtrip(tmp_path):
    maps = [map_column(c) for c in (
        col("id", "int"), col("amt", "decimal", "decimal(10,2)", 10, 2), col("ts", "datetime"),
        col("d", "date"), col("name", "varchar", coll="utf8mb4_0900_ai_ci"))]
    batch = [
        (1, Decimal("3.50"), dt.datetime(2024, 1, 2, 3, 4, 5), dt.date(2024, 1, 2), "x"),
        (2, None, "0000-00-00 00:00:00", "0000-00-00", None),  # MySQL zero dates
    ]
    stats = write_parquet([batch], maps, tmp_path / "t.parquet")
    t = pq.read_table(tmp_path / "t.parquet").to_pylist()
    assert stats.rows == 2 and stats.nulled_invalid_dates == 2
    assert t[0] == {"c0": 1, "c1": Decimal("3.50"), "c2": "2024-01-02 03:04:05", "c3": dt.date(2024, 1, 2), "c4": "x"}
    assert t[1]["c2"] is None and t[1]["c3"] is None


def test_ddl_and_insert():
    maps = [map_column(col("BUILDING KEY", "int")), map_column(col("ts", "datetime")),
            map_column(col("n", "varchar", coll="utf8mb4_general_ci"))]
    ddl = create_table_sql("`c`.`dw`.`t`", maps)
    assert "`BUILDING KEY` INT" in ddl and "columnMapping.mode' = 'name'" in ddl
    assert "STRING COLLATE UTF8_LCASE" in ddl
    ins = insert_from_parquet_sql("`c`.`dw`.`t`", maps, "/Volumes/c/b/s/t.parquet")
    assert "CAST(c1 AS TIMESTAMP_NTZ) AS `ts`" in ins and "c0 AS `BUILDING KEY`" in ins
    assert "CAST(c2" not in ins


def test_profile_comparison():
    maps = [map_column(col("a", "int")), map_column(col("d", "date"))]
    assert compare_profiles(maps, (10, 10, 5, 8, 3), (10, 10, 5, 8, 3), 0) == []
    assert compare_profiles(maps, (10, 10, 5, 8, 3), (10, 10, 4, 8, 3), 0)  # distinct drift
    # zero-dates nulled during staging are tolerated on temporal columns only
    assert compare_profiles(maps, (10, 10, 5, 10, 3), (10, 10, 5, 8, 3), 2) == []
