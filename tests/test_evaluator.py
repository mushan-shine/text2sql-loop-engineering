import datetime as dt
from decimal import Decimal

from benchmark.beaver.evaluator import (
    canonical_match, canonical_value, official_match, result_hash, serialize_rows,
)


class TestOfficialMatch:
    """Port of BEAVER eval/utils/ex_acc.py compare_results."""

    def test_both_empty_is_match(self):
        assert official_match([], []) == (True, "Both empty")

    def test_one_empty(self):
        assert official_match([], [(1,)])[0] is False

    def test_set_semantics_ignores_order_and_duplicates(self):
        assert official_match([(1, "a"), (2, "b"), (2, "b")], [(2, "b"), (1, "a")])[0]

    def test_column_order_matters(self):
        assert not official_match([("a", 1)], [(1, "a")])[0]

    def test_column_count_mismatch(self):
        ok, msg = official_match([(1, 2)], [(1,)])
        assert not ok and "Column count" in msg

    def test_values_stringified_and_stripped(self):
        assert official_match([(" x ",)], [("x",)])[0]
        # str(Decimal) keeps scale: official comparison is representation-sensitive
        assert not official_match([(Decimal("12.5000"),)], [(12.5,)])[0]


class TestCanonical:
    def test_numeric_representations_converge(self):
        assert canonical_value(Decimal("12.5000")) == canonical_value(12.5) == "12.5"
        assert canonical_value(Decimal("3")) == canonical_value(3) == canonical_value(3.0) == "3"

    def test_float_noise_below_precision(self):
        assert canonical_value(0.1 + 0.2) == canonical_value(Decimal("0.3"))

    def test_temporal_and_null(self):
        assert canonical_value(None) is None
        assert canonical_value(dt.datetime(2024, 1, 2, 3, 4, 5)) == "2024-01-02 03:04:05"
        assert canonical_value(dt.date(2024, 1, 2)) == "2024-01-02"
        assert canonical_value(dt.timedelta(hours=26, minutes=3)) == "26:03:00"

    def test_negative_zero(self):
        assert canonical_value(-0.0) == "0"

    def test_cross_engine_avg(self):
        mysql = [(Decimal("12.5000"), "A")]
        spark = [(12.5, "A")]
        c = canonical_match(spark, mysql)
        assert c.set_match and c.multiset_match and c.ordered_match

    def test_duplicate_counts_detected(self):
        c = canonical_match([(1,), (1,)], [(1,)])
        assert c.set_match and not c.multiset_match

    def test_hash_is_order_and_duplicate_insensitive(self):
        assert result_hash([(1,), (2,), (2,)]) == result_hash([(2,), (1,)])
        assert result_hash([(1,)]) != result_hash([(2,)])

    def test_serialize_roundtrip_hash(self):
        import json
        rows = [(Decimal("1.50"), dt.date(2020, 1, 1), None)]
        again = [tuple(r) for r in json.loads(serialize_rows(rows))]
        assert result_hash(again) == result_hash(rows)


class TestCrossEngine:
    """MySQL (reference) vs Databricks (candidate) value representations."""

    def test_mysql_decimal_scale(self):
        from benchmark.beaver.evaluator import cross_engine_value_equal as eq
        assert eq(Decimal("6.8333"), 6.833333333333333)      # AVG(int): MySQL keeps 4 decimals
        assert eq(Decimal("10.9091"), 10.909090909090908)
        assert eq(Decimal("0E-8"), 0.0)
        assert eq(Decimal("57"), 57)
        assert not eq(Decimal("6.8333"), 6.8340)               # differs at the displayed scale
        assert not eq(Decimal("24.8056"), 26.264705882352946)  # population vs sample variance

    def test_float_tolerance_and_nulls(self):
        from benchmark.beaver.evaluator import cross_engine_value_equal as eq
        assert eq(0.4330127020004725, 0.4330127018922194)      # summation-order noise
        assert not eq(4474.394116906014, 4483.289532645389)
        assert eq(None, None) and not eq(0.0, None)
        assert eq("Chemistry", "Chemistry ") and not eq("Chemistry", "chemistry")

    def test_row_matching(self):
        from benchmark.beaver.evaluator import cross_engine_match
        mysql = [("Physics", Decimal("6.8333"), 3), ("Math", Decimal("1.5000"), 2)]
        dbx = [("Math", 1.5, 2), ("Physics", 6.833333333333333, 3)]
        c = cross_engine_match(dbx, mysql)
        assert c.set_match and c.multiset_match and not c.ordered_match
        assert not cross_engine_match([("Math", 1.6, 2)], [("Math", Decimal("1.5000"), 2)]).set_match
        assert cross_engine_match([(1,), (1,)], [(1,)]).set_match
        assert not cross_engine_match([(1,), (1,)], [(1,)]).multiset_match
