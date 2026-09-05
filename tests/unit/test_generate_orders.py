import pandas as pd
from generate_orders import generate_orders


def test_reproducible_with_fixed_seed():
    df1 = generate_orders(seed=42, n_rows=500)
    df2 = generate_orders(seed=42, n_rows=500)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seed_produces_different_data():
    df1 = generate_orders(seed=42, n_rows=500)
    df2 = generate_orders(seed=7, n_rows=500)
    assert not df1["actual_delivery_days"].equals(df2["actual_delivery_days"])


def test_row_count_matches_request():
    df = generate_orders(seed=42, n_rows=1000)
    assert len(df) == 1000


def test_no_negative_target():
    df = generate_orders(seed=42, n_rows=2000)
    assert (df["actual_delivery_days"] > 0).all()
