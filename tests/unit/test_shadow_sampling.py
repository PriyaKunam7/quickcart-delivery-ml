import uuid

from quickcart_ml.inference.shadow import deterministic_bucket, is_sampled


def test_same_key_always_produces_same_bucket():
    key = "order-12345"
    buckets = [deterministic_bucket(key) for _ in range(10)]
    assert len(set(buckets)) == 1


def test_bucket_is_within_valid_range():
    for key in [str(uuid.uuid4()) for _ in range(1000)]:
        bucket = deterministic_bucket(key)
        assert 0 <= bucket <= 99


def test_zero_percent_samples_nothing():
    keys = [str(uuid.uuid4()) for _ in range(1000)]
    assert not any(is_sampled(k, 0) for k in keys)


def test_hundred_percent_samples_everything():
    keys = [str(uuid.uuid4()) for _ in range(1000)]
    assert all(is_sampled(k, 100) for k in keys)


def test_twenty_five_percent_samples_approximately_a_quarter():
    keys = [str(uuid.uuid4()) for _ in range(50_000)]
    sampled_count = sum(is_sampled(k, 25) for k in keys)
    sampled_pct = sampled_count / len(keys) * 100
    # Statistical test: allow a few percentage points of slack rather
    # than requiring an exact 25.0%, since this is a hash-based
    # distribution over random inputs, not a rigged exact split.
    assert 23 <= sampled_pct <= 27


def test_same_key_repeated_sampling_decision_is_stable():
    key = "order-98765"
    decisions = [is_sampled(key, 25) for _ in range(20)]
    assert len(set(decisions)) == 1


def test_sampling_is_monotonic_in_percentage():
    """A key sampled at a lower percentage must remain sampled at any higher percentage."""
    keys = [str(uuid.uuid4()) for _ in range(5000)]
    sampled_at_25 = {k for k in keys if is_sampled(k, 25)}
    sampled_at_50 = {k for k in keys if is_sampled(k, 50)}
    sampled_at_75 = {k for k in keys if is_sampled(k, 75)}

    assert sampled_at_25.issubset(sampled_at_50)
    assert sampled_at_50.issubset(sampled_at_75)


def test_different_keys_can_land_in_different_buckets():
    """Sanity check that bucketing isn't degenerate (e.g. always bucket 0)."""
    keys = [str(uuid.uuid4()) for _ in range(1000)]
    buckets = {deterministic_bucket(k) for k in keys}
    assert len(buckets) > 50  # should see wide spread across 0-99
