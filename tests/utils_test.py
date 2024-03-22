"""Contains utility functions to compare deeply nested dicts for
approximate equality"""


def assert_values_almost_equal(value1, value2, tolerance=1e-5):
    "Compare two arbitrary objects for approximate equality"
    if isinstance(value1, dict):
        assert_dicts_almost_equal(value1, value2, tolerance=tolerance)
    elif isinstance(value1, (list, tuple)):
        assert_sequences_almost_equal(value1, value2, tolerance=tolerance)
    elif isinstance(value1, (int, float)):
        assert_numbers_almost_equal(value1, value2, tolerance=tolerance)
    else:
        assert value1 == value2


def assert_dicts_almost_equal(d1, d2, tolerance=1e-5):
    "Recursively compare two dictionaries for approximate equality"
    for (k1, v1), (k2, v2) in zip(d1.items(), d2.items()):
        assert_values_almost_equal(k1, k2, tolerance=tolerance)
        assert_values_almost_equal(v1, v2, tolerance=tolerance)


def assert_sequences_almost_equal(s1, s2, tolerance=1e-5):
    "Recursively compare two sequences for approximate equality"
    for e1, e2 in zip(s1, s2):
        assert_values_almost_equal(e1, e2, tolerance=tolerance)


def assert_numbers_almost_equal(n1, n2, tolerance=1e-5):
    "Compare two numbers for approximate equality"
    assert abs(n1 - n2) < tolerance
