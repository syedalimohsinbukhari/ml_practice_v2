"""DiagnosticSubsetsCallback subset construction: terciles, cross-tabs."""

import numpy as np

from gwml.training.callbacks import DiagnosticSubsetsCallback


def test_q_terciles_and_mchirp_cross_tab_partition_correctly(synthetic_params):
    subsets = DiagnosticSubsetsCallback._build_subsets(synthetic_params)

    q_low, q_mid, q_high = subsets["q_low"], subsets["q_mid"], subsets["q_high"]
    n = len(synthetic_params)
    assert np.array_equal(q_low | q_mid | q_high, np.ones(n, dtype=bool))
    assert not np.any(q_low & q_mid)
    assert not np.any(q_mid & q_high)
    assert not np.any(q_low & q_high)

    assert np.array_equal(
        subsets["q_low_mchirp_low"] | subsets["q_low_mchirp_high"], q_low
    )
    assert np.array_equal(
        subsets["q_high_mchirp_low"] | subsets["q_high_mchirp_high"], q_high
    )
    assert np.array_equal(
        subsets["q_high_mchirp_low"], q_high & subsets["mchirp_low"]
    )
    assert np.array_equal(
        subsets["q_high_mchirp_high"], q_high & subsets["mchirp_high"]
    )
