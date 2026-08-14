import pandas as pd

from vcoda.data.features import derive_flow_features


def test_derived_flow_features_are_numeric_and_finite():
    frame = pd.DataFrame({
        "in_bytes": [100], "out_bytes": [50], "in_packets": [10], "out_packets": [5],
        "flow_duration_ms": [1000],
    })
    result = derive_flow_features(frame)
    assert result.loc[0, "total_bytes"] == 150
    assert result.loc[0, "total_packets"] == 15
    assert result.loc[0, "packets_per_second"] == 15
    assert result.loc[0, "bytes_per_packet"] == 10
