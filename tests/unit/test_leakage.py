import pandas as pd

from vcoda.data.leakage import detect_leakage


def test_leakage_detection_finds_labels_ids_constants_and_high_cardinality():
    frame = pd.DataFrame({
        "label": [0, 1, 0], "flow_id": ["a", "b", "c"], "constant": [1, 1, 1],
        "host": ["h1", "h2", "h3"], "bytes": [1, 2, 3],
    })
    result = detect_leakage(
        frame, label_columns={"label"}, always_exclude=set(),
        identifier_patterns=["flow_id"], high_cardinality_ratio=0.98,
    )
    assert "label" in result["labels"]
    assert "flow_id" in result["identifiers"]
    assert "constant" in result["constants"]
    assert "host" in result["high_cardinality"]

def test_leakage_detection_supports_pandas_string_dtype():
    frame = pd.DataFrame({
        "host": pd.Series(
            ["h1", "h2", "h3"],
            dtype="string",
        ),
        "bytes": [1, 2, 3],
    })

    result = detect_leakage(
        frame,
        label_columns=set(),
        always_exclude=set(),
        identifier_patterns=[],
        high_cardinality_ratio=0.98,
    )

    assert "host" in result["high_cardinality"]
