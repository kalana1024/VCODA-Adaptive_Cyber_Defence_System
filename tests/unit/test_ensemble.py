import numpy as np
from sklearn.linear_model import LogisticRegression

from vcoda.ensemble.fusion import EnsembleBundle


def test_ensemble_combines_and_reports_disagreement():
    x = np.array([[0.1, 0.2], [0.8, 0.9], [0.4, 0.6], [0.7, 0.8]])
    y = np.array([0, 1, 0, 1])
    meta = LogisticRegression().fit(x, y)
    bundle = EnsembleBundle("stacking", ["a", "b"], None, meta, 0.5, 0.25)
    result = bundle.combine({"a": x[:, 0], "b": x[:, 1]})
    assert result["probability"].shape == (4,)
    assert np.all(result["uncertainty"] >= 0)
