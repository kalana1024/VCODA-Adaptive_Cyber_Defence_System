import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from vcoda.models.bundle import ModelBundle
from vcoda.models.preprocessing import VCODAPreprocessor


def test_model_bundle_alignment_and_checksum(tmp_path):
    x = pd.DataFrame({"bytes": [1, 2, 10, 12], "protocol": ["tcp", "tcp", "udp", "udp"]})
    y = np.array([0, 0, 1, 1])
    prep = VCODAPreprocessor().fit(x)
    model = LogisticRegression().fit(prep.transform(x), y)
    bundle = ModelBundle("logistic", "binary", model, prep, ["benign", "attack"], 0.5)
    path = tmp_path / "bundle.joblib"
    saved = bundle.save(path)
    loaded = ModelBundle.load(path, saved["sha256"])
    probabilities = loaded.predict_proba(pd.DataFrame({"bytes": [5], "protocol": ["new"]}))
    assert probabilities.shape == (1, 2)
