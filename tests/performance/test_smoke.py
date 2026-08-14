import time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


def test_small_inference_performance_smoke():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1000, 20))
    y = (x[:, 0] > 0).astype(int)
    model = ExtraTreesClassifier(n_estimators=20, random_state=42, n_jobs=1).fit(x, y)
    started = time.perf_counter()
    model.predict_proba(x)
    assert time.perf_counter() - started < 5.0
