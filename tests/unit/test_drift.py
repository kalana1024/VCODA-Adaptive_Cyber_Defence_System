from vcoda.drift.monitor import DriftMonitor


def test_drift_monitor_accepts_stream(tmp_path):
    monitor = DriftMonitor(window_size=50, output_dir=tmp_path)
    for i in range(100):
        monitor.update(event_id=str(i), confidence=0.1 if i < 50 else 0.99, anomaly_score=0.0, predicted_attack=i >= 50)
    status = monitor.status()
    assert status["window_rows"] == 50
    assert status["backend"] in {"river", "fallback_page_hinkley"}
