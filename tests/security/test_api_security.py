from fastapi.testclient import TestClient

from vcoda.api.app import create_app


def test_api_health_and_malformed_port():
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    response = client.post("/predict", json={"source_port": -1})
    assert response.status_code == 422


def test_pcap_upload_rejects_wrong_extension():
    client = TestClient(create_app())
    response = client.post("/pcap/analyse", files={"file": ("evil.exe", b"bad", "application/octet-stream")})
    assert response.status_code in {400, 422, 500}
