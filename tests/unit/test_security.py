import pytest

from vcoda.security import is_public_ip, validate_upload_name, validate_upload_size


def test_upload_name_prevents_path_traversal_and_extensions():
    assert validate_upload_name("../../capture.pcap", {".pcap"}) == "capture.pcap"
    with pytest.raises(ValueError):
        validate_upload_name("payload.exe", {".pcap"})


def test_upload_size_limit():
    with pytest.raises(ValueError):
        validate_upload_size(11, 10)


def test_public_ip_policy():
    assert is_public_ip("8.8.8.8") is True
    assert is_public_ip("127.0.0.1") is False
