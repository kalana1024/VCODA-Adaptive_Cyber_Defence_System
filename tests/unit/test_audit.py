import json

from vcoda.audit.chain import AuditChain


def test_audit_chain_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    chain = AuditChain(path)
    chain.append({"event": "one"})
    chain.append({"event": "two"})
    assert chain.verify()["valid"] is True
    rows = path.read_text().splitlines()
    record = json.loads(rows[0])
    record["event"]["event"] = "changed"
    rows[0] = json.dumps(record)
    path.write_text("\n".join(rows) + "\n")
    assert chain.verify()["valid"] is False
