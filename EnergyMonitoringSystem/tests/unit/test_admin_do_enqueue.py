import json
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes_admin import router as admin_router


def test_admin_do_enqueue(monkeypatch):
    app = FastAPI()
    app.include_router(admin_router, prefix="/api/admin")
    client = TestClient(app)

    calls = {"sp": [], "audit": []}

    class DummyUser:
        def __init__(self):
            self.data = {"role": "Admin", "user_id": 99, "username": "admin"}

    def fake_get_current_user():
        return DummyUser().data

    from backend.api import routes_admin as ra
    monkeypatch.setattr(ra, "get_current_user", fake_get_current_user)

    def fake_sp(name, params):
        if name == "app.sp_ControlDigitalOutput":
            calls["sp"].append((name, params))
            return [{"CommandID": 123, "AnalyzerID": params["@AnalyzerID"], "IPAddress": "127.0.0.1", "ModbusID": 1}]
        if name == "ops.sp_LogAuditEvent":
            calls["audit"].append((name, params))
            return [{}]
        return [{}]

    monkeypatch.setattr(ra.db_helper, "execute_stored_procedure", fake_sp)

    body = {"analyzer_id": 7, "coil_address": 1, "command": "ON"}
    resp = client.post("/api/admin/do/enqueue", json=body)

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["command"]["CommandID"] == 123
    assert calls["sp"][0][0] == "app.sp_ControlDigitalOutput"
    assert calls["sp"][0][1]["@AnalyzerID"] == 7
    assert calls["sp"][0][1]["@CoilAddress"] == 1
    assert calls["sp"][0][1]["@Command"] == "ON"
    assert calls["audit"][0][0] == "ops.sp_LogAuditEvent"
