import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes_admin import router as admin_router
from backend.do_worker import process_pending_commands


def test_do_end_to_end(monkeypatch):
    app = FastAPI()
    app.include_router(admin_router, prefix="/api/admin")
    client = TestClient(app)

    store = {
        "commands": [],
        "updates": [],
    }

    from backend.api import routes_admin as ra

    def fake_get_current_user():
        return {"role": "Admin", "user_id": 1, "username": "admin"}

    monkeypatch.setattr(ra, "get_current_user", fake_get_current_user)

    def fake_sp(name, params):
        if name == "app.sp_ControlDigitalOutput":
            cmd = {
                "CommandID": len(store["commands"]) + 1,
                "AnalyzerID": params["@AnalyzerID"],
                "CoilAddress": params["@CoilAddress"],
                "Command": params["@Command"],
                "RequestedBy": params["@RequestedBy"],
                "MaxRetries": params["@MaxRetries"],
                "IPAddress": "127.0.0.1",
                "ModbusID": 1,
            }
            store["commands"].append(cmd)
            return [cmd]
        if name == "ops.sp_LogAuditEvent":
            return [{}]
        return [{}]

    def fake_query(q, params=()):
        if "FROM app.DigitalOutputCommands" in q:
            return [{
                "CommandID": c["CommandID"],
                "AnalyzerID": c["AnalyzerID"],
                "CoilAddress": c["CoilAddress"],
                "Command": c["Command"],
                "RequestedBy": c["RequestedBy"],
                "MaxRetries": c["MaxRetries"],
                "RetryCount": 0,
                "IPAddress": "127.0.0.1",
                "ModbusID": 1,
            } for c in store["commands"]]
        return []

    from backend import do_worker as dw

    async def fake_control(addr, state):
        return True

    class DummyPoller:
        def __init__(self, *args, **kwargs):
            pass
        def connect(self, max_retries):
            return True
        async def read_coil_state(self, address):
            return False
        async def control_digital_output(self, address, state):
            return await fake_control(address, state)
        def disconnect(self):
            pass

    monkeypatch.setattr(ra.db_helper, "execute_stored_procedure", fake_sp)
    monkeypatch.setattr(dw.db_helper, "execute_stored_procedure", lambda name, params: store["updates"].append((name, params)))
    monkeypatch.setattr(dw.db_helper, "execute_query", fake_query)
    monkeypatch.setattr(dw, "ModbusPoller", DummyPoller)

    resp = client.post("/api/admin/do/enqueue", json={"analyzer_id": 5, "coil_address": 1, "command": "OFF"})
    assert resp.status_code == 200
    assert len(store["commands"]) == 1

    asyncio.get_event_loop().run_until_complete(process_pending_commands())

    assert len(store["updates"]) >= 1
    name, params = store["updates"][0]
    assert name == "app.sp_UpdateDigitalOutputResult"
    assert params["@ExecutionResult"] == "SUCCESS"
