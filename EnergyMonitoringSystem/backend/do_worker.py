from typing import Optional, List, Dict, Any
import asyncio

from backend.dal.database import db_helper
from backend.modbus_poller import ModbusPoller

def _get_pending_commands(limit: int = 20) -> List[Dict[str, Any]]:
    q = (
        """
        SELECT TOP (?) c.CommandID, c.AnalyzerID, c.CoilAddress, c.Command,
               c.RequestedBy, c.MaxRetries, ISNULL(c.RetryCount, 0) as RetryCount,
               a.IPAddress, a.ModbusID
        FROM app.DigitalOutputCommands c
        JOIN app.Analyzers a ON c.AnalyzerID = a.AnalyzerID
        WHERE c.ExecutionResult = 'PENDING'
        ORDER BY c.RequestedAt ASC
        """
    )
    return db_helper.execute_query(q, (limit,)) or []

def _update_result(command_id: int, result: str, error_msg: Optional[str] = None):
    params = {"@CommandID": command_id, "@ExecutionResult": result, "@ErrorMessage": error_msg}
    db_helper.execute_stored_procedure("app.sp_UpdateDigitalOutputResult", params)

async def _execute_command(cmd: Dict[str, Any]) -> None:
    command_id = int(cmd["CommandID"])
    host = cmd["IPAddress"]
    unit_id = int(cmd["ModbusID"] or 1)
    coil_address = int(cmd["CoilAddress"])
    command = str(cmd["Command"]).upper()
    max_retries = int(cmd.get("MaxRetries") or 3)

    poller = ModbusPoller(0, f"DO-Worker-{command_id}", host, 502, unit_id)
    if not poller.connect(max_retries):
        _update_result(command_id, "FAILED", f"connect_failed:{host}")
        return

    target_state = {"ON": True, "OFF": False}.get(command)
    if target_state is None:
        try:
            current = await poller.read_coil_state(coil_address)
            target_state = not bool(current) if current is not None else False
        except Exception:
            target_state = False

    success = False
    last_error = None
    for attempt in range(max_retries):
        try:
            ok = await poller.control_digital_output(coil_address, target_state)
            if ok:
                success = True
                break
            else:
                last_error = f"attempt_failed:{attempt+1}"
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        except Exception as e:
            last_error = f"attempt_error:{attempt+1}:{str(e)}"
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

    try:
        poller.disconnect()
    except Exception:
        pass

    if success:
        _update_result(command_id, "SUCCESS", None)
    else:
        _update_result(command_id, "FAILED", last_error or "unknown_error")

async def process_pending_commands(batch_size: int = 20) -> int:
    cmds = _get_pending_commands(batch_size)
    if not cmds:
        return 0
    for cmd in cmds:
        try:
            await _execute_command(cmd)
        except Exception as e:
            try:
                _update_result(int(cmd["CommandID"]), "FAILED", f"unexpected:{str(e)}")
            except Exception:
                pass
    return len(cmds)

async def run_worker_loop(poll_interval_seconds: int = 5):
    while True:
        try:
            count = await process_pending_commands(20)
        except Exception:
            count = 0
        await asyncio.sleep(poll_interval_seconds if count == 0 else 1)
