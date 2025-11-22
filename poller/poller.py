import asyncio
from typing import Dict, Any
from .modbus_client import ModbusClientWrapper
from .utils.time_utils import utcnow, date_part, hour_part

class DevicePoller:
    def __init__(self, device: Dict[str, Any], cfg: Dict[str, Any], readings_repo, analyzers_repo, mapper):
        self.device = device
        self.cfg = cfg
        self.readings_repo = readings_repo
        self.analyzers_repo = analyzers_repo
        self.mapper = mapper
        self.interval = float(cfg["modbus"]["poll_interval_seconds"])
        self.port = int(cfg["modbus"]["port"])
        self.timeout = float(cfg["modbus"]["timeout_seconds"])
        self.retries = int(cfg["modbus"]["retries"])

    async def run(self):
        backoff = 1
        while True:
            client = ModbusClientWrapper(self.device["IPAddress"], self.port, self.timeout)
            if not client.connect():
                await asyncio.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)
                continue
            try:
                self.analyzers_repo.set_online(self.device["AnalyzerID"])
                while True:
                    raw = {}
                    for reg in self.mapper.registers:
                        regs = await asyncio.to_thread(
                            client.read_input, reg["address"], reg["count"], self.device["ModbusID"]
                        )
                        raw[reg["name"]] = regs
                    decoded = self.mapper.decode(raw)
                    now = utcnow()
                    prev_total = self.readings_repo.get_last_kwh_total(self.device["AnalyzerID"]) or 0.0
                    curr_total = decoded.get("KWh_Total")
                    delta = (curr_total - prev_total) if (curr_total is not None and prev_total is not None) else None
                    ok_fields = sum(1 for k,v in decoded.items() if v is not None)
                    total_fields = len(decoded)
                    quality = int((ok_fields/total_fields)*100) if total_fields else 0
                    is_valid = 1 if quality == 100 else 0
                    row = {
                        "AnalyzerID": self.device["AnalyzerID"],
                        "Timestamp": now,
                        "ReadingDate": date_part(now),
                        "ReadingHour": hour_part(now),
                        "KW_L1": None,
                        "KW_L2": None,
                        "KW_L3": None,
                        "KW_Total": decoded.get("KW_Total"),
                        "KWh_L1": None,
                        "KWh_L2": None,
                        "KWh_L3": None,
                        "KWh_Total": decoded.get("KWh_Total"),
                        "VL1": decoded.get("VL1"),
                        "VL2": decoded.get("VL2"),
                        "VL3": decoded.get("VL3"),
                        "IL1": decoded.get("IL1"),
                        "IL2": decoded.get("IL2"),
                        "IL3": decoded.get("IL3"),
                        "ITotal": decoded.get("ITotal"),
                        "Hz": decoded.get("Hz"),
                        "PF_L1": None,
                        "PF_L2": None,
                        "PF_L3": None,
                        "PF_Avg": decoded.get("PF_Avg"),
                        "KWh_Grid": decoded.get("KWh_Grid"),
                        "KWh_Generator": decoded.get("KWh_Generator"),
                        "DeltaKWh": delta,
                        "IsValid": is_valid,
                        "Quality": quality,
                    }
                    await asyncio.to_thread(self.readings_repo.insert_reading, row)
                    self.analyzers_repo.set_online(self.device["AnalyzerID"])
                    await asyncio.sleep(self.interval)
            except Exception:
                self.analyzers_repo.set_offline(self.device["AnalyzerID"])
                await asyncio.sleep(2)
            finally:
                client.close()