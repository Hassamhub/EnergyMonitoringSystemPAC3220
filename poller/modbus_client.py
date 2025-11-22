from typing import Optional, List
from pymodbus.client import ModbusTcpClient

class ModbusClientWrapper:
    def __init__(self, host: str, port: int, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None

    def connect(self) -> bool:
        self.client = ModbusTcpClient(self.host, port=self.port, timeout=self.timeout)
        return bool(self.client.connect())

    def close(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass

    def read_input(self, address: int, count: int, unit: int) -> Optional[List[int]]:
        if not self.client:
            return None
        rr = self.client.read_input_registers(address=address, count=count, slave=unit)
        if rr.isError():
            return None
        regs = getattr(rr, "registers", None)
        if not regs or len(regs) != count:
            return None
        return regs