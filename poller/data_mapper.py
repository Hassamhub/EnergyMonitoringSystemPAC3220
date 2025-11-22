from typing import Dict, Any
from .conversions import to_float32_be, to_float64_be

class DataMapper:
    def __init__(self, register_cfg):
        self.registers = register_cfg

    def decode(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for r in self.registers:
            name = r["name"]
            dtype = r["dtype"]
            column = r["column"]
            regs = raw.get(name)
            if regs is None:
                out[column] = None
                continue
            if dtype == "float32":
                out[column] = to_float32_be(regs)
            elif dtype == "float64":
                out[column] = to_float64_be(regs)
            else:
                out[column] = None
        return out