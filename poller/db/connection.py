import os
import pyodbc
from typing import Any, Dict

class SqlConnection:
    def __init__(self, cfg: Dict[str, Any]):
        self.driver = cfg["database"]["driver"]
        self.server = cfg["database"]["server"]
        self.database = cfg["database"]["database"]
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.timeout = int(cfg["database"].get("timeout_seconds", 5))

    def connect(self):
        conn_str = (
            f"DRIVER={{{self.driver}}};SERVER={self.server};DATABASE={self.database};"
            f"UID={self.user};PWD={self.password};TrustServerCertificate=yes;"
        )
        return pyodbc.connect(conn_str, timeout=self.timeout)