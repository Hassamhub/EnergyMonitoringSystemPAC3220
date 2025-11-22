from typing import Any, Dict, List

class AnalyzersRepo:
    def __init__(self, sql_conn):
        self.sql_conn = sql_conn

    def get_active_analyzers(self) -> List[Dict[str, Any]]:
        conn = self.sql_conn.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT AnalyzerID, IPAddress, ModbusID FROM app.Analyzers WHERE IsActive = 1"
            )
            rows = cur.fetchall()
            result = []
            for r in rows:
                result.append({
                    "AnalyzerID": r[0],
                    "IPAddress": r[1],
                    "ModbusID": r[2],
                })
            return result
        finally:
            conn.close()

    def set_online(self, analyzer_id: int):
        conn = self.sql_conn.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE app.Analyzers SET LastSeen = GETUTCDATE(), ConnectionStatus = 'ONLINE', UpdatedAt = GETUTCDATE() WHERE AnalyzerID = ?",
                (analyzer_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def set_offline(self, analyzer_id: int):
        conn = self.sql_conn.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE app.Analyzers SET ConnectionStatus = 'OFFLINE', UpdatedAt = GETUTCDATE() WHERE AnalyzerID = ?",
                (analyzer_id,),
            )
            conn.commit()
        finally:
            conn.close()