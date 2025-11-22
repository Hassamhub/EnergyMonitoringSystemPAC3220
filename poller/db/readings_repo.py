from typing import Any, Dict

class ReadingsRepo:
    def __init__(self, sql_conn):
        self.sql_conn = sql_conn

    def insert_reading(self, row: Dict[str, Any]):
        sp_params_order = [
            "@AnalyzerID","@KW_L1","@KW_L2","@KW_L3","@KW_Total",
            "@KWh_L1","@KWh_L2","@KWh_L3","@KWh_Total",
            "@VL1","@VL2","@VL3",
            "@IL1","@IL2","@IL3","@ITotal",
            "@Hz","@PF_L1","@PF_L2","@PF_L3","@PF_Avg",
            "@KWh_Grid","@KWh_Generator","@Quality"
        ]
        try:
            conn = self.sql_conn.connect()
            try:
                cur = conn.cursor()
                values = [
                    row.get("AnalyzerID"),
                    row.get("KW_L1"), row.get("KW_L2"), row.get("KW_L3"), row.get("KW_Total"),
                    row.get("KWh_L1"), row.get("KWh_L2"), row.get("KWh_L3"), row.get("KWh_Total"),
                    row.get("VL1"), row.get("VL2"), row.get("VL3"),
                    row.get("IL1"), row.get("IL2"), row.get("IL3"), row.get("ITotal"),
                    row.get("Hz"), row.get("PF_L1"), row.get("PF_L2"), row.get("PF_L3"), row.get("PF_Avg"),
                    row.get("KWh_Grid"), row.get("KWh_Generator"),
                    "GOOD" if row.get("Quality", 100) == 100 else "DEGRADED",
                ]
                assignments = ", ".join([f"{p} = ?" for p in sp_params_order])
                sql_sp = f"EXEC app.sp_InsertReading {assignments}"
                cur.execute(sql_sp, values)
                conn.commit()
                conn.close()
                return
            except Exception:
                conn.close()
        except Exception:
            pass

        cols = [
            "AnalyzerID","Timestamp",
            "KW_L1","KW_L2","KW_L3","KW_Total",
            "KWh_L1","KWh_L2","KWh_L3","KWh_Total",
            "VL1","VL2","VL3",
            "IL1","IL2","IL3","ITotal",
            "Hz","PF_L1","PF_L2","PF_L3","PF_Avg",
            "KWh_Grid","KWh_Generator","DeltaKWh",
            "IsValid","Quality"
        ]
        values = [row.get(c) for c in cols]
        placeholders = ",".join(["?"]*len(cols))
        sql = f"INSERT INTO app.Readings ({','.join(cols)}) VALUES ({placeholders})"
        conn = self.sql_conn.connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, values)
            conn.commit()
        finally:
            conn.close()

    def get_last_kwh_total(self, analyzer_id: int):
        conn = self.sql_conn.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT TOP 1 KWh_Total FROM app.Readings WHERE AnalyzerID = ? ORDER BY Timestamp DESC",
                (analyzer_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            conn.close()