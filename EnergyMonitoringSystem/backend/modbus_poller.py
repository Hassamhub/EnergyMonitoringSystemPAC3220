#!/usr/bin/env python3
"""
PAC3220 Modbus Poller with Billing Integration
Enhanced poller that integrates with the billing engine for real-time cost calculation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import struct
import math
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

"""
PAC3220 Modbus Poller Service
Polls Siemens PAC3220 energy analyzer via Modbus TCP and logs data to database.
"""

import asyncio
import time
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pymodbus.client import ModbusTcpClient
import struct

# Add project root to path for imports (so 'backend' package is importable)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.dal.database import db_helper

# Load environment variables
# Construct the path to the .env file
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Construct DATABASE_URL from components
db_driver = os.getenv("DB_DRIVER")
db_server = os.getenv("DB_SERVER")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

if all([db_driver, db_server, db_name, db_user, db_password]):
    conn_str = f"DRIVER={{{db_driver}}};SERVER={db_server};DATABASE={db_name};UID={db_user};PWD={db_password}"
    os.environ["DATABASE_URL"] = conn_str
else:
    print("WARNING: One or more database environment variables are not set.")

# Configuration from environment
DEVICE_ID = int(os.getenv("DEVICE_ID", "3"))
DEVICE_NAME = os.getenv("DEVICE_NAME", "PAC3220-001")
MODBUS_HOST = os.getenv("MODBUS_HOST", "192.168.10.2")
MODBUS_PORT = int(os.getenv("MODBUS_PORT", "502"))
MODBUS_UNIT_ID = int(os.getenv("MODBUS_UNIT_ID", "1"))
MODBUS_TIMEOUT = float(os.getenv("MODBUS_TIMEOUT", "5"))
POLL_INTERVAL = int(os.getenv("MODBUS_POLL_INTERVAL", "60"))  # Safer default poll interval
MAX_RETRIES = int(os.getenv("MODBUS_MAX_RETRIES", "5"))
RETRY_BACKOFF_BASE = float(os.getenv("MODBUS_RETRY_BACKOFF_BASE", "2.0"))  # seconds
SIMULATE_POLLING = os.getenv("SIMULATE_POLLING", "false").lower() == "true"

# Logging setup (single initialization, UTF-8 safe)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "backend/logs/poller_log.txt")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Debug: Check if env loaded
logger.info(f"Environment loaded - DB_URL exists: {bool(os.getenv('DATABASE_URL'))}")

# --------------------------
#   Register Converters
# --------------------------

def regs_to_float32(regs):
    if len(regs) < 2:
        return None
    try:
        raw = struct.pack('>HH', regs[0], regs[1])
        val = struct.unpack('>f', raw)[0]
        if val is None or math.isnan(val) or math.isinf(val) or abs(val) > 1e9:
            return None
        return val
    except Exception:
        return None

def regs_to_float64(regs):
    """Convert 4 Modbus registers into IEEE754 float64 (Big-Endian)."""
    if len(regs) < 4:
        return None
    raw = struct.pack('>HHHH', regs[0], regs[1], regs[2], regs[3])
    return struct.unpack('>d', raw)[0]

class ModbusPoller:
    """PAC3220 Modbus poller with database logging"""

    def __init__(self, device_id: int, device_name: str, host: str, port: int, unit_id: int):
        self.device_id = device_id
        self.device_name = device_name
        self.poll_interval = POLL_INTERVAL
        self.client = ModbusTcpClient(host=host, port=port, timeout=MODBUS_TIMEOUT)
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.running = False
        self._param_code_to_id: Dict[str, int] = {}
        self._param_cache_loaded = False

    def connect(self, retries: int = MAX_RETRIES) -> bool:
        """Establish Modbus connection with exponential backoff retries."""
        attempt = 0
        while attempt <= retries:
            logger.info(f"Attempting to connect to {self.device_name} at {self.host}:{self.port} (Attempt {attempt+1}/{retries+1})")
            try:
                if self.client.connect():
                    try:
                        cfg = db_helper.execute_query("SELECT ConfigValue FROM ops.Configuration WHERE ConfigKey = 'system.poller_interval'")
                        if cfg and cfg[0].get("ConfigValue"):
                            self.poll_interval = int(cfg[0]["ConfigValue"]) or self.poll_interval
                    except Exception:
                        pass
                    logger.info(f"Connected to {self.device_name}")
                    return True
            except Exception as e:
                logger.error(f"Connection attempt failed: {e}")
            backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
            logger.info(f"Retrying in {backoff:.1f} seconds...")
            # Backoff using blocking sleep (method is synchronous)
            time.sleep(backoff)
            attempt += 1
        logger.error(f"Failed to connect to {self.device_name} after {retries+1} attempts.")
        return False

    def disconnect(self):
        """Close Modbus connection"""
        try:
            self.client.close()
            logger.info(f"Disconnected from {MODBUS_HOST}")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")

    def read_input_register(self, address: int, count: int, datatype: str = "float32"):
        try:
            attempts = 0
            last_err = None
            while attempts < MAX_RETRIES:
                attempts += 1
                try:
                    rr = self.client.read_input_registers(address=address, count=count, slave=self.unit_id)
                    if rr and not rr.isError():
                        regs = rr.registers
                        if regs is None or len(regs) != count:
                            last_err = f"Incorrect register count at {address}: got {len(regs) if regs else 0}, expected {count}"
                        else:
                            if datatype == "float32":
                                return regs_to_float32(regs)
                            elif datatype == "float64":
                                return regs_to_float64(regs)
                            else:
                                return regs
                    else:
                        last_err = f"Modbus error at address {address}"
                except Exception as e:
                    last_err = f"Exception at address {address}: {e}"
                time.sleep(0.2)

            # Fallback to holding registers
            attempts = 0
            while attempts < MAX_RETRIES:
                attempts += 1
                try:
                    rr2 = self.client.read_holding_registers(address=address, count=count, slave=self.unit_id)
                    if rr2 and not rr2.isError():
                        regs = rr2.registers
                        if regs is None or len(regs) != count:
                            last_err = f"Incorrect holding register count at {address}: got {len(regs) if regs else 0}, expected {count}"
                        else:
                            if datatype == "float32":
                                return regs_to_float32(regs)
                            elif datatype == "float64":
                                return regs_to_float64(regs)
                            else:
                                return regs
                    else:
                        last_err = f"Modbus holding error at address {address}"
                except Exception as e:
                    last_err = f"Exception holding at address {address}: {e}"
                time.sleep(0.2)

            logger.warning(f"Error reading address {address} via FC04 and FC03. LastErr={last_err}")
            return None
        except Exception as e:
            logger.error(f"Exception reading address {address}: {e}")
            return None

    def read_all_parameters(self) -> Dict[str, float]:
        """
        Read all parameters from the device

        Returns:
            Dictionary of parameter_name -> scaled_value
        """
        readings = {}

        # PAC3220 register mapping per provided register table (offsets are 1-based Modbus addresses)
        # We map L-L voltages to VL1/VL2/VL3 to match DB parameter codes.
        parameter_reads = {
            "voltage_l1": (7, 2, "float32"),
            "voltage_l2": (9, 2, "float32"),
            "voltage_l3": (11, 2, "float32"),

            "current_l1": (13, 2, "float32"),
            "current_l2": (15, 2, "float32"),
            "current_l3": (17, 2, "float32"),
            "current_avg": (19, 2, "float32"),

            "power_kw_l1": (61, 2, "float32"),
            "power_kw_l2": (63, 2, "float32"),
            "power_kw_l3": (67, 2, "float32"),
            "power_kw_total": (65, 2, "float32"),

            "pf_l1": (71, 2, "float32"),
            "pf_l2": (73, 2, "float32"),
            "pf_l3": (75, 2, "float32"),
            "pf_total": (69, 2, "float32"),

            "frequency": (55, 2, "float32"),

            "energy_kwh_import_t1": (801, 4, "float64"),
            "energy_kwh_import_t2": (805, 4, "float64"),
            "energy_kwh_export_t1": (809, 4, "float64"),
            "energy_kwh_export_t2": (813, 4, "float64"),
        }

        for param_name, (address, count, datatype) in parameter_reads.items():
            value = self.read_input_register(address, count, datatype)
            if value is not None:
                if param_name in ("power_kw_total", "power_kw_l1", "power_kw_l2", "power_kw_l3"):
                    readings[param_name] = value / 1000.0
                else:
                    readings[param_name] = value
            else:
                readings[param_name] = None

        def nz(x):
            try:
                if x is None:
                    return 0.0
                v = float(x)
                if v != v or v == float('inf') or v == float('-inf'):
                    return 0.0
                return v
            except Exception:
                return 0.0

        kw = nz(readings.get("power_kw_total"))
        itotal = nz(readings.get("current_avg"))
        vl1 = nz(readings.get("voltage_l1"))
        pf = readings.get("pf_total")
        if (pf is None or nz(pf) == 0.0) and kw < 0.001 and itotal < 0.01 and vl1 > 100.0:
            readings["pf_total"] = 1.0

        def roundv(x, places):
            try:
                if x is None:
                    return None
                return round(float(x), places)
            except Exception:
                return None

        readings["voltage_l1"] = roundv(readings.get("voltage_l1"), 2)
        readings["voltage_l2"] = roundv(readings.get("voltage_l2"), 2)
        readings["voltage_l3"] = roundv(readings.get("voltage_l3"), 2)
        readings["current_l1"] = roundv(readings.get("current_l1"), 3)
        readings["current_l2"] = roundv(readings.get("current_l2"), 3)
        readings["current_l3"] = roundv(readings.get("current_l3"), 3)
        readings["current_avg"] = roundv(readings.get("current_avg"), 3)
        readings["power_kw_l1"] = roundv(readings.get("power_kw_l1"), 3)
        readings["power_kw_l2"] = roundv(readings.get("power_kw_l2"), 3)
        readings["power_kw_l3"] = roundv(readings.get("power_kw_l3"), 3)
        readings["power_kw_total"] = roundv(readings.get("power_kw_total"), 3)
        readings["pf_l1"] = roundv(readings.get("pf_l1"), 2)
        readings["pf_l2"] = roundv(readings.get("pf_l2"), 2)
        readings["pf_l3"] = roundv(readings.get("pf_l3"), 2)
        readings["pf_total"] = roundv(readings.get("pf_total"), 2)
        readings["frequency"] = roundv(readings.get("frequency"), 2)

        return readings

    # Mapping from parameter names to database parameter codes (MATCHING DATABASE SCHEMA)
    PARAMETER_NAME_TO_CODE = {
        "voltage_l1": "VL1",
        "voltage_l2": "VL2",
        "voltage_l3": "VL3",
        "current_l1": "IL1",
        "current_l2": "IL2",
        "current_l3": "IL3",
        "current_avg": "ITotal",
        "power_kw_l1": "KW_L1",
        "power_kw_l2": "KW_L2",
        "power_kw_l3": "KW_L3",
        "power_kw_total": "KW_Total",
        "pf_l1": "PF_L1",
        "pf_l2": "PF_L2",
        "pf_l3": "PF_L3",
        "pf_total": "PF_Avg",
        "frequency": "Hz",
        "energy_kwh_import_t1": "KWh_Grid",
        "energy_kwh_import_t2": "KWh_Generator",
        "energy_kwh_export_t1": "KWh_Export_T1",
        "energy_kwh_export_t2": "KWh_Export_T2",
    }

    def _ensure_param_cache(self):
        if self._param_cache_loaded:
            return
        try:
            rows = db_helper.execute_query(
                "SELECT ParameterID, ParameterCode FROM meta.DeviceParameters WHERE IsActive = 1"
            ) or []
            self._param_code_to_id = {r["ParameterCode"]: int(r["ParameterID"]) for r in rows}
            self._param_cache_loaded = True
            logger.info(f"Loaded {len(self._param_code_to_id)} parameter IDs from database")
        except Exception as e:
            logger.error(f"Failed to load parameter cache: {e}")

    def _resolve_parameter_id(self, param_name: str) -> Optional[int]:
        code = self.PARAMETER_NAME_TO_CODE.get(param_name)
        if not code:
            return None
        if not self._param_cache_loaded:
            self._ensure_param_cache()
        return self._param_code_to_id.get(code)

    async def log_readings_to_database(self, readings: Dict[str, float]) -> bool:
        """
        Log parameter readings to database using the new Readings table structure
        with integrated billing engine processing.

        Args:
            readings: Dictionary of parameter_name -> value

        Returns:
            True if successful, False otherwise
        """
        try:
            # Helper function to sanitize float values
            def sanitize_float(value, default=0.0):
                """Convert invalid floats (nan, None, inf) to default value"""
                try:
                    if value is None:
                        return default
                    v = float(value)
                    if v != v or v == float('inf') or v == float('-inf'):
                        return default
                    return v
                except Exception:
                    return default

            # Extract and scale energy readings (PAC manual indicates DOUBLE Wh at 801/805/809/813)
            imp_t1_wh = readings.get("energy_kwh_import_t1")
            imp_t2_wh = readings.get("energy_kwh_import_t2")
            # Convert Wh -> kWh unconditionally
            def wh_to_kwh(raw):
                if raw is None:
                    return None
                try:
                    v = float(raw)
                    return v / 1000.0
                except Exception:
                    return None

            kwh_grid_val = wh_to_kwh(imp_t1_wh)
            kwh_gen_val = wh_to_kwh(imp_t2_wh)
            # Total = sum of import tariffs when available
            if kwh_grid_val is not None and kwh_gen_val is not None:
                kwh_total_val = kwh_grid_val + kwh_gen_val
            else:
                kwh_total_val = kwh_grid_val if kwh_grid_val is not None else kwh_gen_val

            # Map readings to the database schema parameters with proper sanitization
            params = {
                "@AnalyzerID": self.device_id,
                "@KW_L1": sanitize_float(readings.get("power_kw_l1")),
                "@KW_L2": sanitize_float(readings.get("power_kw_l2")),
                "@KW_L3": sanitize_float(readings.get("power_kw_l3")),
                "@KW_Total": sanitize_float(readings.get("power_kw_total")),
                "@KWh_L1": 0.0,  # Per-phase energy not directly available
                "@KWh_L2": 0.0,
                "@KWh_L3": 0.0,
                "@KWh_Total": sanitize_float(kwh_total_val, 0.0),
                "@VL1": sanitize_float(readings.get("voltage_l1")),
                "@VL2": sanitize_float(readings.get("voltage_l2")),
                "@VL3": sanitize_float(readings.get("voltage_l3")),
                "@IL1": sanitize_float(readings.get("current_l1")),
                "@IL2": sanitize_float(readings.get("current_l2")),
                "@IL3": sanitize_float(readings.get("current_l3")),
                "@ITotal": sanitize_float(readings.get("current_avg")),
                "@Hz": sanitize_float(readings.get("frequency")),
                "@PF_L1": sanitize_float(readings.get("pf_l1")),
                "@PF_L2": sanitize_float(readings.get("pf_l2")),
                "@PF_L3": sanitize_float(readings.get("pf_l3")),
                "@PF_Avg": sanitize_float(readings.get("pf_total")),
                "@KWh_Grid": sanitize_float(kwh_grid_val, 0.0),
                "@KWh_Generator": sanitize_float(kwh_gen_val, 0.0),
                "@Quality": "GOOD"
            }

            major = [params["@KW_Total"], params["@ITotal"], params["@Hz"], params["@PF_Avg"], params["@VL1"]]
            if sum(1 for v in major if v and v > 0.0) == 0:
                params["@Quality"] = "SUSPECT"

            try:
                prev_rows = db_helper.execute_query(
                    "SELECT TOP 1 KWh_Total FROM app.Readings WHERE AnalyzerID = ? ORDER BY Timestamp DESC",
                    (self.device_id,)
                ) or []
                prev_kwh = None
                if prev_rows:
                    try:
                        prev_kwh = float(prev_rows[0].get("KWh_Total"))
                    except Exception:
                        prev_kwh = None
                cur_kwh = params["@KWh_Total"] if params["@KWh_Total"] is not None else None

                sp_params = {
                    "@AnalyzerID": self.device_id,
                    "@KW_L1": params.get("@KW_L1"),
                    "@KW_L2": params.get("@KW_L2"),
                    "@KW_L3": params.get("@KW_L3"),
                    "@KW_Total": params.get("@KW_Total"),
                    "@KWh_L1": params.get("@KWh_L1"),
                    "@KWh_L2": params.get("@KWh_L2"),
                    "@KWh_L3": params.get("@KWh_L3"),
                    "@KWh_Total": params.get("@KWh_Total"),
                    "@VL1": params.get("@VL1"),
                    "@VL2": params.get("@VL2"),
                    "@VL3": params.get("@VL3"),
                    "@IL1": params.get("@IL1"),
                    "@IL2": params.get("@IL2"),
                    "@IL3": params.get("@IL3"),
                    "@ITotal": params.get("@ITotal"),
                    "@Hz": params.get("@Hz"),
                    "@PF_L1": params.get("@PF_L1"),
                    "@PF_L2": params.get("@PF_L2"),
                    "@PF_L3": params.get("@PF_L3"),
                    "@PF_Avg": params.get("@PF_Avg"),
                    "@KWh_Grid": params.get("@KWh_Grid"),
                    "@KWh_Generator": params.get("@KWh_Generator"),
                    "@Quality": params.get("@Quality") or "GOOD",
                }

                # If KWh_Total is missing or non-positive, bypass SP and insert directly
                if sp_params.get("@KWh_Total") is None or (isinstance(sp_params.get("@KWh_Total"), (int, float)) and sp_params.get("@KWh_Total") <= 0):
                    sp_err = None
                    insert_q = (
                        """
                        INSERT INTO app.Readings (
                            AnalyzerID, KW_L1, KW_L2, KW_L3, KW_Total,
                            KWh_L1, KWh_L2, KWh_L3, KWh_Total,
                            VL1, VL2, VL3, IL1, IL2, IL3, ITotal,
                            Hz, PF_L1, PF_L2, PF_L3, PF_Avg,
                            KWh_Grid, KWh_Generator, Quality
                        ) VALUES (
                            ?, ?, ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?,
                            ?, ?, ?
                        )
                        """
                    )
                    db_helper.execute_query(
                        insert_q,
                        (
                            self.device_id,
                            params.get("@KW_L1"), params.get("@KW_L2"), params.get("@KW_L3"), params.get("@KW_Total"),
                            params.get("@KWh_L1"), params.get("@KWh_L2"), params.get("@KWh_L3"), params.get("@KWh_Total") or 0.0,
                            params.get("@VL1"), params.get("@VL2"), params.get("@VL3"), params.get("@IL1"), params.get("@IL2"), params.get("@IL3"), params.get("@ITotal"),
                            params.get("@Hz"), params.get("@PF_L1"), params.get("@PF_L2"), params.get("@PF_L3"), params.get("@PF_Avg"),
                            params.get("@KWh_Grid") or 0.0, params.get("@KWh_Generator") or 0.0, params.get("@Quality") or "GOOD",
                        )
                    )
                else:
                    try:
                        db_helper.execute_stored_procedure("app.sp_InsertReading", sp_params)
                    except Exception as sp_err:
                        insert_q = (
                            """
                            INSERT INTO app.Readings (
                                AnalyzerID, KW_L1, KW_L2, KW_L3, KW_Total,
                                KWh_L1, KWh_L2, KWh_L3, KWh_Total,
                                VL1, VL2, VL3, IL1, IL2, IL3, ITotal,
                                Hz, PF_L1, PF_L2, PF_L3, PF_Avg,
                                KWh_Grid, KWh_Generator, Quality
                            ) VALUES (
                                ?, ?, ?, ?, ?,
                                ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?,
                                ?, ?, ?
                            )
                            """
                        )
                        db_helper.execute_query(
                            insert_q,
                            (
                                self.device_id,
                                params.get("@KW_L1"), params.get("@KW_L2"), params.get("@KW_L3"), params.get("@KW_Total"),
                                params.get("@KWh_L1"), params.get("@KWh_L2"), params.get("@KWh_L3"), params.get("@KWh_Total") or 0.0,
                                params.get("@VL1"), params.get("@VL2"), params.get("@VL3"), params.get("@IL1"), params.get("@IL2"), params.get("@IL3"), params.get("@ITotal"),
                                params.get("@Hz"), params.get("@PF_L1"), params.get("@PF_L2"), params.get("@PF_L3"), params.get("@PF_Avg"),
                                params.get("@KWh_Grid") or 0.0, params.get("@KWh_Generator") or 0.0, params.get("@Quality") or "GOOD",
                            )
                        )
                    insert_q = (
                        """
                        INSERT INTO app.Readings (
                            AnalyzerID, KW_L1, KW_L2, KW_L3, KW_Total,
                            KWh_L1, KWh_L2, KWh_L3, KWh_Total,
                            VL1, VL2, VL3, IL1, IL2, IL3, ITotal,
                            Hz, PF_L1, PF_L2, PF_L3, PF_Avg,
                            KWh_Grid, KWh_Generator, Quality
                        ) VALUES (
                            ?, ?, ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?,
                            ?, ?, ?
                        )
                        """
                    )
                    db_helper.execute_query(
                        insert_q,
                        (
                            self.device_id,
                            params.get("@KW_L1"), params.get("@KW_L2"), params.get("@KW_L3"), params.get("@KW_Total"),
                            params.get("@KWh_L1"), params.get("@KWh_L2"), params.get("@KWh_L3"), params.get("@KWh_Total"),
                            params.get("@VL1"), params.get("@VL2"), params.get("@VL3"), params.get("@IL1"), params.get("@IL2"), params.get("@IL3"), params.get("@ITotal"),
                            params.get("@Hz"), params.get("@PF_L1"), params.get("@PF_L2"), params.get("@PF_L3"), params.get("@PF_Avg"),
                            params.get("@KWh_Grid"), params.get("@KWh_Generator"), params.get("@Quality") or "GOOD",
                        )
                    )

                try:
                    db_helper.execute_query(
                        "UPDATE app.Analyzers SET LastSeen = GETUTCDATE(), ConnectionStatus = 'ONLINE', UpdatedAt = GETUTCDATE() WHERE AnalyzerID = ?",
                        (self.device_id,),
                    )
                except Exception:
                    pass

                try:
                    db_helper.execute_stored_procedure("app.sp_ApplyBillingForAnalyzer", {"@AnalyzerID": self.device_id})
                except Exception:
                    pass

                logger.info(f"Successfully logged reading for analyzer {self.device_id}")
                return True
            except Exception as db_error:
                logger.error(f"Database error logging reading for analyzer {self.device_id}: {db_error}")
                return False

        except Exception as e:
            logger.error(f"Database error logging readings: {e}")
            return False

    async def poll_and_log_once(self) -> bool:
        """
        Poll the device once and log readings to database
        Used for multi-device polling

        Returns:
            True if successful
        """
        try:
            # Persistent connection check with recovery
            if not self.client.is_socket_open():
                if not self.connect():
                    if SIMULATE_POLLING:
                        logger.warning("Connection failed, SIMULATE_POLLING enabled — generating synthetic readings")
                        now = datetime.utcnow()
                        synthetic = {
                            "voltage_l1": 230.0,
                            "voltage_l2": 231.0,
                            "voltage_l3": 229.5,
                            "current_l1": 5.2,
                            "current_l2": 5.0,
                            "current_l3": 5.1,
                            "current_avg": 5.1,
                            "power_kw_total": 1.2,
                            "pf_total": 0.98,
                            "frequency": 50.0,
                            "energy_kwh_import_t1": 100000.0,  # Wh
                            "energy_kwh_import_t2": 0.0,
                            "energy_kwh_export_t1": 0.0,
                            "energy_kwh_export_t2": 0.0,
                        }
                        success = await self.log_readings_to_database(synthetic)
                        if success:
                            try:
                                db_helper.execute_query(
                                    "UPDATE app.Analyzers SET LastSeen = GETUTCDATE(), ConnectionStatus = 'ONLINE' WHERE AnalyzerID = ?",
                                    (self.device_id,)
                                )
                            except Exception:
                                pass
                            return True
                        return False
                    logger.error(f"Failed to connect to {self.device_name}")
                    return False

            readings = self.read_all_parameters()

            if readings:
                success = await self.log_readings_to_database(readings)
                if success:
                    logger.info(f"Successfully polled and logged {len(readings)} parameters for {self.device_name}")
                    return True
                else:
                    logger.error(f"Failed to log readings for {self.device_name}")
                    return False
            else:
                logger.warning(f"No readings obtained from {self.device_name}")
                return False

        except Exception as e:
            logger.error(f"Error polling {self.device_name}: {e}")
            return False
        # Don't disconnect here as we want to keep connection open for multi-device polling
            pass

    async def control_digital_output(self, coil_address: int, state: bool) -> bool:
        """
        Control digital output (breaker control)

        Args:
            coil_address: Modbus coil address
            state: True to turn ON, False to turn OFF

        Returns:
            True if successful
        """
        try:
            response = self.client.write_coil(
                address=coil_address,
                value=state,
                slave=self.unit_id
            )

            if response and not response.isError():
                logger.info(f"Digital output {coil_address} set to {'ON' if state else 'OFF'}")
                return True
            else:
                logger.error(f"Failed to set digital output {coil_address}")
                return False
        except Exception as e:
            logger.error(f"Error controlling digital output {coil_address}: {e}")
            return False

    async def run_polling_loop(self):
        """Main polling loop"""
        logger.info(f"Starting Modbus polling for {self.device_name} (Device ID: {self.device_id})")
        logger.info(f"Poll interval: {self.poll_interval} seconds")

        consecutive_errors = 0
        max_consecutive_errors = MAX_RETRIES

        while self.running:
            try:
                # Persistent connection check with recovery
                if not self.client.is_socket_open():
                    logger.info("Reconnecting to Modbus device...")
                    if not self.connect():
                        logger.error("Failed to reconnect, will retry...")
                        await asyncio.sleep(self.poll_interval)
                        continue

                logger.debug("Polling all parameters...")
                readings = self.read_all_parameters()

                if readings:
                    logger.debug(f"Logging {len(readings)} readings to database...")
                    db_success = await self.log_readings_to_database(readings)

                    if db_success:
                        logger.info(f"Successfully polled and logged {len(readings)} parameters")
                        consecutive_errors = 0
                    else:
                        logger.error("Failed to log some readings to database")
                        consecutive_errors += 1
                else:
                    logger.warning("No readings obtained from device")
                    consecutive_errors += 1

                # Exponential backoff on too many errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), resetting connection and backing off...")
                    # Mark device OFFLINE in DB
                    try:
                        db_helper.execute_query(
                            "UPDATE app.Analyzers SET ConnectionStatus = 'OFFLINE', UpdatedAt = GETUTCDATE() WHERE AnalyzerID = ?",
                            (self.device_id,)
                        )
                        db_helper.execute_query(
                            """
                            INSERT INTO ops.Events (AnalyzerID, Level, EventType, Message, Source)
                            VALUES (?, 'WARN', 'poller_offline', 'Device marked OFFLINE due to consecutive errors', 'Poller')
                            """,
                            (self.device_id,)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to mark analyzer {self.device_id} OFFLINE: {e}")
                    self.disconnect()
                    backoff = RETRY_BACKOFF_BASE * (2 ** (consecutive_errors - max_consecutive_errors))
                    logger.info(f"Backing off for {backoff:.1f} seconds...")
                    await asyncio.sleep(backoff)
                    consecutive_errors = 0
                    continue

            except Exception as e:
                logger.error(f"Unexpected error in polling loop: {e}")
                consecutive_errors += 1

            await asyncio.sleep(self.poll_interval)

    async def start(self):
        """Start the polling service"""
        self.running = True
        logger.info("Starting PAC3220 Modbus Poller Service")

        try:
            await self.run_polling_loop()
        except KeyboardInterrupt:
            logger.info("Polling stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in polling service: {e}")
        finally:
            self.disconnect()
            logger.info("Polling service stopped")

    def stop(self):
        """Stop the polling service"""
        logger.info("Stopping polling service...")
        self.running = False


async def poll_multiple_devices():
    """Poll all active devices from the database concurrently"""
    logger.info("Starting multi-device polling service")

    # Get poll interval from configuration
    try:
        config_result = db_helper.execute_query("SELECT ConfigValue FROM ops.Configuration WHERE ConfigKey = 'system.poller_interval'")
        poll_interval = int(config_result[0]["ConfigValue"]) if config_result else POLL_INTERVAL
    except Exception as e:
        logger.warning(f"Could not load poll interval from config, using default {POLL_INTERVAL}: {e}")
        poll_interval = POLL_INTERVAL

    # Get Modbus TCP port from configuration or environment
    default_port = 502
    try:
        port_cfg = db_helper.execute_query("SELECT ConfigValue FROM ops.Configuration WHERE ConfigKey = 'modbus.port'")
        configured_port = int(port_cfg[0]["ConfigValue"]) if port_cfg and port_cfg[0].get("ConfigValue") else None
    except Exception:
        configured_port = None
    env_port = None
    try:
        env_port = int(os.getenv("MODBUS_PORT", "0")) or None
    except Exception:
        env_port = None
    modbus_port_global = configured_port or env_port or default_port

    while True:
        try:
            # Query active analyzers from database using DAL (FIXED: No more direct pyodbc)
            devices_result = db_helper.execute_query(
                "SELECT AnalyzerID, SerialNumber, IPAddress, ModbusID FROM app.Analyzers WHERE IsActive = 1"
            ) or []

            if not devices_result:
                logger.warning("No active devices found in database")
                await asyncio.sleep(poll_interval)
                continue

            logger.info(f"Found {len(devices_result)} active devices to poll")

            # Create tasks for each device (isolate clients for thread-safety)
            tasks = []

            async def poll_device(device_row):
                device_id = device_row["AnalyzerID"]
                device_name = device_row["SerialNumber"] or f"Analyzer-{device_id}"
                host = device_row["IPAddress"]
                port = modbus_port_global  # Port resolved from configuration/environment
                unit_id = device_row["ModbusID"] or 1

                logger.info(f"Creating isolated poller for device {device_id} at {host}:{port}")
                # Create poller instance for this device (isolated client)
                poller = ModbusPoller(device_id, device_name, host, port, unit_id)
                # Poll and log readings
                success = await poller.poll_and_log_once()
                # Always disconnect after polling to free resources
                poller.disconnect()
                return success

            # Create concurrent tasks
            for device_row in devices_result:
                tasks.append(asyncio.create_task(poll_device(device_row)))

            # Run all polling tasks concurrently with timeout
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for r in results if r is True)
                error_count = sum(1 for r in results if isinstance(r, Exception) or r is False)
                logger.info(f"Polling cycle completed: {success_count} successful, {error_count} errors")
            except Exception as gather_error:
                logger.error(f"Error gathering polling results: {gather_error}")

        except Exception as e:
            logger.error(f"Error in multi-device polling: {e}", exc_info=True)

        # Reload poll interval from configuration before next cycle
        try:
            cfg = db_helper.execute_query("SELECT ConfigValue FROM ops.Configuration WHERE ConfigKey = 'system.poller_interval'")
            poll_interval = int(cfg[0]["ConfigValue"]) if cfg else poll_interval
        except Exception as e:
            logger.debug(f"Poll interval reload failed, keeping {poll_interval}: {e}")
        logger.info(f"Waiting {poll_interval} seconds for next poll cycle...")
        await asyncio.sleep(poll_interval)


async def main():
    """Main entry point for the poller service"""
    try:
        await poll_multiple_devices()
    except KeyboardInterrupt:
        print("\nMulti-device poller stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    print("PAC3220 Modbus Poller Service")
    print(f"Device: {DEVICE_NAME} (ID: {DEVICE_ID})")
    print(f"Modbus: {MODBUS_HOST}:{MODBUS_PORT} (Unit: {MODBUS_UNIT_ID})")
    print(f"Poll Interval: {POLL_INTERVAL} seconds")
    print("Mode: Multi-device polling (dynamically loads from database)")
    print("=" * 60)

    asyncio.run(main())
