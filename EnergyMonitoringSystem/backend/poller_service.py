"""
PAC3220 Robust Modbus Poller Service
Polls Siemens PAC3220 energy analyzers with concurrent polling, validation, outbox consumer, and retry logic.
"""

import json
import time
import logging
import threading
import argparse
import os
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pymodbus.client import ModbusTcpClient
import pyodbc
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Fix encoding issues for Windows
try:
    import codecs
    codecs.register_error('replace_with_question', lambda e: (u'?', e.start + 1))
except:
    pass

# Configure logging with proper encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding='utf-8',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backend/logs/poller.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("poller")

# --------------------------
# Register Converters (from working code)
# --------------------------

def regs_to_float32(regs):
    """Convert 2 Modbus registers to float32 (Big-Endian)."""
    if len(regs) != 2:
        return None
    raw = struct.pack('>HH', regs[0], regs[1])
    return struct.unpack('>f', raw)[0]


def regs_to_float64(regs):
    """Convert 4 Modbus registers to float64 (Big-Endian)."""
    if len(regs) != 4:
        return None
    raw = struct.pack('>HHHH', regs[0], regs[1], regs[2], regs[3])
    return struct.unpack('>d', raw)[0]


# --------------------------
# Safe Input Register Reader (from working code)
# --------------------------

def read_input(client, address, count, datatype="float32"):
    """Reads input registers with retry + converter."""
    try:
        rr = client.read_input_registers(address=address, count=count, slave=1)

        if rr.isError():
            logger.error(f"❌ Modbus error at address {address}")
            return None

        regs = rr.registers
        if len(regs) != count:
            logger.error(f"❌ Incorrect register count at {address}: got {len(regs)}, expected {count}")
            return None

        if datatype == "float32":
            return regs_to_float32(regs)

        if datatype == "float64":
            return regs_to_float64(regs)

        return regs

    except Exception as e:
        logger.error(f"❌ Exception at address {address}: {e}")
        return None

# DB connection factory
DB_CONN_TEMPLATE = "DRIVER={{{driver}}};SERVER={server};DATABASE={db};UID={user};PWD={pwd}"
def get_db_conn(driver, server, db, user, pwd):
    conn_str = DB_CONN_TEMPLATE.format(driver=driver, server=server, db=db, user=user, pwd=pwd)
    return pyodbc.connect(conn_str, autocommit=False)

# Poll single device
def parse_registers(registers, mapping):
    """Parse raw register values using the register map configuration"""
    result = {}
    for key, spec in mapping.items():
        if spec["type"] == "input_register":
            addr = spec["address"]
            # Read 1 register for each mapping in this simplified example
            val = registers.get(addr, None)
            if val is None:
                result[key] = None
            else:
                result[key] = val * spec.get("scale", 1.0)
        elif spec["type"] == "coil":
            addr = spec["address"]
            result[key] = registers.get(f"coil_{addr}", None)
    return result

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30),
       retry=retry_if_exception_type((ConnectionError, OSError, Exception)))
def connect_client(host, port):
    """Connect to Modbus device with enhanced retry logic"""
    try:
        client = ModbusTcpClient(host, port=port, timeout=10)  # Increased timeout
        if not client.connect():
            raise ConnectionError(f"Cannot connect {host}:{port}")
        # Test connection with a simple read
        try:
            test_read = client.read_coils(0, 1, unit=1)
            if test_read.isError():
                client.close()
                raise ConnectionError(f"Connection test failed for {host}:{port}")
        except:
            client.close()
            raise
        return client
    except Exception as e:
        logger.warning(f"Connection attempt failed for {host}:{port}: {e}")
        raise

def read_device(host, port, unit=1):
    """
    Read PAC3220 parameters using the exact working register addresses
    Returns data dictionary and client for cleanup
    """
    client = connect_client(host, port)
    try:
        data = {}

        # Read parameters exactly as in the working code - print debug info
        print(f"\n---------------- PAC3220 LIVE DATA FROM {host} ----------------")

        data['avg_current_a'] = read_input(client, 61, 2) or 0.0
        print("Avg Current (A):", data['avg_current_a'])

        data['total_apparent_power'] = read_input(client, 63, 2) or 0.0
        print("Total Apparent Power:", data['total_apparent_power'])

        data['total_active_power'] = read_input(client, 65, 2) or 0.0
        print("Total Active Power:", data['total_active_power'])

        data['total_reactive_power'] = read_input(client, 67, 2) or 0.0
        print("Total Reactive Power:", data['total_reactive_power'])

        data['voltage_l1_l2'] = read_input(client, 7, 2) or 0.0
        print("Voltage L1-L2:", data['voltage_l1_l2'])

        data['voltage_l2_l3'] = read_input(client, 9, 2) or 0.0
        print("Voltage L2-L3:", data['voltage_l2_l3'])

        data['voltage_l3_l1'] = read_input(client, 11, 2) or 0.0
        print("Voltage L3-L1:", data['voltage_l3_l1'])

        data['current_l1'] = read_input(client, 13, 2) or 0.0
        print("Current L1:", data['current_l1'])

        data['current_l2'] = read_input(client, 15, 2) or 0.0
        print("Current L2:", data['current_l2'])

        data['current_l3'] = read_input(client, 17, 2) or 0.0
        print("Current L3:", data['current_l3'])

        data['frequency'] = read_input(client, 55, 2) or 0.0
        print("Frequency:", data['frequency'])

        data['total_pf'] = read_input(client, 69, 2) or 0.0
        print("Total PF:", data['total_pf'])

        # Energy readings (Wh to KWh conversion)
        energy_import_t1 = read_input(client, 801, 4, "float64") or 0.0
        data['kwh_import_t1'] = energy_import_t1 / 1000.0 if energy_import_t1 else 0.0
        print("Energy Import T1 (KWh):", data['kwh_import_t1'])

        energy_import_t2 = read_input(client, 805, 4, "float64") or 0.0
        data['kwh_import_t2'] = energy_import_t2 / 1000.0 if energy_import_t2 else 0.0
        print("Energy Import T2 (KWh):", data['kwh_import_t2'])

        energy_export_t1 = read_input(client, 809, 4, "float64") or 0.0
        data['kwh_export_t1'] = energy_export_t1 / 1000.0 if energy_export_t1 else 0.0
        print("Energy Export T1 (KWh):", data['kwh_export_t1'])

        energy_export_t2 = read_input(client, 813, 4, "float64") or 0.0
        data['kwh_export_t2'] = energy_export_t2 / 1000.0 if energy_export_t2 else 0.0
        print("Energy Export T2 (KWh):", data['kwh_export_t2'])

        # Total energy (sum of all tariffs)
        data['kwh_total'] = data['kwh_import_t1'] + data['kwh_import_t2'] + data['kwh_export_t1'] + data['kwh_export_t2']
        print(f"Total Energy (KWh): {data['kwh_total']:.3f}")

        logger.info(f"PAC3220 data: KWh Total={data['kwh_total']:.3f}, Power={data['total_active_power']:.3f} KW")

        return data, client

    except Exception as e:
        logger.error(f"Failed to read device {host}: {e}")
        client.close()
        raise

def compute_energy_delta(prev, now, interval):
    """
    Compute energy delta and validate against power readings
    interval in seconds
    """
    prev_e = prev.get("energy_kwh_total") if prev else None
    now_e = now.get("energy_kwh_total")
    if prev_e is None or now_e is None:
        return None, None
    delta = now_e - prev_e
    if delta < 0:
        # Treat negative as rollover - ignore for now
        return None, "rollover"

    expected = now.get("power_kw_total", 0.0) * (interval / 3600.0)
    rel_err = abs(delta - expected) / max(1e-9, expected)
    return {"measured": delta, "expected": expected, "rel_err": rel_err}, None

def validate_measurement_accuracy(analyzer_id, rel_err, conn, consecutive_warnings=None):
    """
    Validate measurement accuracy and track persistent issues
    """
    WARNING_THRESHOLD = 0.02  # 2% relative error
    CRITICAL_THRESHOLD = 0.05  # 5% relative error
    CONSECUTIVE_CRITICAL = 3   # Alert after 3 consecutive critical readings

    # Initialize persistent tracking if not provided
    if consecutive_warnings is None:
        consecutive_warnings = getattr(validate_measurement_accuracy, f'_consecutive_{analyzer_id}', 0)

    issue_logged = False

    if rel_err > CRITICAL_THRESHOLD:
        consecutive_warnings += 1
        setattr(validate_measurement_accuracy, f'_consecutive_{analyzer_id}', consecutive_warnings)

        if consecutive_warnings >= CONSECUTIVE_CRITICAL:
            # Log persistent critical issue
            message = f"Persistent measurement inaccuracy: {rel_err:.3f} relative error for {consecutive_warnings} consecutive readings"
            log_measurement_issue(conn, analyzer_id, "PersistentInaccuracy", message)
            issue_logged = True
            logger.warning("Analyzer %s: %s", analyzer_id, message)
        else:
            logger.warning("Analyzer %s: High measurement error %.3f (attempt %d/%d)",
                         analyzer_id, rel_err, consecutive_warnings, CONSECUTIVE_CRITICAL)

    elif rel_err > WARNING_THRESHOLD:
        # Reset consecutive counter on warning-level issues
        setattr(validate_measurement_accuracy, f'_consecutive_{analyzer_id}', 0)
        logger.warning("Analyzer %s: Measurement warning: %.3f relative error", analyzer_id, rel_err)

    else:
        # Good reading - reset counter
        setattr(validate_measurement_accuracy, f'_consecutive_{analyzer_id}', 0)

    return consecutive_warnings, issue_logged

def log_measurement_issue(conn, analyzer_id, issue_type, message):
    """Log or update measurement issue in database"""
    try:
        cur = conn.cursor()

        # Check if issue already exists
        cur.execute("""
            SELECT IssueId, Count FROM dbo.MeasurementIssues
            WHERE AnalyzerId = ? AND IssueType = ? AND LastSeen >= DATEADD(HOUR, -24, GETUTCDATE())
        """, analyzer_id, issue_type)

        existing = cur.fetchone()

        if existing:
            # Update existing issue
            issue_id, count = existing
            cur.execute("""
                UPDATE dbo.MeasurementIssues
                SET Count = ?, LastSeen = GETUTCDATETIME(), Message = ?
                WHERE IssueId = ?
            """, count + 1, message, issue_id)
        else:
            # Create new issue
            cur.execute("""
                INSERT INTO dbo.MeasurementIssues (AnalyzerId, IssueType, Message)
                VALUES (?, ?, ?)
            """, analyzer_id, issue_type, message)

        conn.commit()
        cur.close()

    except Exception as e:
        logger.exception("Failed to log measurement issue: %s", e)

def insert_reading_db(cursor, analyzer_id, power_kw_total, energy_kwh_total):
    cursor.execute(
        "EXEC app.sp_InsertReading @AnalyzerID=?, @KW_Total=?, @KWh_Total=?, @Quality=?",
        analyzer_id, power_kw_total, energy_kwh_total, "GOOD"
    )

# Outbox consumer: reads ActuatorOutbox table and executes DO writes
def consume_outbox(db_conn, driver, server, db, user, pwd, max_batch=10, max_attempts_per_command=3):
    conn = db_conn
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT TOP (?) OutboxId, AnalyzerId, DesiredState, Attempts
            FROM dbo.ActuatorOutbox
            WHERE Status='Pending'
            ORDER BY CreatedAt ASC
        """, max_batch)
        rows = cur.fetchall()

        for out in rows:
            outbox_id, analyzer_id, desired_state, attempts = out.OutboxId, out.AnalyzerId, out.DesiredState, out.Attempts

            # Skip if too many attempts already
            if attempts >= max_attempts_per_command:
                logger.warning("Outbox %s exceeded max attempts (%d), marking as Fail", outbox_id, max_attempts_per_command)
                cur.execute("UPDATE dbo.ActuatorOutbox SET Status='Fail', LastError=? WHERE OutboxId=?",
                           f"ExceededMaxAttempts({max_attempts_per_command})", outbox_id)
                # Create alert for admin
                cur.execute("""
                    INSERT INTO dbo.Alerts (AnalyzerId, AlertType, Message)
                    VALUES (?, 'OutboxCommandFailed', ?)
                """, analyzer_id, f"DO command failed after {max_attempts_per_command} attempts - requires manual intervention")
                conn.commit()
                continue

            # Lookup analyzer ip/port/unit
            cur.execute("SELECT IpAddress, ModbusUnitId FROM app.Devices WHERE DeviceID=?", analyzer_id)
            a = cur.fetchone()
            if not a:
                logger.error("Analyzer not found for outbox %s", outbox_id)
                cur.execute("UPDATE dbo.ActuatorOutbox SET Status='Fail', LastError=? WHERE OutboxId=?",
                           "AnalyzerNotFound", outbox_id)
                conn.commit()
                continue

            ip = a.IpAddress
            port = 502  # Change if your devices use diff port
            unit = a.ModbusUnitId or 1

            success = False
            try:
                # Attempt write with confirmation
                success = write_coil_with_confirmation(ip, port, REGISTER_MAP['coil_do_control']['address'],
                                                      bool(desired_state), unit=unit)
            except Exception as e:
                logger.exception("Write exception for outbox %s", outbox_id)
                cur.execute("UPDATE dbo.ActuatorOutbox SET Attempts=Attempts+1, LastError=? WHERE OutboxId=?",
                           str(e), outbox_id)
                conn.commit()
                continue

            if success:
                logger.info("Outbox command %s completed successfully - DO set to %s", outbox_id, desired_state)
                cur.execute("UPDATE dbo.ActuatorOutbox SET Status='Done', CompletedAt=GETUTCDATE() WHERE OutboxId=?", outbox_id)
            else:
                logger.warning("Outbox command %s write failed, incrementing attempts", outbox_id)
                cur.execute("UPDATE dbo.ActuatorOutbox SET Attempts=Attempts+1 WHERE OutboxId=?", outbox_id)
            conn.commit()

    finally:
        cur.close()

def write_coil_with_confirmation(host, port, coil_addr, desired_state, unit=1, max_attempts=3):
    """
    Write coil with read-back confirmation and safety checks

    Args:
        host: Device IP address
        port: Device port
        coil_addr: Coil address to write
        desired_state: Boolean state to set (True=ON, False=OFF)
        unit: Modbus unit ID
        max_attempts: Maximum write attempts

    Returns:
        bool: True if write confirmed, False otherwise
    """
    client = ModbusTcpClient(host, port=port, timeout=3)

    # Safety check: ensure we can connect
    if not client.connect():
        logger.error("Cannot connect to device %s:%s for DO write", host, port)
        return False

    try:
        # Pre-write read to verify current state (safety check)
        pre_read = client.read_coils(coil_addr, 1, unit=unit)
        if not pre_read.isError():
            current_state = pre_read.bits[0]
            logger.debug("Pre-write state for %s:%s coil %s: %s", host, port, coil_addr, current_state)
        else:
            logger.warning("Could not read pre-write state for %s:%s coil %s", host, port, coil_addr)

        # Attempt write with confirmation
        for attempt in range(1, max_attempts + 1):
            logger.debug("Write attempt %d/%d for %s:%s coil %s -> %s", attempt, max_attempts, host, port, coil_addr, desired_state)

            # Write the coil
            wr = client.write_coil(coil_addr, desired_state, unit=unit)
            if wr.isError():
                logger.warning("Write coil attempt %d failed: %s", attempt, wr)
                if attempt < max_attempts:
                    time.sleep(0.5 * attempt)  # Exponential backoff
                continue

            # Confirm by reading back
            time.sleep(0.1)  # Brief delay for device to process
            rr = client.read_coils(coil_addr, 1, unit=unit)
            if rr.isError():
                logger.warning("Read back failed attempt %d: %s", attempt, rr)
                if attempt < max_attempts:
                    time.sleep(0.5 * attempt)
                continue

            # Check if read matches desired state
            if rr.bits[0] == desired_state:
                logger.info("✅ Write confirmed for %s:%s coil %s -> %s (attempt %d)", host, port, coil_addr, desired_state, attempt)
                client.close()
                return True
            else:
                logger.warning("Readback mismatch attempt %d: expected=%s, read=%s", attempt, desired_state, rr.bits[0])
                if attempt < max_attempts:
                    time.sleep(0.5 * attempt)

        # All attempts failed
        logger.error("[ERROR] Write failed after %d attempts for %s:%s coil %s -> %s", max_attempts, host, port, coil_addr, desired_state)
        client.close()
        return False

    except Exception as e:
        logger.exception("Unexpected error during coil write to %s:%s: %s", host, port, e)
        return False

    finally:
        try:
            client.close()
        except:
            pass

def poll_worker(analyzer, db_cfg, interval):
    """Worker function for polling a single analyzer"""
    analyzer_id = analyzer['AnalyzerId']
    host = analyzer['IpAddress']
    unit = analyzer.get('ModbusUnitId', 1)
    last_energy = None

    conn = get_db_conn(**db_cfg)
    cur = conn.cursor()

    while True:
        start = time.time()

        try:
            parsed, client = read_device(host, 502, unit)
            now_ts = datetime.utcnow().replace(microsecond=0)

            power_kw = parsed.get("power_kw_total", 0.0)
            energy_kwh = parsed.get("energy_kwh_total", 0.0)
            coil = parsed.get("coil_do_control", 0)

            # Data validation and quality checks
            if not validate_reading_data(power_kw, energy_kwh, analyzer_id):
                logger.warning("Invalid data from analyzer %s, skipping insertion", analyzer_id)
                consecutive_failures += 1
                client.close()
                continue

            consecutive_failures = 0  # Reset on successful read

            # Energy delta check and validation
            delta_info, status = compute_energy_delta(
                {"energy_kwh_total": last_energy} if last_energy is not None else None,
                {"energy_kwh_total": energy_kwh, "power_kw_total": power_kw},
                interval
            )

            # Validate measurement accuracy and track issues
            if delta_info:
                consecutive_warnings, issue_logged = validate_measurement_accuracy(
                    analyzer_id, delta_info['rel_err'], conn
                )

                # Log detailed information for monitoring
                if delta_info['rel_err'] > 0.02:
                    logger.debug("Analyzer %s accuracy check: measured=%.3f kWh, expected=%.3f kWh, rel_err=%.3f",
                               analyzer_id, delta_info['measured'], delta_info['expected'], delta_info['rel_err'])

            # Insert via stored procedure
            try:
                insert_reading_db(cur, analyzer_id, now_ts, power_kw, energy_kwh, coil)
                conn.commit()
                logger.debug("Inserted reading for analyzer %s", analyzer_id)
            except Exception as e:
                logger.exception("DB insert failed for analyzer %s", analyzer_id)
                conn.rollback()

            last_energy = energy_kwh
            client.close()

        except Exception as e:
            consecutive_failures += 1
            logger.exception("Polling failed for analyzer %s (attempt %d/%d): %s",
                           analyzer_id, consecutive_failures, max_consecutive_failures, e)

            # Exponential backoff on failures
            if consecutive_failures >= max_consecutive_failures:
                backoff_interval = min(interval * (2 ** min(consecutive_failures - max_consecutive_failures + 1, 5)), 300)  # Max 5 minutes
                logger.warning("Analyzer %s failed %d times, backing off for %d seconds",
                             analyzer_id, consecutive_failures, backoff_interval)
                time.sleep(backoff_interval)
                continue

        # Honor interval with jitter to avoid thundering herd
        elapsed = time.time() - start
        base_sleep = max(0, interval - elapsed)
        # Add small random jitter (±10%) to prevent synchronized polling
        import random
        jitter = base_sleep * 0.1 * (random.random() * 2 - 1)
        to_sleep = max(0, base_sleep + jitter)
        time.sleep(to_sleep)

def start_pollers(db_cfg, poll_interval, max_workers=20):
    """Start concurrent polling for all analyzers"""
    # Fetch analyzers list from DB
    conn = get_db_conn(**db_cfg)
    cur = conn.cursor()
    cur.execute("SELECT AnalyzerId, IpAddress, ModbusUnitId FROM dbo.Analyzers")
    analyzers = [dict(AnalyzerId=r.AnalyzerId, IpAddress=r.IpAddress, ModbusUnitId=r.ModbusUnitId)
                 for r in cur.fetchall()]
    cur.close()
    conn.close()

    logger.info("Starting pollers for %d analyzers", len(analyzers))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for a in analyzers:
            futures.append(ex.submit(poll_worker, a, db_cfg, poll_interval))

        # Run indefinitely
        try:
            for f in as_completed(futures):
                f.result()
        except KeyboardInterrupt:
            logger.info("Shutting down pollers")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAC3220 Modbus Poller Service")
    parser.add_argument("--db-driver", default="ODBC Driver 17 for SQL Server")
    parser.add_argument("--db-server", default="localhost")
    parser.add_argument("--db-name", default="PAC3220DB")
    parser.add_argument("--db-user", default="sa")
    parser.add_argument("--db-pass", default="YourStrongPassword")
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=20)
    args = parser.parse_args()

    db_cfg = dict(driver=args.db_driver, server=args.db_server, db=args.db_name,
                  user=args.db_user, pwd=args.db_pass)

    # Start outbox consumer thread
    outbox_conn = get_db_conn(**db_cfg)
    def outbox_loop():
        while True:
            try:
                consume_outbox(outbox_conn, **db_cfg)
            except Exception:
                logger.exception("Outbox consumer exception")
            time.sleep(2)

    t = threading.Thread(target=outbox_loop, daemon=True)
    t.start()

    start_pollers(db_cfg, args.poll_interval, args.max_workers)