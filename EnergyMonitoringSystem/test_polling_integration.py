
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.modbus_poller import ModbusPoller

# Load environment variables
load_dotenv()

# Configuration from environment
DEVICE_ID = int(os.getenv("DEVICE_ID", "3"))
DEVICE_NAME = os.getenv("DEVICE_NAME", "PAC3220-001")
MODBUS_HOST = os.getenv("MODBUS_HOST", "192.168.10.2")
MODBUS_PORT = int(os.getenv("MODBUS_PORT", "502"))
MODBUS_UNIT_ID = int(os.getenv("MODBUS_UNIT_ID", "1"))

async def main():
    """
    This script tests the Modbus poller by connecting to a real device,
    polling it once, and printing the readings.
    """
    print("--- Modbus Polling Integration Test ---")
    print(f"Device: {DEVICE_NAME} (ID: {DEVICE_ID})")
    print(f"Modbus Host: {MODBUS_HOST}:{MODBUS_PORT} (Unit: {MODBUS_UNIT_ID})")
    
    # Create a poller instance
    poller = ModbusPoller(
        device_id=DEVICE_ID,
        device_name=DEVICE_NAME,
        host=MODBUS_HOST,
        port=MODBUS_PORT,
        unit_id=MODBUS_UNIT_ID
    )

    # Poll the device once
    success = await poller.poll_and_log_once()

    if success:
        print("--- Test Result: SUCCESS ---")
    else:
        print("--- Test Result: FAILED ---")

if __name__ == "__main__":
    asyncio.run(main())
