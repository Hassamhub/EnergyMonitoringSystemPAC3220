#!/usr/bin/env python3
"""
Real PAC3220 Device Testing Script
Tests actual Modbus communication with physical device at 192.168.10.2
"""

import asyncio
import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.modbus_poller import ModbusPoller
from backend.dal.database import db_helper

pytestmark = pytest.mark.skip(reason="Requires real PAC3220 device and database connection")

async def test_real_device():
    print('🔌 Testing real PAC3220 device at 192.168.10.2:502')
    print('=' * 60)

    # Create poller for real device using env config
    poller = ModbusPoller(
        device_id=1,
        device_name='PAC3220-REAL-TEST',
        host=os.getenv('MODBUS_IP', '192.168.1.100'),
        port=int(os.getenv('MODBUS_PORT', '502')),
        unit_id=int(os.getenv('MODBUS_ID', '1'))
    )

    try:
        # Test 1: Connection
        print('📡 Test 1: Modbus TCP Connection...')
        if poller.connect(retries=2):
            print('✅ Connection successful!')
        else:
            print('❌ Connection failed - device not reachable')
            return

        # Test 2: Parameter Reading
        print('\n📊 Test 2: Reading electrical parameters...')
        readings = poller.read_all_parameters()

        if readings and len(readings) > 0:
            print('✅ Parameters read successfully!')
            print(f'   📈 Total readings: {len(readings)}')

            # Show key values
            print('   📋 Sample values:')
            count = 0
            for key, value in readings.items():
                if count >= 5:
                    break
                if value is not None:
                    print(f'      {key}: {value:.2f}')
                else:
                    print(f'      {key}: NULL (failed to read)')
                count += 1
        else:
            print('❌ Failed to read parameters')
            return

        # Test 3: Database Logging
        print('\n💾 Test 3: Database logging with delta calculation...')
        success = await poller.log_readings_to_database(readings)

        if success:
            print('✅ Database logging successful!')

            # Check if reading was inserted
            latest_reading = db_helper.execute_query(
                'SELECT TOP 1 * FROM app.Readings ORDER BY ReadingID DESC'
            )

            if latest_reading:
                reading = latest_reading[0]
                print(f'   📝 Reading ID: {reading["ReadingID"]}')
                print(f'   ⚡ KW Total: {reading["KW_Total"]}')
                print(f'   🔋 KWh Total: {reading["KWh_Total"]}')
                print(f'   📊 Delta KWh: {reading["DeltaKWh"]}')
                if reading['DeltaKWh'] is not None:
                    print('   ✅ Delta calculation working!')
                else:
                    print('   ❌ Delta calculation failed')
        else:
            print('❌ Database logging failed')

        # Test 4: Event Logging
        print('\n📋 Test 4: Event logging validation...')
        events = db_helper.execute_query(
            'SELECT TOP 3 * FROM ops.Events ORDER BY EventID DESC'
        )

        if events:
            print('✅ Events logged successfully!')
            for event in events[:2]:
                msg = event["Message"][:50] if event["Message"] else "No message"
                print(f'   📝 {event["Level"]}: {msg}...')
        else:
            print('❌ Event logging failed')

        print('\n🎉 ALL TESTS COMPLETED!')
        print('=' * 60)

    except Exception as e:
        print(f'❌ Test error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        poller.disconnect()
        print('🔌 Connection closed')

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_real_device())