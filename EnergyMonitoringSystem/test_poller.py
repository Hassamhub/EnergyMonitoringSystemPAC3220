
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add backend to path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.modbus_poller import ModbusPoller

class TestModbusPoller(unittest.TestCase):

    @patch('backend.modbus_poller.ModbusTcpClient')
    @patch('backend.modbus_poller.db_helper')
    def test_poll_and_log(self, mock_db_helper, mock_modbus_client):
        print("Testing polling and logging...")

        # Mock the Modbus client
        mock_client_instance = mock_modbus_client.return_value
        mock_client_instance.connect.return_value = True

        # Mock the read_input_registers method to return some dummy data
        def mock_read_input_registers(address, count, slave):
            response = MagicMock()
            response.isError.return_value = False
            # ModbusPoller subtracts 1 from the logical address, so mock sees 6 for 7, 12 for 13
            if address == 6: # voltage_l1 (VL1-N)
                response.registers = [17264, 0] # 240.0 IEEE-754 (0x43700000)
            elif address == 12: # current_l1
                response.registers = [16544, 0] # 5.0 IEEE-754 (0x40A00000)
            else:
                response.registers = [0, 0]
            return response
        mock_client_instance.read_input_registers.side_effect = mock_read_input_registers

        # Mock the db_helper
        mock_db_helper.execute_stored_procedure.return_value = [{'Status': 'Success'}]

        # Create an instance of the poller
        poller = ModbusPoller(
            device_id=1,
            device_name='Test-Device',
            host='localhost',
            port=502,
            unit_id=1
        )

        # Connect to the mock client
        poller.connect()

        # Call the read_all_parameters method
        readings = poller.read_all_parameters()
        print(f"Readings: {readings}")

        # Assert that the readings are what we expect
        self.assertIn('voltage_l1', readings)
        self.assertAlmostEqual(readings['voltage_l1'], 240.0, delta=1)
        self.assertIn('current_l1', readings)
        self.assertAlmostEqual(readings['current_l1'], 5.0, delta=0.1)

        # Call the log_readings_to_database method
        import asyncio
        asyncio.run(poller.log_readings_to_database(readings))

        # Assert that the stored procedure was called
        mock_db_helper.execute_stored_procedure.assert_called_once()
        args, kwargs = mock_db_helper.execute_stored_procedure.call_args
        self.assertEqual(args[0], 'app.sp_InsertReading')
        self.assertIn('@AnalyzerID', args[1])
        self.assertEqual(args[1]['@AnalyzerID'], 1)
        self.assertIn('@VL1', args[1])
        self.assertAlmostEqual(args[1]['@VL1'], 240.0, delta=1)
        self.assertIn('@IL1', args[1])
        self.assertAlmostEqual(args[1]['@IL1'], 5.0, delta=0.1)

        print("Test completed successfully!")

if __name__ == '__main__':
    unittest.main()
