import asyncio
import os
import yaml
from dotenv import load_dotenv
from poller.utils.logger import setup_logger
from poller.db.connection import SqlConnection
from poller.db.analyzers_repo import AnalyzersRepo
from poller.db.readings_repo import ReadingsRepo
from poller.device_loader import DeviceLoader
from poller.scheduler import Scheduler
from poller.data_mapper import DataMapper
from poller.poller import DevicePoller

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

async def main():
    load_dotenv()
    cfg = load_config('config.yaml')
    # Override config with environment when provided
    cfg['database']['driver'] = os.getenv('DB_DRIVER', cfg['database']['driver'])
    cfg['database']['server'] = os.getenv('DB_SERVER', cfg['database']['server'])
    cfg['database']['database'] = os.getenv('DB_NAME', cfg['database']['database'])
    cfg['modbus']['port'] = int(os.getenv('MODBUS_PORT', cfg['modbus']['port']))
    _timeout_env = os.getenv('MODBUS_TIMEOUT') or os.getenv('MODBUS_TIMEOUT_SECONDS')
    if _timeout_env is not None:
        try:
            cfg['modbus']['timeout_seconds'] = int(_timeout_env)
        except Exception:
            pass
    cfg['modbus']['retries'] = int(os.getenv('MODBUS_RETRIES', cfg['modbus']['retries']))
    _poll_env = os.getenv('MODBUS_POLL_INTERVAL') or os.getenv('POLL_INTERVAL_SECONDS')
    if _poll_env is not None:
        try:
            cfg['modbus']['poll_interval_seconds'] = int(_poll_env)
        except Exception:
            pass
    cfg['modbus']['max_concurrency'] = int(os.getenv('MAX_CONCURRENCY', cfg['modbus']['max_concurrency']))
    cfg['scheduler']['refresh_devices_seconds'] = int(os.getenv('SCHEDULER_REFRESH_SECONDS', cfg['scheduler']['refresh_devices_seconds']))
    log_level = os.getenv('LOG_LEVEL', cfg['logging']['level'])
    log_file = os.getenv('LOG_FILE', cfg['logging']['file'])
    logger = setup_logger('poller', level=log_level, file_path=log_file)
    sql = SqlConnection(cfg)
    analyzers_repo = AnalyzersRepo(sql)
    readings_repo = ReadingsRepo(sql)
    mapper = DataMapper(cfg['registers'])
    loader = DeviceLoader(analyzers_repo, cfg['scheduler']['refresh_devices_seconds'])

    def make_poller(device):
        return DevicePoller(device, cfg, readings_repo, analyzers_repo, mapper)

    scheduler = Scheduler(loader, make_poller, cfg['modbus']['max_concurrency'])
    logger.info('Starting scheduler')
    await scheduler.start()

if __name__ == '__main__':
    asyncio.run(main())