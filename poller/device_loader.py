import asyncio
from typing import Dict, Any, List

class DeviceLoader:
    def __init__(self, analyzers_repo, refresh_seconds: int):
        self.repo = analyzers_repo
        self.refresh_seconds = refresh_seconds
        self._devices: List[Dict[str, Any]] = []

    async def start(self):
        while True:
            self._devices = self.repo.get_active_analyzers()
            await asyncio.sleep(self.refresh_seconds)

    def list(self) -> List[Dict[str, Any]]:
        return list(self._devices)