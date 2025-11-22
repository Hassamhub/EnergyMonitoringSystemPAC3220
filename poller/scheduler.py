import asyncio
from typing import Dict

class Scheduler:
    def __init__(self, device_loader, poller_factory, max_concurrency: int):
        self.device_loader = device_loader
        self.poller_factory = poller_factory
        self.max_concurrency = max_concurrency
        self.tasks: Dict[int, asyncio.Task] = {}

    async def start(self):
        asyncio.create_task(self.device_loader.start())
        sem = asyncio.Semaphore(self.max_concurrency)
        while True:
            devices = self.device_loader.list()
            current_ids = set(self.tasks.keys())
            new_ids = set([d["AnalyzerID"] for d in devices])
            for did in current_ids - new_ids:
                t = self.tasks.pop(did, None)
                if t:
                    t.cancel()
            for d in devices:
                did = d["AnalyzerID"]
                if did not in self.tasks:
                    poller = self.poller_factory(d)
                    async def run_device(p):
                        async with sem:
                            await p.run()
                    self.tasks[did] = asyncio.create_task(run_device(poller))
            await asyncio.sleep(5)