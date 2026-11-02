"""Wearable: bursty. Buffers HR/SpO2 at ~1 Hz and publishes them as a batch every 5-30 s,
with occasional disconnect/reconnect (modelled as a reboot)."""
import time

from .base import Device
from .monitor import vitals_source
from .noise import utcnow


class Wearable(Device):
    def __init__(self):
        super().__init__("wearable")
        self.source_iter, self.source = vitals_source(self.noise.rng)

    def batch(self):
        n = self.noise.rng.randint(5, 30)   # one reading per buffered second
        samples = []
        for _ in range(n):
            hr, spo2 = next(self.source_iter)
            samples.append({"hr": hr, "spo2": spo2})
        return {"device": "wearable", "bed": self.bed, "ts": utcnow(),
                "count": n, "vitals_source": self.source, "samples": samples}

    def run(self):
        self.start()
        topic = f"ward1/bed{self.bed}/wearable/vitals"
        while True:
            gap = self.noise.rng.uniform(5, 30)
            if self.noise.due_for_reboot(gap):
                self.reboot()
            batch = self.batch()
            # buffering delay: time spent collecting the batch before the burst
            time.sleep(min(gap, batch["count"]))
            self.publish(topic, batch)
            time.sleep(max(0.0, gap - batch["count"]))


def run():
    Wearable().run()
