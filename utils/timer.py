import time
import logging


class Timer:
    def __init__(self, warmup=0):
        self.warmup = warmup
        self.times = []
        self.count = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        end = time.perf_counter()
        duration = end - self.start
        if self.count >= self.warmup:
            self.times.append(duration)
        self.count += 1

    def avg_time(self):
        if not self.times:
            return 0
        return sum(self.times) / len(self.times)

    def report(self):
        # logger = logging.getLogger(__name__)
        print(f"Executed {len(self.times)} times (after warmup={self.warmup})")
        print(f"Average time: {self.avg_time():.6f} seconds")
