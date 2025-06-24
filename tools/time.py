import time
from functools import wraps

__all__ = ['format_time', 'timer_decorator', 'ETATimer']


def format_time(seconds):
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Elapsed time for data sync: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")
        return result
    return wrapper


class ETATimer:
    def __init__(self, decay=0.95):
        self.decay = decay
        self.ema_time = None

    def update(self, batch_time):
        if self.ema_time is None:
            self.ema_time = batch_time
        else:
            self.ema_time = self.decay * batch_time + (1 - self.decay) * self.ema_time

    def get_remaining_time(self, remaining_batches):
        if self.ema_time is None:
            return "Calculating..."
        total_seconds = self.ema_time * remaining_batches
        return format_time(total_seconds)