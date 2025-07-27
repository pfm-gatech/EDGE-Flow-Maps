import time
import taichi as ti

class SimpleTimer:
    _start_times = {}
    _accumulated_times = {}
    _counters = {}
    _recorded_counters = {}

    @staticmethod
    def set_logger(logger):
        SimpleTimer._logger = logger

    @staticmethod
    def start(key):
        """Start the timer for a specific key."""
        if key in SimpleTimer._start_times:
            raise ValueError(f"Timer for '{key}' is already running.")
        ti.sync()
        SimpleTimer._start_times[key] = time.time()
        SimpleTimer._counters[key] = SimpleTimer._counters.get(key, 0) + 1

    @staticmethod
    def stop(key, skip_count = 0):
        if key not in SimpleTimer._start_times:
            raise ValueError(f"Timer for '{key}' was not started.")
        ti.sync()
        elapsed = time.time() - SimpleTimer._start_times.pop(key)
        if skip_count >= SimpleTimer._counters[key]:
            return
        if key not in SimpleTimer._accumulated_times:
            SimpleTimer._accumulated_times[key] = 0.0
        SimpleTimer._accumulated_times[key] += elapsed
        SimpleTimer._recorded_counters[key] = SimpleTimer._recorded_counters.get(key, 0) + 1

    @staticmethod
    def print_times():
        """Print all accumulated times."""
        if not SimpleTimer._accumulated_times:
            if SimpleTimer._logger is not None:
                SimpleTimer._logger.info("No times recorded.")
            else:
                print("No times recorded.")
        else:
            for key, time in SimpleTimer._accumulated_times.items():
                cnt = SimpleTimer._recorded_counters[key]
                text = f"'{key}' execution time: {time:.8f} seconds, average: {time / cnt:.8f} seconds, count: {cnt}"
                if SimpleTimer._logger is not None:
                    SimpleTimer._logger.info(text)
                else:
                    print(text)
