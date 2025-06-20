from time import perf_counter
from collections import OrderedDict
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank


class Clock:
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
        self.total_time = 0.0
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise RuntimeError(f"Clock '{self.name}' is already running.")
        self._start_time = perf_counter()
        self.call_count += 1

    def stop(self):
        if self._start_time is None:
            raise RuntimeError(f"Clock '{self.name}' was not started.")
        elapsed = perf_counter() - self._start_time
        self.total_time += elapsed
        self._start_time = None

    def avg_time(self):
        return self.total_time / self.call_count if self.call_count > 0 else 0.0


class TimingManager:
    def __init__(self):
        self.clocks = OrderedDict()

    def start(self, name: str):
        clock = self.clocks.setdefault(name, Clock(name))
        clock.start()

    def stop(self, name: str):
        if name not in self.clocks:
            raise ValueError(f"No clock with name '{name}' was started.")
        self.clocks[name].stop()

    def report(self, header: str = "<global routines>"):
        if rank != 0:
            return
        print()
        print(f"{header:>10}")
        print(f"{'':13}clock number : {len(self.clocks):5}")
        print()
        for clock in self.clocks.values():
            time_s = clock.total_time
            calls = clock.call_count
            avg = clock.avg_time()
            if calls == 1:
                print(f"{clock.name:>20} : {time_s:8.2f}s CPU")
            else:
                print(
                    f"{clock.name:>20} : {time_s:8.2f}s CPU ({calls:8d} calls,{avg:8.3f} s avg)"
                )
        print()


global_timing = TimingManager()
