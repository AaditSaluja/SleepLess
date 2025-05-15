# implements the alpha scaling. The base.py corresponding to it is sleepless2_base, need to plug that in to run that

from collections import deque
import math
from .base import Scheduler
from core.events import bus
from core.resources import ElasticCPU

class SleepLess(Scheduler):
    """
    Dynamically allocate CPU cores per batch to balance CPU and GPU stages,
    and adapt when extra cores no longer yield improvements.
    """
    def __init__(self, env, cluster, jobs, metrics,
                 alpha=1.0, window=10, threshold=0.05, overhead_ms=0.0, **params):
        super().__init__(env, cluster, jobs, metrics, **params)
        self.alpha       = alpha
        self.window      = window
        self.threshold   = threshold
        self.overhead_ms = overhead_ms
        # Determine total CPU cores from cluster
        if isinstance(cluster.cpu, ElasticCPU):
            total = cluster.cpu.bucket.capacity
        else:
            total = cluster.cpu.capacity
        # History buffer for each core count
        self.history = {c: deque(maxlen=window) for c in range(1, total+1)}
        # Subscribe to record actual CPU times
        bus.subscribe("batch_done", self._record)

    def step(self):
        # Sort jobs by submit time
        queue = sorted(self.jobs, key=lambda j: j.submit_time)
        while queue:
            job = queue[0]
            # Wait until job arrives
            if self.env.now < job.submit_time:
                yield self.env.timeout(job.submit_time - self.env.now)

            # Remove job if no batches left
            if not job.batches:
                queue.pop(0)
                continue

            batch = job.batches.pop(0)

            # Compute desired cores to equalize CPU vs GPU time
            desired = math.ceil(batch.cpu_work / batch.gpu_ms)
            # Clamp to available cores
            if isinstance(self.cluster.cpu, ElasticCPU):
                cap = self.cluster.cpu.bucket.capacity
            else:
                cap = self.cluster.cpu.capacity
            cores = max(1, min(cap, desired))

            # Check marginal speedup from history
            hist = self.history.get(cores, [])
            if len(hist) == self.window and cores > 1:
                avg_curr = sum(hist) / len(hist)
                prev_hist = self.history.get(cores-1, [])
                if prev_hist:
                    avg_prev = sum(prev_hist) / len(prev_hist)
                    speedup = (avg_prev - avg_curr) / avg_prev
                    if speedup < self.threshold:
                        cores -= 1

            # Launch with non-linear scaling parameters
            self.launch_batch(
                job, batch,
                cpu_cores=cores,
                alpha=self.alpha,
                overhead_ms=self.overhead_ms
            )
            # Yield to allow event processing
            yield self.env.timeout(0)

    def _record(self, job_id, batch):
        # Record actual CPU time into history based on used cores
        cores = getattr(batch, 'used_cores', None)
        actual = getattr(batch, 'actual_cpu_ms', None)
        if cores and actual is not None and cores in self.history:
            self.history[cores].append(actual)
