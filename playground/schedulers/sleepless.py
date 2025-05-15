import math
from .base import Scheduler
from core.resources import ElasticCPU

class SleepLess(Scheduler):
    """
    For each batch, allocate CPU cores so that
      cpu_time_per_batch / cores ≈ gpu_time_per_batch.
    """

    def __init__(self, env, cluster, jobs, metrics, **params):
        super().__init__(env, cluster, jobs, metrics, **params)

    def step(self):
        # jobs in submit‐time order
        ready = sorted(self.jobs, key=lambda j: j.submit_time)

        while ready:
            job = ready[0]
            if self.env.now < job.submit_time:
                yield self.env.timeout(job.submit_time - self.env.now)

            batch = job.batches.pop(0)

            # Determine total available cores (ElasticCPU only)
            if isinstance(self.cluster.cpu, ElasticCPU):
                total_cores = self.cluster.cpu.bucket.capacity
            else:
                total_cores = 1

            # target: batch.cpu_ms / cores ≈ batch.gpu_ms
            desired = math.ceil(batch.cpu_ms / batch.gpu_ms)
            # clamp between 1 and what we have
            cores = max(1, min(total_cores, desired))

            # launch with dynamic CPU allocation
            self.launch_batch(job, batch, cpu_cores=cores)

            # if job done, drop it
            if not job.batches:
                ready.pop(0)

            # let the event loop proceed
            yield self.env.timeout(0)
