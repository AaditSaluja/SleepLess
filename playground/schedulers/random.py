import random
from core.events import bus
from .base import Scheduler
from core.resources import ElasticCPU

class RandomScheduler(Scheduler):
    """
    RandomScheduler picks among all ready jobs at random and processes one batch
    end-to-end (CPU then GPU) before selecting the next batch—no overlap.
    """
    def __init__(self, env, cluster, jobs, metrics, **params):
        super().__init__(env, cluster, jobs, metrics, **params)

    def step(self):
        # Copy jobs into a pending list (submit_time, job)
        pending = [(j.submit_time, j) for j in self.jobs]

        while pending:
            now = self.env.now
            # Split pending into ready and future
            ready = [(t, j) for t, j in pending if t <= now]
            future = [(t, j) for t, j in pending if t > now]

            if not ready:
                # No job ready: advance to next arrival
                next_t = min(t for t, _ in future)
                yield self.env.timeout(next_t - now)
                pending = future
                continue

            # Pick a random ready job
            t_submit, job = random.choice(ready)
            pending.remove((t_submit, job))

            # Pop and process the next batch end-to-end
            batch = job.batches.pop(0)

            # CPU stage: single-core
            if isinstance(self.cluster.cpu, ElasticCPU):
                # always allocate one core
                yield self.cluster.cpu.request(1)
                yield self.env.timeout(batch.cpu_ms / 1000)
                self.cluster.cpu.release(1)
            else:
                with self.cluster.cpu.request() as cpu_res:
                    yield cpu_res
                    yield self.env.timeout(batch.cpu_ms / 1000)

            # GPU stage
            with self.cluster.gpu.request() as gpu_res:
                yield gpu_res
                yield self.env.timeout(batch.gpu_ms / 1000)

            # Notify metrics and trackers
            bus.publish("batch_done", job_id=job.jid, batch=batch)

            # If job still has more batches, requeue it
            if job.batches:
                pending.append((now, job))

            # Let other events proceed
            yield self.env.timeout(0)
