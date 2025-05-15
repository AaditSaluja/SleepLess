import heapq
from core.events import bus
from core.resources import ElasticCPU
from .base import Scheduler

class FCFS(Scheduler):
    def __init__(self, env, cluster, jobs, metrics, **params):
        super().__init__(env, cluster, jobs, metrics, **params)

    def step(self):
        """
        Strictly sequence CPU→GPU for each batch before moving on to the next.
        """
        q = [(j.submit_time, j) for j in self.jobs]
        heapq.heapify(q)

        while q:
            submit_time, job = heapq.heappop(q)
            # wait until job arrival
            if self.env.now < submit_time:
                yield self.env.timeout(submit_time - self.env.now)

            # process each batch end-to-end
            for batch in job.batches:
                # CPU stage
                if isinstance(self.cluster.cpu, ElasticCPU):
                    yield self.cluster.cpu.request()
                    yield self.env.timeout(batch.cpu_ms / 1000)
                    self.cluster.cpu.release()
                else:
                    with self.cluster.cpu.request() as cpu:
                        yield cpu
                        yield self.env.timeout(batch.cpu_ms / 1000)

                # GPU stage
                with self.cluster.gpu.request() as gpu:
                    yield gpu
                    yield self.env.timeout(batch.gpu_ms / 1000)

                # signal completion
                bus.publish("batch_done", job_id=job.jid, batch=batch)

            # immediately proceed to next job