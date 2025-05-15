import simpy
from abc import ABC, abstractmethod
from core.events import bus
from core.resources import ElasticCPU

class Scheduler(ABC):
    def __init__(self, env, cluster, jobs, metrics, **params):
        self.env = env
        self.cluster = cluster
        self.jobs = jobs
        self.metrics = metrics

    @abstractmethod
    def step(self):
        pass

    def launch_batch(self, job, batch, cpu_cores=1, alpha=1.0, overhead_ms=0.0):
        env, cl = self.env, self.cluster

        def cpu_proc():
            # grab the cores
            if isinstance(cl.cpu, ElasticCPU):
                yield cl.cpu.request(cpu_cores)
            else:
                # static CPU only gives you one “token” anyway
                yield cl.cpu.request()

            # compute the real ms from work units
            work   = batch.cpu_work
            actual = work / (cpu_cores ** alpha) + overhead_ms
            batch.actual_cpu_ms = actual

            yield env.timeout(actual / 1000.0)

            # release
            if isinstance(cl.cpu, ElasticCPU):
                cl.cpu.release(cpu_cores)

            # kick off GPU
            env.process(gpu_proc())

        def gpu_proc():
            # GPU stage
            with cl.gpu.request() as gpu:
                yield gpu
                yield env.timeout(batch.gpu_ms / 1000)
            # notify metrics and trackers
            bus.publish("batch_done", job_id=job.jid, batch=batch)

        # start the CPU process immediately
        env.process(cpu_proc())