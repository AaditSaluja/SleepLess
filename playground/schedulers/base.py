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

    def launch_batch(self, job, batch, cpu_cores=1):
        """
        Split CPU and GPU stages into separate processes so they can overlap.
        """
        env, cl = self.env, self.cluster

        def cpu_proc():
            # CPU stage
            if isinstance(cl.cpu, ElasticCPU):
                yield cl.cpu.request(cpu_cores)
                actual_ms = batch.cpu_ms / cpu_cores
                yield env.timeout(actual_ms / 1000)
                cl.cpu.release(cpu_cores)
            else:
                with cl.cpu.request() as cpu:
                    yield cpu
                    yield env.timeout(batch.cpu_ms / 1000)
            # when CPU is done, launch GPU stage
            env.process(gpu_proc())

        def gpu_proc():
            # GPU stage
            with cl.gpu.request() as gpu:
                yield gpu
                yield env.timeout(batch.gpu_ms / 1000)
            # Launch metrics
            bus.publish("batch_done", job_id=job.jid, batch=batch)

        # start the CPU process immediately
        env.process(cpu_proc())