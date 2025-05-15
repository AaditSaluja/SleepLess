from .resources import Stage, ElasticCPU

class Cluster:
    def __init__(self, env, gpus: int, cpu_cores: int, elastic=False):
        self.env   = env
        self.gpu   = Stage(env, capacity=gpus)
        self.cpu   = ElasticCPU(env, cpu_cores) if elastic else Stage(env, capacity=cpu_cores)

    # helpers
    def gpu_busy(self): return self.gpu.count
    def cpu_busy(self):
        return (self.cpu.bucket.capacity - self.cpu.bucket.level
                if isinstance(self.cpu, ElasticCPU) else self.cpu.count)
