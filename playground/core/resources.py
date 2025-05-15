import simpy

class Stage(simpy.Resource):
    """Fixed‑capacity stage (CPU preprocess or GPU compute)."""
    pass

class ElasticCPU:
    """A resizable pool of CPU tokens."""
    def __init__(self, env, initial):
        self.env = env
        self.bucket = simpy.Container(env, capacity=initial, init=initial)

    def request(self, n=1):
        return self.bucket.get(n)

    def release(self, n=1):
        self.bucket.put(n)

    # exposed to schedulers
    def grow(self, n): self.bucket.capacity += n; self.bucket.put(n)
    def shrink(self, n): self.bucket.capacity -= n
