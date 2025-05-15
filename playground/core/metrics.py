# # from .events import bus

# # class Metrics:
# #     def __init__(self, env, cluster):
# #         self.env, self.cluster = env, cluster
# #         self.batch_times = []
# #         bus.subscribe("batch_done", self._on_batch)

# #     def _on_batch(self, job_id, batch):
# #         self.batch_times.append(batch)

# #     def gpu_util(self):
# #         busy = self.cluster.gpu_busy()
# #         return busy / self.cluster.gpu.capacity if busy else 0.0


# # core/metrics.py
# """
# Metrics collection for batch processing, including GPU utilization.
# """

# from core.events import bus

# class Metrics:
#     def __init__(self, env, cluster):
#         # Simulation environment and cluster resource tracker
#         self.env = env
#         self.cluster = cluster
#         # Record per-batch timing details
#         self.batch_times = []
#         # Accumulate total GPU busy time in milliseconds
#         self.total_gpu_ms = 0.0
#         # Subscribe to batch completion events
#         bus.subscribe("batch_done", self._on_batch)

#     def _on_batch(self, job_id, batch):
#         # Track individual batch metrics
#         self.batch_times.append(batch)
#         # Add this batch's GPU service time (ms)
#         self.total_gpu_ms += batch.gpu_ms

#     def gpu_util(self):
#         """
#         Compute time-averaged GPU utilization over the entire simulation run.

#         Returns:
#             float: fraction of GPU capacity used (0.0 - 1.0)
#         """
#         # Total simulation time in seconds
#         sim_time = self.env.now
#         if sim_time <= 0:
#             return 0.0

#         # Convert accumulated GPU busy time from ms to seconds
#         busy_time_s = self.total_gpu_ms / 1000.0
#         # Number of GPUs in the cluster
#         cap = self.cluster.gpu.capacity

#         # Fraction of available GPU-seconds consumed
#         return busy_time_s / (cap * sim_time)


# core/metrics.py
"""
Metrics collection for batch processing, including CPU/GPU utilization and idle times.
"""

from core.events import bus

class Metrics:
    def __init__(self, env, cluster):
        # Simulation environment and cluster resource tracker
        self.env = env
        self.cluster = cluster
        # Record per-batch timing details
        self.batch_times = []
        # Accumulate total busy time in milliseconds
        self.total_gpu_ms = 0.0
        self.total_cpu_ms = 0.0
        # Subscribe to batch completion events
        bus.subscribe("batch_done", self._on_batch)

    def _on_batch(self, job_id, batch):
        # Track individual batch metrics
        self.batch_times.append(batch)
        # Add this batch's CPU and GPU service times (ms)
        self.total_cpu_ms += batch.cpu_ms
        self.total_gpu_ms += batch.gpu_ms

    def _sim_time(self):
        # Total simulation time in seconds
        return self.env.now

    def _utilization(self, busy_ms, capacity):
        sim_time = self._sim_time()
        if sim_time <= 0 or capacity <= 0:
            return 0.0
        busy_s = busy_ms / 1000.0
        return busy_s / (capacity * sim_time)

    def cpu_util(self):
        """
        Fraction of CPU capacity utilized over the run.
        """
        # Handle ElasticCPU vs. static CPU
        if hasattr(self.cluster.cpu, 'bucket'):
            cap = self.cluster.cpu.bucket.capacity
        else:
            cap = self.cluster.cpu.capacity
        return self._utilization(self.total_cpu_ms, cap)

    def gpu_util(self):
        """
        Fraction of GPU capacity utilized over the run.
        """
        cap = self.cluster.gpu.capacity
        return self._utilization(self.total_gpu_ms, cap)

    def cpu_idle_time(self):
        """
        Total CPU idle time (in seconds) over the run.
        """
        sim_time = self._sim_time()
        if hasattr(self.cluster.cpu, 'bucket'):
            cap = self.cluster.cpu.bucket.capacity
        else:
            cap = self.cluster.cpu.capacity
        busy_s = self.total_cpu_ms / 1000.0
        return max(0.0, cap * sim_time - busy_s)

    def gpu_idle_time(self):
        """
        Total GPU idle time (in seconds) over the run.
        """
        sim_time = self._sim_time()
        cap = self.cluster.gpu.capacity
        busy_s = self.total_gpu_ms / 1000.0
        return max(0.0, cap * sim_time - busy_s)
