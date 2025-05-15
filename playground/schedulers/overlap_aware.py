from .base import Scheduler

class OverlapAware(Scheduler):
    def step(self):
        ready = sorted(self.jobs, key=lambda j: j.submit_time)
        while ready:
            job = ready[0]
            if self.env.now < job.submit_time:
                yield self.env.timeout(job.submit_time - self.env.now)
            batch = job.batches.pop(0)
            self.launch_batch(job, batch)
            if not job.batches:
                ready.pop(0)
            yield self.env.timeout(0)   # allow other procs to run
