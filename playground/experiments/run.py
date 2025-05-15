import yaml, importlib
import inspect
from pathlib import Path

from core import env as core_env
from core.cluster import Cluster
from core.metrics import Metrics
from core.events  import bus
from workloads import synthetic
from schedulers.base import Scheduler as BaseScheduler
import random


# def load_scheduler(name):
#     mod = importlib.import_module(f"schedulers.{name}")
#     return next(v for v in vars(mod).values()
#                 if isinstance(v, type) and issubclass(v, mod.base.Scheduler))

def load_scheduler(name: str):
    """
    Return the unique subclass of schedulers.base.Scheduler that
    lives in module  `schedulers.{name}`.
    """
    mod = importlib.import_module(f"schedulers.{name}")
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if issubclass(cls, BaseScheduler) and cls is not BaseScheduler:
            return cls
    raise ValueError(f"No Scheduler subclass found in schedulers.{name}")

def main(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    n = random.randint(1, 99999)
    env      = core_env.make_env(seed=cfg.get("seed", n))
    print(env, n)
    cluster  = Cluster(env, **cfg["cluster"], elastic=cfg["scheduler"]["name"]=="elastic_cpu")
    jobs     = getattr(synthetic, cfg["workload"]["generator"])(**cfg["workload"]["params"])
    metrics  = Metrics(env, cluster)
    schedCls = load_scheduler(cfg["scheduler"]["name"])
    scheduler= schedCls(env, cluster, jobs, metrics, **cfg["scheduler"].get("params", {}))
    env.process(scheduler.step())
    env.run()                 # blocks until all processes finish
    sim_time_s = env.now
    print(f"Simulation time:       {sim_time_s:6.2f} s")
    print(f"Total CPU busy time:   {metrics.total_cpu_ms/1000.0:6.2f} s")
    print(f"Total GPU busy time:   {metrics.total_gpu_ms/1000.0:6.2f} s")
    print(f"CPU util:              {metrics.cpu_util()*100:6.2f} %")
    print(f"GPU util:              {metrics.gpu_util()*100:6.2f} %")
    print(f"CPU idle time:         {metrics.cpu_idle_time():6.2f} s")
    print(f"GPU idle time:         {metrics.gpu_idle_time():6.2f} s")
 

if __name__ == "__main__":
    import sys; main(sys.argv[1])
