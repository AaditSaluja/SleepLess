import simpy, random

__all__ = ["make_env"]

def make_env(seed: int = 0) -> simpy.Environment:
    # random.seed(seed)
    return simpy.Environment()
