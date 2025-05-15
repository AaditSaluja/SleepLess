from dataclasses import dataclass, field

@dataclass
class Batch:
    cpu_ms: float
    gpu_ms: float

@dataclass
class JobCfg:
    jid: int
    submit_time: float
    batches: list[Batch]                 # list length == epochs × batches/epoch
