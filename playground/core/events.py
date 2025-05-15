from collections import defaultdict

class Bus:
    def __init__(self): self._subs = defaultdict(list)
    def subscribe(self, name, cb): self._subs[name].append(cb)
    def publish(self, name, **payload):
        for fn in self._subs[name]: fn(**payload)

bus = Bus()
