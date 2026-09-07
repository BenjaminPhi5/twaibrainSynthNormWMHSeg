from collections import defaultdict

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory:
            value = self[key] = self.default_factory(key)
            return value
        else:
            raise KeyError(key)
