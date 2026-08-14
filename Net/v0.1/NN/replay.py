from __future__ import annotations

import random


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._data: list[dict] = []
        self._next = 0

    def extend(self, samples: list[dict]) -> None:
        for sample in samples:
            if len(self._data) < self.capacity:
                self._data.append(sample)
            else:
                self._data[self._next] = sample
            self._next = (self._next + 1) % self.capacity

    def sample(self, batch_size: int) -> list[dict]:
        if not self._data:
            return []
        count = min(int(batch_size), len(self._data))
        indices = random.sample(range(len(self._data)), count)
        return [self._data[index] for index in indices]

    def __len__(self) -> int:
        return len(self._data)
