from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import collections


@dataclass
class LRUCache:
    capacity_bytes: int
    _store: Dict[Tuple[str, int], Any] = field(default_factory=dict)
    _order: "collections.OrderedDict[Tuple[str, int], int]" = field(default_factory=collections.OrderedDict)
    _approx_bytes: int = 0

    def get(self, key: Tuple[str, int]) -> Optional[Any]:
        if key in self._store:
            self._order.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: Tuple[str, int], value: Any, size_bytes: int) -> None:
        self._store[key] = value
        self._order[key] = 1
        self._approx_bytes += size_bytes
        # naive eviction by approx size
        while self._approx_bytes > self.capacity_bytes and self._order:
            k, _ = self._order.popitem(last=False)
            v = self._store.pop(k, None)
            # size unknown; decrement a rough fraction
            self._approx_bytes = max(0, self._approx_bytes - size_bytes)


@dataclass
class CheckpointRing:
    k: int
    _store: Dict[Tuple[str, int], Any] = field(default_factory=dict)

    def key(self, stage_id: str, t_idx: int) -> Tuple[str, int]:
        return (stage_id, t_idx % self.k)

    def save(self, stage_id: str, t_idx: int, payload: Any) -> None:
        self._store[self.key(stage_id, t_idx)] = payload

    def load(self, stage_id: str, t_idx: int) -> Optional[Any]:
        return self._store.get(self.key(stage_id, t_idx))
