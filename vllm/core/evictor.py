# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    SLIDE = enum.auto()  # <-- [SLIDE] add

class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float,
                 is_dirty: bool = False,
                 selected_token_count: int = 0):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        # [SLIDE] metadata
        self.is_dirty = is_dirty
        self.selected_token_count = selected_token_count


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float,
            is_dirty: bool = False,
            selected_token_count: int = 0):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  is_dirty=is_dirty,
                                                  selected_token_count=selected_token_count
                                                  )
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.SLIDE:
        return SLIDEEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")

class SLIDEEvictor(Evictor):
    """SLIDE dirty-page eviction:
    1) Evict clean pages first (LRU among clean).
    2) If clean pages are exhausted, rank dirty pages by selected-token count,
       evicting in descending order (ties: older first).
    """

    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.clean_heap: List[Tuple[float, int, int, int]] = []
        self.dirty_heap: List[Tuple[int, float, int, int]] = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, is_dirty: bool = False,
            selected_token_count: int = 0):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  is_dirty=is_dirty,
                                                  selected_token_count=selected_token_count)
        if is_dirty:
            # higher selected_token_count first => use negative for max-heap effect
            heapq.heappush(self.dirty_heap,
                           (-selected_token_count, last_accessed, block_id, content_hash))
        else:
            heapq.heappush(self.clean_heap,
                           (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        meta = self.free_table[block_id]
        meta.last_accessed = last_accessed
        # lazy update: push a new entry, old ones will be skipped in evict()
        if meta.is_dirty:
            heapq.heappush(self.dirty_heap,
                           (-meta.selected_token_count, meta.last_accessed, block_id, meta.content_hash))
        else:
            heapq.heappush(self.clean_heap,
                           (meta.last_accessed, -meta.num_hashed_tokens, block_id, meta.content_hash))
        self._cleanup_if_necessary()

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        # 1) clean first (LRU)
        while self.clean_heap:
            last_accessed, _, block_id, content_hash = heapq.heappop(self.clean_heap)
            meta = self.free_table.get(block_id)
            if meta is None:
                continue
            if (not meta.is_dirty) and meta.last_accessed == last_accessed:
                self.free_table.pop(block_id, None)
                return block_id, content_hash

        # 2) dirty by selected-token count desc
        while self.dirty_heap:
            neg_cnt, last_accessed, block_id, content_hash = heapq.heappop(self.dirty_heap)
            meta = self.free_table.get(block_id)
            if meta is None:
                continue
            if meta.is_dirty and meta.last_accessed == last_accessed and (-neg_cnt) == meta.selected_token_count:
                self.free_table.pop(block_id, None)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError("Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    def _cleanup_if_necessary(self):
        if (len(self.clean_heap) + len(self.dirty_heap)) > SLIDEEvictor.CLEANUP_THRESHOLD * max(1, len(self.free_table)):
            self._cleanup()

    def _cleanup(self):
        new_clean: List[Tuple[float, int, int, int]] = []
        new_dirty: List[Tuple[int, float, int, int]] = []
        for block_id, meta in self.free_table.items():
            if meta.is_dirty:
                new_dirty.append((-meta.selected_token_count, meta.last_accessed, block_id, meta.content_hash))
            else:
                new_clean.append((meta.last_accessed, -meta.num_hashed_tokens, block_id, meta.content_hash))
        heapq.heapify(new_clean)
        heapq.heapify(new_dirty)
        self.clean_heap = new_clean
        self.dirty_heap = new_dirty

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)
