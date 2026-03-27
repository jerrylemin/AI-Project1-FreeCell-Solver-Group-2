"""Base solver infrastructure shared by all search algorithms."""

from __future__ import annotations

import threading
import time
import tracemalloc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from game.moves import Move
from game.state import GameState


@dataclass
class SolverResult:
    algorithm: str
    solved: bool
    solution: List[Move] = field(default_factory=list)
    replay_trace: Optional[List[Move]] = None
    best_trace_length: int = 0
    replay_label: str = "Replay Solution"
    replay_message: str = ""
    status: str = ""
    search_time: float = 0.0
    memory_kb: float = 0.0
    expanded_nodes: int = 0
    generated_nodes: int = 0
    frontier_size: int = 0
    search_length: int = 0
    current_depth: int = 0
    message: str = ""

    @property
    def solution_length(self) -> int:
        return len(self.solution)

    @property
    def replay_available(self) -> bool:
        return self.replay_trace is not None

    @property
    def replay_length(self) -> int:
        return len(self.replay_trace or [])

    @property
    def display_moves(self) -> int:
        if self.best_trace_length > 0:
            return self.best_trace_length
        if self.solved:
            return self.solution_length
        return self.replay_length

    def summary(self) -> str:
        status = self.status or ("Solved" if self.solved else "Failed, no solution found")
        move_label = "Solution length" if self.solved else "Best trace length"
        lines = [
            f"Algorithm    : {self.algorithm}",
            f"Status       : {status}",
            f"Moves        : {self.display_moves} ({move_label})",
            f"Solution len : {self.solution_length}",
            f"Replay length: {self.replay_length}",
            f"Time         : {self.search_time:.3f} s",
            f"Memory       : {self.memory_kb:.1f} KB",
            f"Nodes expanded: {self.expanded_nodes:,}",
            f"Nodes generated: {self.generated_nodes:,}",
            f"Frontier size : {self.frontier_size:,}",
            f"Search length : {self.search_length:,}",
        ]
        if self.message:
            lines.append(f"Note         : {self.message}")
        if self.replay_message:
            lines.append(f"Replay       : {self.replay_message}")
        return "\n".join(lines)


@dataclass
class ProgressSnapshot:
    algorithm: str
    status: str = "Searching"
    moves: int = 0
    elapsed_time: float = 0.0
    memory_kb: float = 0.0
    expanded_nodes: int = 0
    generated_nodes: int = 0
    frontier_size: int = 0
    search_length: int = 0
    current_depth: int = 0
    done: bool = False
    solved: bool = False
    message: str = ""
    result: Optional[SolverResult] = None


SolverProgress = ProgressSnapshot


class BaseSolver(ABC):
    """Abstract base for all FreeCell solvers."""

    MAX_NODES = 150_000
    PROGRESS_INTERVAL_S = 0.05
    MEMORY_INTERVAL_S = 0.20
    WORKER_YIELD_INTERVAL_S = 0.01

    def __init__(
        self,
        use_auto_moves: bool = True,
        progress_callback: Optional[Callable[[ProgressSnapshot], None]] = None,
    ):
        self.use_auto_moves = use_auto_moves
        self.progress_callback = progress_callback
        self._solve_started_at = 0.0
        self._last_progress_at = 0.0
        self._last_memory_at = 0.0
        self._last_yield_at = 0.0
        self._last_memory_kb = 0.0
        self._stop_event = threading.Event()

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _search(self, initial: GameState) -> SolverResult:
        """Subclasses implement the actual search here."""
        ...

    def solve(self, initial: GameState) -> SolverResult:
        """Public entry-point: wraps _search with timing and memory tracking."""
        tracemalloc.start()
        t0 = time.perf_counter()
        self._solve_started_at = t0
        self._last_progress_at = 0.0
        self._last_memory_at = 0.0
        self._last_yield_at = t0
        self._last_memory_kb = 0.0
        self._stop_event.clear()

        result = self._search(initial)

        result.search_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.memory_kb = peak / 1024
        result.algorithm = self.name
        if result.replay_trace is None:
            result.replay_trace = list(result.solution)
        if result.best_trace_length <= 0:
            result.best_trace_length = (
                result.solution_length if result.solved else result.replay_length
            )
        if result.solved:
            result.replay_label = "Replay Solution"
        elif not result.replay_label:
            result.replay_label = "Replay Failed Attempt"
        if not result.status:
            result.status = "Solved" if result.solved else "Failed, no solution found"
        return result

    def _reconstruct(self, parent: dict, state: GameState) -> List[Move]:
        """Walk the parent dict backwards to recover the move sequence."""
        path: List[Move] = []
        cur = state
        while parent[cur.canonical_key()][1] is not None:
            _, move = parent[cur.canonical_key()]
            path.append(move)
            cur = parent[cur.canonical_key()][0]
        path.reverse()
        return path

    def _sample_memory_kb(self, *, force: bool = False) -> float:
        now = time.perf_counter()
        if force or now - self._last_memory_at >= self.MEMORY_INTERVAL_S:
            _, peak = tracemalloc.get_traced_memory()
            self._last_memory_at = now
            self._last_memory_kb = peak / 1024
        return self._last_memory_kb

    def _maybe_yield(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if force or now - self._last_yield_at >= self.WORKER_YIELD_INTERVAL_S:
            self._last_yield_at = now
            time.sleep(0)

    def _report_progress(
        self,
        *,
        moves: int = 0,
        expanded_nodes: int = 0,
        generated_nodes: int = 0,
        frontier_size: int = 0,
        search_length: int = 0,
        current_depth: int = 0,
        status: str = "Searching",
        message: str = "",
        done: bool = False,
        solved: bool = False,
        force: bool = False,
    ) -> None:
        if self.progress_callback is None:
            return

        now = time.perf_counter()
        if not force and now - self._last_progress_at < self.PROGRESS_INTERVAL_S:
            return

        self._last_progress_at = now
        self.progress_callback(
            ProgressSnapshot(
                algorithm=self.name,
                status=status,
                moves=moves,
                elapsed_time=now - self._solve_started_at,
                memory_kb=self._sample_memory_kb(force=force),
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                frontier_size=frontier_size,
                search_length=search_length,
                current_depth=current_depth,
                done=done,
                solved=solved,
                message=message,
            )
        )

    def _hidden_blockers(self, state: GameState) -> int:
        blockers = 0
        for suit in range(4):
            needed_rank = state.foundations[suit] + 1
            if needed_rank > 13:
                continue

            for free_card in state.free_cells:
                if free_card is not None and free_card.rank == needed_rank and free_card.suit == suit:
                    break
            else:
                for column in state.cascades:
                    for depth_from_top, card in enumerate(reversed(column)):
                        if card.rank == needed_rank and card.suit == suit:
                            blockers += depth_from_top
                            break
                    else:
                        continue
                    break

        return blockers

    def _progress_score(self, state: GameState) -> tuple[int, int]:
        return (state.cards_on_foundation, -self._hidden_blockers(state))

    def _solved_status(self) -> str:
        return "Solved"

    def _failure_status(self) -> str:
        return "Failed, no solution found"

    def _nodelimit_status(self) -> str:
        return "Node limit reached"

    def _stop_status(self) -> str:
        return "Stopped by user"

    def _nodelimit_msg(self) -> str:
        return f"Search stopped after {self.MAX_NODES:,} expanded nodes."

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _stop_msg(self) -> str:
        return "Search stopped by user."
