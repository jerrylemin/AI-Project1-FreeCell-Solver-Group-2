"""Breadth-First Search solver for FreeCell."""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

from game.moves import Move, apply_auto_moves, apply_move, get_valid_moves
from game.state import GameState
from .base import BaseSolver, SolverResult


class BFSSolver(BaseSolver):
    """FreeCell solver using Breadth-First Search."""

    @property
    def name(self) -> str:
        return "BFS"

    def _search(self, initial: GameState) -> SolverResult:
        if self.use_auto_moves:
            initial = apply_auto_moves(initial)

        if initial.is_goal():
            return SolverResult(
                self.name,
                solved=True,
                solution=[],
                replay_trace=[],
                status=self._solved_status(),
                replay_message="Replaying final solution path.",
            )

        parent: Dict[tuple, Tuple[Optional[GameState], Optional[Move]]] = {
            initial.canonical_key(): (None, None)
        }
        frontier: deque[Tuple[GameState, int]] = deque([(initial, 0)])
        expanded = 0
        generated = 1
        deepest_depth = 0
        best_state = initial
        best_score = (*self._progress_score(initial), 0)
        best_trace_length = 0

        self._report_progress(
            moves=0,
            expanded_nodes=0,
            generated_nodes=generated,
            frontier_size=1,
            search_length=0,
            current_depth=0,
            status="Searching",
            force=True,
        )

        while frontier:
            if self.stop_requested():
                best_trace = self._reconstruct(parent, best_state)
                return SolverResult(
                    self.name,
                    solved=False,
                    replay_trace=best_trace,
                    best_trace_length=best_trace_length,
                    replay_label="Replay Failed Attempt",
                    replay_message="Stopped by user, showing best-progress trace.",
                    status=self._stop_status(),
                    expanded_nodes=expanded,
                    generated_nodes=generated,
                    frontier_size=len(frontier),
                    search_length=deepest_depth,
                    message=self._stop_msg(),
                )

            if expanded >= self.MAX_NODES:
                best_trace = self._reconstruct(parent, best_state)
                return SolverResult(
                    self.name,
                    solved=False,
                    replay_trace=best_trace,
                    best_trace_length=best_trace_length,
                    replay_label="Replay Failed Attempt",
                    replay_message="Node limit reached, showing best-progress trace.",
                    status=self._nodelimit_status(),
                    expanded_nodes=expanded,
                    generated_nodes=generated,
                    frontier_size=len(frontier),
                    search_length=deepest_depth,
                    message=self._nodelimit_msg(),
                )

            state, depth = frontier.popleft()
            expanded += 1
            deepest_depth = max(deepest_depth, depth)
            self._maybe_yield()

            for move in get_valid_moves(state):
                if self.stop_requested():
                    best_trace = self._reconstruct(parent, best_state)
                    return SolverResult(
                        self.name,
                        solved=False,
                        replay_trace=best_trace,
                        best_trace_length=best_trace_length,
                        replay_label="Replay Failed Attempt",
                        replay_message="Stopped by user, showing best-progress trace.",
                        status=self._stop_status(),
                        expanded_nodes=expanded,
                        generated_nodes=generated,
                        frontier_size=len(frontier),
                        search_length=deepest_depth,
                        message=self._stop_msg(),
                    )

                child = apply_move(state, move)
                if self.use_auto_moves:
                    child = apply_auto_moves(child)

                key = child.canonical_key()
                if key in parent:
                    continue

                parent[key] = (state, move)
                child_depth = depth + 1
                deepest_depth = max(deepest_depth, child_depth)
                generated += 1
                frontier.append((child, child_depth))
                child_score = (*self._progress_score(child), child_depth)
                if child_score > best_score:
                    best_score = child_score
                    best_state = child
                    best_trace_length = max(best_trace_length, child_depth)
                self._report_progress(
                    moves=best_trace_length,
                    expanded_nodes=expanded,
                    generated_nodes=generated,
                    frontier_size=len(frontier),
                    search_length=deepest_depth,
                    current_depth=child_depth,
                    status="Searching",
                )

                if child.is_goal():
                    solution = self._reconstruct(parent, child)
                    return SolverResult(
                        self.name,
                        solved=True,
                        solution=solution,
                        replay_trace=list(solution),
                        best_trace_length=len(solution),
                        replay_label="Replay Solution",
                        replay_message="Replaying final solution path.",
                        status=self._solved_status(),
                        expanded_nodes=expanded,
                        generated_nodes=generated,
                        frontier_size=len(frontier),
                        search_length=len(solution),
                        current_depth=child_depth,
                    )

        best_trace = self._reconstruct(parent, best_state)
        return SolverResult(
            self.name,
            solved=False,
            replay_trace=best_trace,
            best_trace_length=best_trace_length,
            replay_label="Replay Failed Attempt",
            replay_message="No solution found, showing best-progress trace.",
            status=self._failure_status(),
            expanded_nodes=expanded,
            generated_nodes=generated,
            frontier_size=0,
            search_length=deepest_depth,
            message="No solution exists.",
        )
