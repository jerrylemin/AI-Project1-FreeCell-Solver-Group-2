"""Depth-First Search solver for FreeCell (iterative deepening)."""

from __future__ import annotations

from typing import List, Set

from game.moves import Move, apply_auto_moves, apply_move, get_valid_moves
from game.state import GameState
from .base import BaseSolver, SolverResult


_NOT_FOUND = "NOT_FOUND"
_CUTOFF = "CUTOFF"
_NODE_LIMIT = "NODE_LIMIT"
_STOPPED = "STOPPED"


class DFSSolver(BaseSolver):
    """FreeCell solver using iterative-deepening DFS."""

    MAX_DEPTH = 200

    @property
    def name(self) -> str:
        return "DFS (IDS)"

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

        self._total_expanded = 0
        self._total_generated = 1
        self._deepest_depth = 0
        self._aborted = None
        self._best_trace: List[Move] = []
        self._best_trace_length = 0
        self._best_score = (0, initial.cards_on_foundation, -self._hidden_blockers(initial))
        self._report_progress(
            moves=0,
            expanded_nodes=0,
            generated_nodes=self._total_generated,
            frontier_size=1,
            search_length=0,
            current_depth=0,
            status="Searching",
            force=True,
        )

        depth_limit = 0
        for depth_limit in range(1, self.MAX_DEPTH + 1):
            path_states: List[GameState] = [initial]
            path_moves: List[Move] = []
            path_keys: Set[tuple] = {initial.canonical_key()}

            outcome = self._dls(initial, depth_limit, path_states, path_moves, path_keys)

            if isinstance(outcome, list):
                return SolverResult(
                    self.name,
                    solved=True,
                    solution=outcome,
                    replay_trace=list(outcome),
                    best_trace_length=len(outcome),
                    replay_label="Replay Solution",
                    replay_message="Replaying final solution path.",
                    status=self._solved_status(),
                    expanded_nodes=self._total_expanded,
                    generated_nodes=self._total_generated,
                    frontier_size=0,
                    search_length=len(outcome),
                )

            if self._aborted or outcome == _NOT_FOUND:
                break

        reason = {
            _NODE_LIMIT: self._nodelimit_msg(),
            _STOPPED: self._stop_msg(),
            None: "No solution found within depth limit.",
        }.get(self._aborted, "No solution found.")

        return SolverResult(
            self.name,
            solved=False,
            replay_trace=list(self._best_trace),
            best_trace_length=self._best_trace_length,
            replay_label="Replay Failed Attempt",
            replay_message=f"{reason} Showing best-progress trace.",
            status={
                _NODE_LIMIT: self._nodelimit_status(),
                _STOPPED: self._stop_status(),
                None: self._failure_status(),
            }.get(self._aborted, self._failure_status()),
            expanded_nodes=self._total_expanded,
            generated_nodes=self._total_generated,
            frontier_size=0,
            search_length=max(self._deepest_depth, depth_limit),
            message=reason,
        )

    def _dls(
        self,
        state: GameState,
        limit: int,
        path_states: List[GameState],
        path_moves: List[Move],
        path_keys: Set[tuple],
    ):
        if self.stop_requested():
            self._aborted = _STOPPED
            return _STOPPED

        if self._total_expanded >= self.MAX_NODES:
            self._aborted = _NODE_LIMIT
            return _NODE_LIMIT

        if state.is_goal():
            return list(path_moves)

        if limit == 0:
            return _CUTOFF

        self._total_expanded += 1
        self._deepest_depth = max(self._deepest_depth, len(path_moves))
        self._maybe_yield()
        found_cutoff = False

        for move in get_valid_moves(state):
            if self.stop_requested():
                self._aborted = _STOPPED
                return _STOPPED

            child = apply_move(state, move)
            if self.use_auto_moves:
                child = apply_auto_moves(child)

            key = child.canonical_key()
            if key in path_keys:
                continue

            path_states.append(child)
            path_moves.append(move)
            path_keys.add(key)
            depth = len(path_moves)
            self._total_generated += 1
            self._deepest_depth = max(self._deepest_depth, depth)
            child_score = (depth, child.cards_on_foundation, -self._hidden_blockers(child))
            if child_score > self._best_score:
                self._best_score = child_score
                self._best_trace = list(path_moves)
                self._best_trace_length = max(self._best_trace_length, depth)
            self._report_progress(
                moves=self._best_trace_length,
                expanded_nodes=self._total_expanded,
                generated_nodes=self._total_generated,
                frontier_size=len(path_states),
                search_length=self._deepest_depth,
                current_depth=depth,
                status="Searching",
            )

            result = self._dls(child, limit - 1, path_states, path_moves, path_keys)

            path_states.pop()
            path_moves.pop()
            path_keys.discard(key)

            if isinstance(result, list):
                return result
            if result in (_NODE_LIMIT, _STOPPED):
                return result
            if result == _CUTOFF:
                found_cutoff = True

        return _CUTOFF if found_cutoff else _NOT_FOUND
