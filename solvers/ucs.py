"""Uniform-Cost Search solver for FreeCell."""

from __future__ import annotations

import heapq
from typing import Dict, Tuple

from game.moves import Move, apply_auto_moves, apply_move
from game.state import GameState
from .base import BaseSolver, SolverResult
from .search_utils import (
    exact_state_key,
    move_order_key,
    ordered_legal_moves,
    reconstruct_path,
    should_prune_immediate_reverse,
)


def move_cost(move: Move) -> int:
    if move.dst_type == "foundation":
        return 0
    if move.dst_type == "freecell":
        return 3
    return 1


class UCSSolver(BaseSolver):
    """FreeCell solver using Uniform-Cost Search."""

    @property
    def name(self) -> str:
        return "UCS"

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

        move_cache = {}
        initial_key = exact_state_key(initial)
        counter = 0
        frontier = [(0, (0,), counter, initial, initial_key)]
        best_g: Dict[tuple, int] = {initial_key: 0}
        best_depth: Dict[tuple, int] = {initial_key: 0}
        parent_links = {initial_key: (None, None)}
        expanded = 0
        generated = 1
        deepest_depth = 0
        best_state_key = initial_key
        best_score = (initial.cards_on_foundation, -self._hidden_blockers(initial), 0, 0)
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
                return self._failed_result(
                    parent_links,
                    best_state_key,
                    best_trace_length,
                    expanded,
                    generated,
                    len(frontier),
                    deepest_depth,
                    self._stop_status(),
                    self._stop_msg(),
                    "Stopped by user, showing best-progress trace.",
                )

            if expanded >= self.MAX_NODES:
                return self._failed_result(
                    parent_links,
                    best_state_key,
                    best_trace_length,
                    expanded,
                    generated,
                    len(frontier),
                    deepest_depth,
                    self._nodelimit_status(),
                    self._nodelimit_msg(),
                    "Node limit reached, showing best-progress trace.",
                )

            g, _tie_break, _counter, state, state_key = heapq.heappop(frontier)
            if g > best_g.get(state_key, float("inf")):
                continue

            expanded += 1
            depth = best_depth[state_key]
            deepest_depth = max(deepest_depth, depth)
            self._maybe_yield()

            if state.is_goal():
                solution = reconstruct_path(parent_links, state_key)
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
                    current_depth=depth,
                )

            previous_move = parent_links[state_key][1]
            for move in ordered_legal_moves(state, move_cache):
                if self.stop_requested():
                    return self._failed_result(
                        parent_links,
                        best_state_key,
                        best_trace_length,
                        expanded,
                        generated,
                        len(frontier),
                        deepest_depth,
                        self._stop_status(),
                        self._stop_msg(),
                        "Stopped by user, showing best-progress trace.",
                    )

                if should_prune_immediate_reverse(
                    move,
                    previous_move,
                    use_auto_moves=self.use_auto_moves,
                ):
                    continue

                child = apply_move(state, move)
                if self.use_auto_moves:
                    child = apply_auto_moves(child)

                child_key = exact_state_key(child)
                new_g = g + move_cost(move)
                new_depth = depth + 1
                old_g = best_g.get(child_key, float("inf"))
                if new_g >= old_g:
                    continue

                best_g[child_key] = new_g
                best_depth[child_key] = new_depth
                parent_links[child_key] = (state_key, move)
                counter += 1
                generated += 1
                deepest_depth = max(deepest_depth, new_depth)

                tie_break = tuple(-part for part in move_order_key(state, move))
                heapq.heappush(frontier, (new_g, tie_break, counter, child, child_key))

                child_score = (
                    child.cards_on_foundation,
                    -self._hidden_blockers(child),
                    -new_g,
                    new_depth,
                )
                if child_score > best_score:
                    best_score = child_score
                    best_state_key = child_key
                    best_trace_length = max(best_trace_length, new_depth)

                self._report_progress(
                    moves=best_trace_length,
                    expanded_nodes=expanded,
                    generated_nodes=generated,
                    frontier_size=len(frontier),
                    search_length=deepest_depth,
                    current_depth=new_depth,
                    status="Searching",
                )

        return self._failed_result(
            parent_links,
            best_state_key,
            best_trace_length,
            expanded,
            generated,
            0,
            deepest_depth,
            self._failure_status(),
            "No solution exists.",
            "No solution found, showing best-progress trace.",
        )

    def _failed_result(
        self,
        parent_links,
        best_state_key,
        best_trace_length,
        expanded,
        generated,
        frontier_size,
        deepest_depth,
        status,
        message,
        replay_message,
    ) -> SolverResult:
        best_trace = reconstruct_path(parent_links, best_state_key)
        return SolverResult(
            self.name,
            solved=False,
            replay_trace=best_trace,
            best_trace_length=best_trace_length,
            replay_label="Replay Failed Attempt",
            replay_message=replay_message,
            status=status,
            expanded_nodes=expanded,
            generated_nodes=generated,
            frontier_size=frontier_size,
            search_length=deepest_depth,
            message=message,
        )
