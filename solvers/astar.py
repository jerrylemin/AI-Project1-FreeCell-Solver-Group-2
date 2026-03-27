"""A* Search solver for FreeCell."""

from __future__ import annotations

import heapq
from typing import Dict, Optional, Tuple

from game.moves import Move, apply_auto_moves, apply_move, get_valid_moves
from game.state import GameState
from .base import BaseSolver, SolverResult


def heuristic(state: GameState) -> float:
    foundations = state.foundations
    cascades = state.cascades

    h1 = 52 - sum(foundations)
    h2 = 0

    for suit in range(4):
        needed_rank = foundations[suit] + 1
        if needed_rank > 13:
            continue

        for fc in state.free_cells:
            if fc is not None and fc.rank == needed_rank and fc.suit == suit:
                break
        else:
            for col in cascades:
                for depth_from_top, card in enumerate(reversed(col)):
                    if card.rank == needed_rank and card.suit == suit:
                        h2 += depth_from_top
                        break
                else:
                    continue
                break

    h3 = (4 - state.empty_free_cells) * 0.5
    return h1 + h2 + h3


class AStarSolver(BaseSolver):
    """FreeCell solver using A* Search."""

    @property
    def name(self) -> str:
        return "A*"

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

        counter = 0
        initial_h = heuristic(initial)
        frontier = [(initial_h, counter, 0, initial)]
        best_g: Dict[tuple, float] = {initial.canonical_key(): 0}
        best_depth: Dict[tuple, int] = {initial.canonical_key(): 0}
        parent: Dict[tuple, Tuple[Optional[GameState], Optional[Move]]] = {
            initial.canonical_key(): (None, None)
        }
        expanded = 0
        generated = 1
        deepest_depth = 0
        best_state = initial
        best_score = (initial.cards_on_foundation, -initial_h, -initial_h, 0)
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

            _f, _, g, state = heapq.heappop(frontier)
            key = state.canonical_key()
            if g > best_g.get(key, float('inf')):
                continue

            expanded += 1
            depth = best_depth[key]
            deepest_depth = max(deepest_depth, depth)
            self._maybe_yield()

            if state.is_goal():
                solution = self._reconstruct(parent, state)
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

                child_key = child.canonical_key()
                new_g = g + 1
                new_depth = depth + 1

                if new_g < best_g.get(child_key, float('inf')):
                    best_g[child_key] = new_g
                    best_depth[child_key] = new_depth
                    parent[child_key] = (state, move)
                    counter += 1
                    generated += 1
                    deepest_depth = max(deepest_depth, new_depth)
                    child_h = heuristic(child)
                    child_f = new_g + child_h
                    heapq.heappush(frontier, (child_f, counter, new_g, child))
                    child_score = (child.cards_on_foundation, -child_h, -child_f, new_depth)
                    if child_score > best_score:
                        best_score = child_score
                        best_state = child
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
