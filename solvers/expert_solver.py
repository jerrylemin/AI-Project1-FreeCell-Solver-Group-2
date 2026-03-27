"""High-power practical FreeCell solver.

This solver is intentionally tuned for practical deal coverage rather than
formal move-count optimality. It uses a weighted best-first search on top of
the shared legal move generator and immutable state transitions used by the
rest of the app.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from game.card import Card
from game.moves import Move, apply_move, get_valid_moves, is_safe_to_auto_move
from game.state import GameState
from .base import BaseSolver, SolverResult


# Search weights. These are kept near the top so the practical tuning is easy
# to read and adjust without digging through the search loop.
FOUNDATION_REMAINING_WEIGHT = 5.4
LOW_RANK_BURIED_WEIGHT = 2.6
NEXT_FOUNDATION_BLOCKED_WEIGHT = 3.8
TRAPPED_TARGET_WEIGHT = 1.9
EMPTY_FREE_CELL_WEIGHT = 2.2
EMPTY_CASCADE_WEIGHT = 6.4
MOVABLE_SEQUENCE_WEIGHT = 1.4
EXPOSED_LOW_CARD_WEIGHT = 2.5
MOBILITY_WEIGHT = 1.2

SAFE_FOUNDATION_ORDER_BONUS = 7
FOUNDATION_ORDER_BONUS = 3
EXPOSE_ORDER_MULTIPLIER = 4
EMPTY_CASCADE_ORDER_BONUS = 3
SEQUENCE_GAIN_ORDER_MULTIPLIER = 2
FREE_CELL_RELEASE_BONUS = 2
FREE_CELL_DESTINATION_PENALTY = 6


@dataclass(frozen=True)
class HeuristicBreakdown:
    """Readable breakdown of the practical evaluation terms for a state."""

    foundation_remaining: int
    low_rank_buried_penalty: int
    blocked_next_foundation_penalty: int
    trapped_target_penalty: int
    empty_free_cells: int
    empty_cascades: int
    movable_sequence_bonus: int
    exposed_low_card_reward: int
    mobility_bonus: int
    heuristic_cost: float
    progress_key: Tuple[int, ...]


@dataclass(frozen=True)
class RankedChild:
    """A legal child state plus the ordering data used by the frontier."""

    move: Move
    child: GameState
    child_key: tuple
    evaluation: HeuristicBreakdown
    priority: float
    order_key: Tuple[float, ...]


class ExpertSolver(BaseSolver):
    """Practical weighted best-first FreeCell solver.

    The solver keeps every move explicit so replay always matches the search
    trace. It therefore relies on move ordering rather than hidden auto-closure
    to drive foundation progress.
    """

    @property
    def name(self) -> str:
        return "Expert Solver"

    def _running_status(self) -> str:
        return f"{self.name} running"

    def _solved_status(self) -> str:
        return f"{self.name} solved"

    def _failure_status(self) -> str:
        return f"{self.name} failed, best-progress replay ready"

    def _stop_status(self) -> str:
        return f"{self.name} stopped"

    def _nodelimit_status(self) -> str:
        return f"{self.name} failed, best-progress replay ready"

    def _stop_msg(self) -> str:
        return "Search stopped by user. Best-progress replay is ready."

    def _nodelimit_msg(self) -> str:
        return (
            f"Search stopped after {self.MAX_NODES:,} expanded nodes. "
            "Best-progress replay is ready."
        )

    def _search(self, initial: GameState) -> SolverResult:
        if initial.is_goal():
            return SolverResult(
                self.name,
                solved=True,
                solution=[],
                replay_trace=[],
                status=self._solved_status(),
                replay_message="Replaying final solution path.",
            )

        initial_eval = self._evaluate_state(initial)
        initial_key = initial.canonical_key()
        counter = 0
        frontier: List[Tuple[float, int, int, GameState]] = [
            (self._frontier_priority(0, initial_eval), -initial.cards_on_foundation, counter, initial)
        ]
        parent_links: Dict[tuple, Tuple[Optional[tuple], Optional[Move]]] = {initial_key: (None, None)}
        best_record: Dict[tuple, Tuple[int, float]] = {
            initial_key: (0, self._frontier_priority(0, initial_eval))
        }
        depth_by_key: Dict[tuple, int] = {initial_key: 0}
        closed_keys = set()

        expanded = 0
        generated = 1
        deepest_depth = 0
        best_state_key = initial_key
        best_eval = initial_eval
        best_trace_length = 0

        self._report_progress(
            moves=0,
            expanded_nodes=0,
            generated_nodes=generated,
            frontier_size=1,
            search_length=0,
            current_depth=0,
            status=self._running_status(),
            force=True,
        )

        while frontier:
            if self.stop_requested():
                return self._stopped_result(
                    parent_links,
                    best_state_key,
                    best_trace_length,
                    expanded,
                    generated,
                    len(frontier),
                    deepest_depth,
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

            priority, _neg_foundation, _counter, state = heapq.heappop(frontier)
            state_key = state.canonical_key()
            if state_key in closed_keys:
                continue
            depth = depth_by_key.get(state_key, 0)
            record = best_record.get(state_key)
            if record is None:
                continue
            record_depth, record_priority = record
            if depth != record_depth or priority > record_priority + 1e-9:
                continue

            expanded += 1
            closed_keys.add(state_key)
            deepest_depth = max(deepest_depth, depth)
            self._maybe_yield()

            if state.is_goal():
                solution = self._reconstruct_links(parent_links, state_key)
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

            parent_key = parent_links[state_key][0]
            current_eval = self._evaluate_state(state)
            ranked_children = self._rank_moves(state, depth, current_eval, parent_key)

            for candidate in ranked_children:
                if self.stop_requested():
                    return self._stopped_result(
                        parent_links,
                        best_state_key,
                        best_trace_length,
                        expanded,
                        generated,
                        len(frontier),
                        deepest_depth,
                    )

                child_key = candidate.child_key
                if child_key in closed_keys:
                    continue
                child_depth = depth + 1
                prior = best_record.get(child_key)
                if prior is not None:
                    prior_depth, prior_priority = prior
                    if child_depth >= prior_depth and candidate.priority >= prior_priority - 1e-9:
                        continue

                best_record[child_key] = (child_depth, candidate.priority)
                depth_by_key[child_key] = child_depth
                parent_links[child_key] = (state_key, candidate.move)
                counter += 1
                generated += 1
                deepest_depth = max(deepest_depth, child_depth)
                heapq.heappush(
                    frontier,
                    (
                        candidate.priority,
                        -candidate.child.cards_on_foundation,
                        counter,
                        candidate.child,
                    ),
                )

                if candidate.evaluation.progress_key > best_eval.progress_key:
                    best_state_key = child_key
                    best_eval = candidate.evaluation
                    best_trace_length = child_depth

                self._report_progress(
                    moves=best_trace_length,
                    expanded_nodes=expanded,
                    generated_nodes=generated,
                    frontier_size=len(frontier),
                    search_length=deepest_depth,
                    current_depth=child_depth,
                    status=self._running_status(),
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
            "No solution found. Best-progress replay is ready.",
            "No solution found, showing best-progress trace.",
        )

    def _rank_moves(
        self,
        state: GameState,
        depth: int,
        current_eval: HeuristicBreakdown,
        parent_key: Optional[tuple],
    ) -> List[RankedChild]:
        """Order legal moves so practical progress is explored first."""

        ranked: List[RankedChild] = []
        current_exposed = current_eval.exposed_low_card_reward
        current_sequence = current_eval.movable_sequence_bonus

        for move in get_valid_moves(state):
            child = apply_move(state, move)
            child_key = child.canonical_key()
            if parent_key is not None and child_key == parent_key:
                continue

            child_eval = self._evaluate_state(child)
            moved_cards = self._moved_cards(state, move)
            moved_card = moved_cards[0]
            foundation_order = 0
            if move.dst_type == "foundation":
                foundation_order = FOUNDATION_ORDER_BONUS
                if is_safe_to_auto_move(moved_card, state.foundations):
                    foundation_order = SAFE_FOUNDATION_ORDER_BONUS

            expose_gain = max(0, child_eval.exposed_low_card_reward - current_exposed)
            create_empty_cascade = int(
                move.src_type == "cascade"
                and len(state.cascades[move.src_idx]) == move.num_cards
                and move.dst_type != "foundation"
            )
            sequence_gain = max(0, child_eval.movable_sequence_bonus - current_sequence)
            release_free_cell = int(move.src_type == "freecell" and move.dst_type != "freecell")
            free_cell_penalty = int(move.dst_type == "freecell")
            child_priority = self._frontier_priority(depth + 1, child_eval)

            order_key = (
                foundation_order,
                expose_gain * EXPOSE_ORDER_MULTIPLIER,
                create_empty_cascade * EMPTY_CASCADE_ORDER_BONUS,
                sequence_gain * SEQUENCE_GAIN_ORDER_MULTIPLIER,
                release_free_cell * FREE_CELL_RELEASE_BONUS,
                -free_cell_penalty * FREE_CELL_DESTINATION_PENALTY,
                -child_priority,
            )
            ranked.append(
                RankedChild(
                    move=move,
                    child=child,
                    child_key=child_key,
                    evaluation=child_eval,
                    priority=child_priority,
                    order_key=order_key,
                )
            )

        ranked.sort(key=lambda item: item.order_key, reverse=True)
        return ranked

    def _evaluate_state(self, state: GameState) -> HeuristicBreakdown:
        """Combine FreeCell-specific signals into one practical search score."""

        foundation_remaining = 52 - state.cards_on_foundation
        low_rank_buried_penalty = self._low_rank_buried_penalty(state)
        blocked_next_foundation_penalty = self._blocked_next_foundation_penalty(state)
        trapped_target_penalty = self._trapped_target_penalty(state)
        movable_sequence_bonus = self._movable_sequence_bonus(state)
        exposed_low_card_reward = self._exposed_low_card_reward(state)
        mobility_bonus = self._mobility_bonus(state, movable_sequence_bonus)

        heuristic_cost = (
            FOUNDATION_REMAINING_WEIGHT * foundation_remaining
            + LOW_RANK_BURIED_WEIGHT * low_rank_buried_penalty
            + NEXT_FOUNDATION_BLOCKED_WEIGHT * blocked_next_foundation_penalty
            + TRAPPED_TARGET_WEIGHT * trapped_target_penalty
            - EMPTY_FREE_CELL_WEIGHT * state.empty_free_cells
            - EMPTY_CASCADE_WEIGHT * state.empty_cascades
            - MOVABLE_SEQUENCE_WEIGHT * movable_sequence_bonus
            - EXPOSED_LOW_CARD_WEIGHT * exposed_low_card_reward
            - MOBILITY_WEIGHT * mobility_bonus
        )

        progress_key = (
            state.cards_on_foundation,
            exposed_low_card_reward,
            state.empty_cascades,
            state.empty_free_cells,
            movable_sequence_bonus,
            mobility_bonus,
            -blocked_next_foundation_penalty,
            -low_rank_buried_penalty,
            -trapped_target_penalty,
        )
        return HeuristicBreakdown(
            foundation_remaining=foundation_remaining,
            low_rank_buried_penalty=low_rank_buried_penalty,
            blocked_next_foundation_penalty=blocked_next_foundation_penalty,
            trapped_target_penalty=trapped_target_penalty,
            empty_free_cells=state.empty_free_cells,
            empty_cascades=state.empty_cascades,
            movable_sequence_bonus=movable_sequence_bonus,
            exposed_low_card_reward=exposed_low_card_reward,
            mobility_bonus=mobility_bonus,
            heuristic_cost=heuristic_cost,
            progress_key=progress_key,
        )

    def _frontier_priority(self, g_cost: int, evaluation: HeuristicBreakdown) -> float:
        """Weighted best-first priority: low is good, but still keeps g in play."""

        return g_cost + evaluation.heuristic_cost

    def _low_rank_buried_penalty(self, state: GameState) -> int:
        """Penalise buried A-4 cards because early ranks unlock the whole deal."""

        penalty = 0
        for column in state.cascades:
            for depth_from_top, card in enumerate(reversed(column)):
                if card.rank <= 4 and state.foundations[card.suit] < card.rank:
                    penalty += self._low_rank_weight(card.rank) * depth_from_top
        return penalty

    def _blocked_next_foundation_penalty(self, state: GameState) -> int:
        """Penalise the exact cards each foundation currently needs when buried."""

        penalty = 0
        for suit in range(4):
            needed_rank = state.foundations[suit] + 1
            if needed_rank > 13:
                continue

            if any(
                free_card is not None and free_card.suit == suit and free_card.rank == needed_rank
                for free_card in state.free_cells
            ):
                continue

            for column in state.cascades:
                for depth_from_top, card in enumerate(reversed(column)):
                    if card.suit == suit and card.rank == needed_rank:
                        penalty += 1 + depth_from_top * 2
                        break
                else:
                    continue
                break
        return penalty

    def _trapped_target_penalty(self, state: GameState) -> int:
        """Penalise cards sitting above important low-rank or next-foundation targets."""

        penalty = 0
        for column in state.cascades:
            for depth_from_top, card in enumerate(reversed(column)):
                if state.foundations[card.suit] >= card.rank:
                    continue

                important = card.rank <= 4 or card.rank == state.foundations[card.suit] + 1
                if not important or depth_from_top == 0:
                    continue

                penalty += depth_from_top * max(2, self._low_rank_weight(min(card.rank, 4)))
        return penalty

    def _movable_sequence_bonus(self, state: GameState) -> int:
        """Reward long alternating runs that can be moved or extended productively."""

        bonus = 0
        for column in state.cascades:
            run = self._top_run_length(column)
            if run >= 2:
                bonus += run * run
        return bonus

    def _exposed_low_card_reward(self, state: GameState) -> int:
        """Reward visible A-4 cards and visible next-foundation cards."""

        reward = 0
        visible_cards = [column[-1] for column in state.cascades if column]
        visible_cards.extend(card for card in state.free_cells if card is not None)
        for card in visible_cards:
            if state.foundations[card.suit] >= card.rank:
                continue
            if card.rank <= 4:
                reward += self._low_rank_weight(card.rank)
            if card.rank == state.foundations[card.suit] + 1:
                reward += 4
        return reward

    def _mobility_bonus(self, state: GameState, movable_sequence_bonus: int) -> int:
        """Reward open space and useful landing spots without running a full subsearch."""

        mobile_free_cells = sum(
            1
            for card in state.free_cells
            if card is not None
            and any((not column) or card.can_stack_on(column[-1]) for column in state.cascades)
        )
        return (
            state.empty_free_cells * 2
            + state.empty_cascades * 3
            + mobile_free_cells
            + min(12, movable_sequence_bonus // 6)
        )

    def _top_run_length(self, column: Tuple[Card, ...]) -> int:
        if not column:
            return 0

        run = 1
        idx = len(column) - 1
        while idx > 0 and column[idx].can_stack_on(column[idx - 1]):
            run += 1
            idx -= 1
        return run

    def _moved_cards(self, state: GameState, move: Move) -> Tuple[Card, ...]:
        if move.src_type == "cascade":
            return tuple(state.cascades[move.src_idx][-move.num_cards:])
        return (state.free_cells[move.src_idx],)

    def _low_rank_weight(self, rank: int) -> int:
        return {1: 12, 2: 9, 3: 6, 4: 4}.get(rank, 1)

    def _stopped_result(
        self,
        parent_links: Dict[tuple, Tuple[Optional[tuple], Optional[Move]]],
        best_state_key: tuple,
        best_trace_length: int,
        expanded: int,
        generated: int,
        frontier_size: int,
        deepest_depth: int,
    ) -> SolverResult:
        return self._failed_result(
            parent_links,
            best_state_key,
            best_trace_length,
            expanded,
            generated,
            frontier_size,
            deepest_depth,
            self._stop_status(),
            self._stop_msg(),
            "Stopped by user, showing best-progress trace.",
        )

    def _failed_result(
        self,
        parent_links: Dict[tuple, Tuple[Optional[tuple], Optional[Move]]],
        best_state_key: tuple,
        best_trace_length: int,
        expanded: int,
        generated: int,
        frontier_size: int,
        deepest_depth: int,
        status: str,
        message: str,
        replay_message: str,
    ) -> SolverResult:
        best_trace = self._reconstruct_links(parent_links, best_state_key)
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
            search_length=max(deepest_depth, best_trace_length),
            message=message,
        )

    def _reconstruct_links(
        self,
        parent_links: Dict[tuple, Tuple[Optional[tuple], Optional[Move]]],
        state_key: tuple,
    ) -> List[Move]:
        """Recover a move path using immutable parent links."""

        path: List[Move] = []
        current_key = state_key
        while parent_links[current_key][1] is not None:
            parent_key, move = parent_links[current_key]
            path.append(move)
            current_key = parent_key
        path.reverse()
        return path
