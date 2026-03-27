from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from game.card import Card
from game.moves import Move, apply_move, get_valid_moves, is_safe_to_auto_move
from game.state import GameState
from .base import BaseSolver, SolverResult


FOUNDATION_REMAINING_WEIGHT = 5.2
LOW_RANK_BURIED_WEIGHT = 2.8
NEXT_FOUNDATION_BLOCKED_WEIGHT = 4.0
TRAPPED_TARGET_WEIGHT = 2.1
EMPTY_FREE_CELL_WEIGHT = 2.4
EMPTY_CASCADE_WEIGHT = 7.0
MOVABLE_SEQUENCE_WEIGHT = 1.6
EXPOSED_LOW_CARD_WEIGHT = 2.8
MOBILITY_WEIGHT = 1.35
READY_FOUNDATION_WEIGHT = 5.0

SAFE_FOUNDATION_ORDER_BONUS = 10
FOUNDATION_ORDER_BONUS = 4
EXPOSE_ORDER_MULTIPLIER = 4
EMPTY_CASCADE_ORDER_BONUS = 4
SEQUENCE_GAIN_ORDER_MULTIPLIER = 2
FREE_CELL_RELEASE_BONUS = 3
FREE_CELL_DESTINATION_PENALTY = 7
READY_FOUNDATION_ORDER_BONUS = 5

FRONTIER_SOFT_LIMIT = 180_000
FRONTIER_TRIM_TO = 110_000
REOPEN_EPSILON = 1e-9


@dataclass(frozen=True)
class HeuristicBreakdown:
    foundation_remaining: int
    low_rank_buried_penalty: int
    blocked_next_foundation_penalty: int
    trapped_target_penalty: int
    ready_foundation_reward: int
    empty_free_cells: int
    empty_cascades: int
    movable_sequence_bonus: int
    exposed_low_card_reward: int
    mobility_bonus: int
    heuristic_cost: float
    progress_key: Tuple[int, ...]


@dataclass(frozen=True)
class RankedChild:
    move_seq: Tuple[Move, ...]
    child: GameState
    child_key: tuple
    evaluation: HeuristicBreakdown
    priority: float
    order_key: Tuple[float, ...]


class ExpertSolver(BaseSolver):
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
        self._eval_cache: Dict[tuple, HeuristicBreakdown] = {}
        self._legal_moves_cache: Dict[tuple, Tuple[Move, ...]] = {}
        self._closure_cache: Dict[tuple, Tuple[GameState, Tuple[Move, ...]]] = {}

        initial, initial_auto_moves = self._collapse_safe_foundations(initial)
        if initial.is_goal():
            return SolverResult(
                self.name,
                solved=True,
                solution=list(initial_auto_moves),
                replay_trace=list(initial_auto_moves),
                best_trace_length=len(initial_auto_moves),
                replay_label="Replay Solution",
                replay_message="Replaying final solution path.",
                status=self._solved_status(),
            )

        initial_key = initial.canonical_key()
        initial_eval = self._evaluate_state(initial)
        initial_g = len(initial_auto_moves)

        counter = 0
        frontier: List[Tuple[float, int, int, int, GameState]] = [
            (
                self._frontier_priority(initial_g, initial_eval),
                -initial.cards_on_foundation,
                initial_g,
                counter,
                initial,
            )
        ]

        parent_links: Dict[tuple, Tuple[Optional[tuple], Tuple[Move, ...]]] = {
            initial_key: (None, tuple(initial_auto_moves))
        }
        best_record: Dict[tuple, Tuple[int, float]] = {
            initial_key: (initial_g, self._frontier_priority(initial_g, initial_eval))
        }
        expanded_best_priority: Dict[tuple, float] = {}
        expanded = 0
        generated = 1
        best_state_key = initial_key
        best_eval = initial_eval
        best_trace_length = initial_g
        deepest_search_length = initial_g

        self._report_progress(
            moves=initial_g,
            expanded_nodes=0,
            generated_nodes=generated,
            frontier_size=1,
            search_length=initial_g,
            current_depth=initial_g,
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
                    deepest_search_length,
                )

            if expanded >= self.MAX_NODES:
                return self._failed_result(
                    parent_links,
                    best_state_key,
                    best_trace_length,
                    expanded,
                    generated,
                    len(frontier),
                    deepest_search_length,
                    self._nodelimit_status(),
                    self._nodelimit_msg(),
                    "Node limit reached, showing best-progress trace.",
                )

            priority, _neg_foundation, g_cost, _counter, state = heapq.heappop(frontier)
            state_key = state.canonical_key()

            record = best_record.get(state_key)
            if record is None:
                continue
            record_g, record_priority = record
            if g_cost != record_g or priority > record_priority + REOPEN_EPSILON:
                continue

            expanded_priority = expanded_best_priority.get(state_key)
            if expanded_priority is not None and priority >= expanded_priority - REOPEN_EPSILON:
                continue
            expanded_best_priority[state_key] = priority

            expanded += 1
            deepest_search_length = max(deepest_search_length, g_cost)
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
                    current_depth=g_cost,
                )

            parent_key = parent_links[state_key][0]
            current_eval = self._evaluate_state(state)
            ranked_children = self._rank_moves(state, g_cost, current_eval, parent_key)

            for candidate in ranked_children:
                if self.stop_requested():
                    return self._stopped_result(
                        parent_links,
                        best_state_key,
                        best_trace_length,
                        expanded,
                        generated,
                        len(frontier),
                        deepest_search_length,
                    )

                child_key = candidate.child_key
                child_g = g_cost + len(candidate.move_seq)
                prior = best_record.get(child_key)
                if prior is not None:
                    prior_g, prior_priority = prior
                    if child_g > prior_g and candidate.priority >= prior_priority - REOPEN_EPSILON:
                        continue
                    if child_g == prior_g and candidate.priority >= prior_priority - REOPEN_EPSILON:
                        continue

                best_record[child_key] = (child_g, candidate.priority)
                parent_links[child_key] = (state_key, candidate.move_seq)
                counter += 1
                generated += 1
                deepest_search_length = max(deepest_search_length, child_g)

                heapq.heappush(
                    frontier,
                    (
                        candidate.priority,
                        -candidate.child.cards_on_foundation,
                        child_g,
                        counter,
                        candidate.child,
                    ),
                )

                if candidate.evaluation.progress_key > best_eval.progress_key:
                    best_state_key = child_key
                    best_eval = candidate.evaluation
                    best_trace_length = max(best_trace_length, child_g)

                self._report_progress(
                    moves=best_trace_length,
                    expanded_nodes=expanded,
                    generated_nodes=generated,
                    frontier_size=len(frontier),
                    search_length=deepest_search_length,
                    current_depth=child_g,
                    status=self._running_status(),
                )

            if len(frontier) > FRONTIER_SOFT_LIMIT:
                frontier = self._trim_frontier(frontier, best_record)
                heapq.heapify(frontier)

        return self._failed_result(
            parent_links,
            best_state_key,
            best_trace_length,
            expanded,
            generated,
            0,
            deepest_search_length,
            self._failure_status(),
            "No solution found. Best-progress replay is ready.",
            "No solution found, showing best-progress trace.",
        )

    def _trim_frontier(
        self,
        frontier: List[Tuple[float, int, int, int, GameState]],
        best_record: Dict[tuple, Tuple[int, float]],
    ) -> List[Tuple[float, int, int, int, GameState]]:
        trimmed = heapq.nsmallest(FRONTIER_TRIM_TO, frontier)
        refreshed: List[Tuple[float, int, int, int, GameState]] = []
        for item in trimmed:
            priority, neg_foundation, g_cost, counter, state = item
            state_key = state.canonical_key()
            record = best_record.get(state_key)
            if record is None:
                continue
            record_g, record_priority = record
            if g_cost == record_g and priority <= record_priority + REOPEN_EPSILON:
                refreshed.append((priority, neg_foundation, g_cost, counter, state))
        return refreshed

    def _rank_moves(
        self,
        state: GameState,
        g_cost: int,
        current_eval: HeuristicBreakdown,
        parent_key: Optional[tuple],
    ) -> List[RankedChild]:
        ranked: List[RankedChild] = []
        current_exposed = current_eval.exposed_low_card_reward
        current_sequence = current_eval.movable_sequence_bonus
        current_ready = current_eval.ready_foundation_reward

        all_moves = self._get_legal_moves(state)
        move_budget = self._move_budget(state, len(all_moves))

        for move in all_moves:
            child = apply_move(state, move)
            child, auto_moves = self._collapse_safe_foundations(child)
            move_seq = (move,) + auto_moves
            child_key = child.canonical_key()

            if parent_key is not None and child_key == parent_key:
                continue

            child_eval = self._evaluate_state(child)
            moved_card = self._moved_card_or_none(state, move)
            if moved_card is None:
                continue

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
            ready_gain = max(0, child_eval.ready_foundation_reward - current_ready)
            release_free_cell = int(move.src_type == "freecell" and move.dst_type != "freecell")
            free_cell_penalty = int(move.dst_type == "freecell")
            child_priority = self._frontier_priority(g_cost + len(move_seq), child_eval)

            order_key = (
                foundation_order,
                ready_gain * READY_FOUNDATION_ORDER_BONUS,
                expose_gain * EXPOSE_ORDER_MULTIPLIER,
                create_empty_cascade * EMPTY_CASCADE_ORDER_BONUS,
                sequence_gain * SEQUENCE_GAIN_ORDER_MULTIPLIER,
                release_free_cell * FREE_CELL_RELEASE_BONUS,
                -free_cell_penalty * FREE_CELL_DESTINATION_PENALTY,
                -child_priority,
            )

            ranked.append(
                RankedChild(
                    move_seq=move_seq,
                    child=child,
                    child_key=child_key,
                    evaluation=child_eval,
                    priority=child_priority,
                    order_key=order_key,
                )
            )

        ranked.sort(key=lambda item: item.order_key, reverse=True)

        if len(ranked) <= move_budget:
            return ranked

        kept: List[RankedChild] = []
        forced_keys = set()

        for item in ranked:
            lead_move = item.move_seq[0]
            moved_card = self._moved_card_or_none(state, lead_move)
            force_keep = (
                lead_move.dst_type == "foundation"
                or item.evaluation.progress_key > current_eval.progress_key
                or (
                    lead_move.src_type == "cascade"
                    and len(state.cascades[lead_move.src_idx]) == lead_move.num_cards
                    and lead_move.dst_type != "foundation"
                )
                or (
                    moved_card is not None
                    and is_safe_to_auto_move(moved_card, state.foundations)
                )
            )
            if force_keep:
                kept.append(item)
                forced_keys.add(item.child_key)

        if len(kept) >= move_budget:
            kept.sort(key=lambda item: item.order_key, reverse=True)
            return kept

        for item in ranked:
            if item.child_key in forced_keys:
                continue
            kept.append(item)
            if len(kept) >= move_budget:
                break

        kept.sort(key=lambda item: item.order_key, reverse=True)
        return kept

    def _move_budget(self, state: GameState, move_count: int) -> int:
        if state.cards_on_foundation >= 40:
            return move_count
        budget = 12 + state.empty_free_cells * 3 + state.empty_cascades * 4
        return min(move_count, max(10, budget))

    def _get_legal_moves(self, state: GameState) -> Tuple[Move, ...]:
        key = self._slot_sensitive_key(state)
        cached = self._legal_moves_cache.get(key)
        if cached is None:
            cached = tuple(get_valid_moves(state))
            self._legal_moves_cache[key] = cached
        return cached

    def _collapse_safe_foundations(self, state: GameState) -> Tuple[GameState, Tuple[Move, ...]]:
        key = self._slot_sensitive_key(state)
        cached = self._closure_cache.get(key)
        if cached is not None:
            return cached

        current = state
        path: List[Move] = []

        while True:
            safe_move = self._pick_safe_foundation_move(current)
            if safe_move is None:
                break
            path.append(safe_move)
            current = apply_move(current, safe_move)

        result = (current, tuple(path))
        self._closure_cache[key] = result
        return result

    def _pick_safe_foundation_move(self, state: GameState) -> Optional[Move]:
        best_move: Optional[Move] = None
        best_rank = 99

        for move in self._get_legal_moves(state):
            if move.dst_type != "foundation":
                continue
            moved_card = self._moved_card_or_none(state, move)
            if moved_card is None:
                continue
            if not is_safe_to_auto_move(moved_card, state.foundations):
                continue
            rank_key = moved_card.rank
            if best_move is None or rank_key < best_rank:
                best_move = move
                best_rank = rank_key

        return best_move

    def _evaluate_state(self, state: GameState) -> HeuristicBreakdown:
        key = state.canonical_key()
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached

        foundation_remaining = 52 - state.cards_on_foundation
        low_rank_buried_penalty = self._low_rank_buried_penalty(state)
        blocked_next_foundation_penalty = self._blocked_next_foundation_penalty(state)
        trapped_target_penalty = self._trapped_target_penalty(state)
        ready_foundation_reward = self._ready_foundation_reward(state)
        movable_sequence_bonus = self._movable_sequence_bonus(state)
        exposed_low_card_reward = self._exposed_low_card_reward(state)
        mobility_bonus = self._mobility_bonus(state, movable_sequence_bonus)

        heuristic_cost = (
            FOUNDATION_REMAINING_WEIGHT * foundation_remaining
            + LOW_RANK_BURIED_WEIGHT * low_rank_buried_penalty
            + NEXT_FOUNDATION_BLOCKED_WEIGHT * blocked_next_foundation_penalty
            + TRAPPED_TARGET_WEIGHT * trapped_target_penalty
            - READY_FOUNDATION_WEIGHT * ready_foundation_reward
            - EMPTY_FREE_CELL_WEIGHT * state.empty_free_cells
            - EMPTY_CASCADE_WEIGHT * state.empty_cascades
            - MOVABLE_SEQUENCE_WEIGHT * movable_sequence_bonus
            - EXPOSED_LOW_CARD_WEIGHT * exposed_low_card_reward
            - MOBILITY_WEIGHT * mobility_bonus
        )

        progress_key = (
            state.cards_on_foundation,
            ready_foundation_reward,
            exposed_low_card_reward,
            state.empty_cascades,
            state.empty_free_cells,
            movable_sequence_bonus,
            mobility_bonus,
            -blocked_next_foundation_penalty,
            -low_rank_buried_penalty,
            -trapped_target_penalty,
        )

        cached = HeuristicBreakdown(
            foundation_remaining=foundation_remaining,
            low_rank_buried_penalty=low_rank_buried_penalty,
            blocked_next_foundation_penalty=blocked_next_foundation_penalty,
            trapped_target_penalty=trapped_target_penalty,
            ready_foundation_reward=ready_foundation_reward,
            empty_free_cells=state.empty_free_cells,
            empty_cascades=state.empty_cascades,
            movable_sequence_bonus=movable_sequence_bonus,
            exposed_low_card_reward=exposed_low_card_reward,
            mobility_bonus=mobility_bonus,
            heuristic_cost=heuristic_cost,
            progress_key=progress_key,
        )
        self._eval_cache[key] = cached
        return cached

    def _frontier_priority(self, g_cost: int, evaluation: HeuristicBreakdown) -> float:
        return g_cost + evaluation.heuristic_cost

    def _ready_foundation_reward(self, state: GameState) -> int:
        reward = 0
        for move in self._get_legal_moves(state):
            if move.dst_type != "foundation":
                continue
            moved_card = self._moved_card_or_none(state, move)
            if moved_card is None:
                continue
            reward += 2 if is_safe_to_auto_move(moved_card, state.foundations) else 1
        return reward

    def _low_rank_buried_penalty(self, state: GameState) -> int:
        penalty = 0
        for column in state.cascades:
            for depth_from_top, card in enumerate(reversed(column)):
                if card.rank <= 4 and state.foundations[card.suit] < card.rank:
                    penalty += self._low_rank_weight(card.rank) * depth_from_top
        return penalty

    def _blocked_next_foundation_penalty(self, state: GameState) -> int:
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
        bonus = 0
        transfer_capacity = self._sequence_transfer_capacity(state)
        for column in state.cascades:
            run = self._top_run_length(column)
            if run >= 2:
                bonus += run * min(run, transfer_capacity)
        return bonus

    def _exposed_low_card_reward(self, state: GameState) -> int:
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
        mobile_free_cells = sum(
            1
            for card in state.free_cells
            if card is not None
            and any((not column) or card.can_stack_on(column[-1]) for column in state.cascades)
        )
        return (
            state.empty_free_cells * 2
            + state.empty_cascades * 4
            + mobile_free_cells
            + min(16, movable_sequence_bonus // 5)
            + min(12, self._sequence_transfer_capacity(state))
        )

    def _sequence_transfer_capacity(self, state: GameState) -> int:
        return (state.empty_free_cells + 1) * (2 ** state.empty_cascades)

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
        card = state.free_cells[move.src_idx]
        return (card,) if card is not None else tuple()

    def _moved_card_or_none(self, state: GameState, move: Move) -> Optional[Card]:
        moved_cards = self._moved_cards(state, move)
        return moved_cards[0] if moved_cards else None

    def _low_rank_weight(self, rank: int) -> int:
        return {1: 12, 2: 9, 3: 6, 4: 4}.get(rank, 1)

    def _slot_sensitive_key(self, state: GameState) -> tuple:
        free_cells = tuple(fc.to_int() if fc is not None else -1 for fc in state.free_cells)
        return (state.cascades, free_cells, state.foundations)

    def _stopped_result(
        self,
        parent_links: Dict[tuple, Tuple[Optional[tuple], Tuple[Move, ...]]],
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
        parent_links: Dict[tuple, Tuple[Optional[tuple], Tuple[Move, ...]]],
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
        parent_links: Dict[tuple, Tuple[Optional[tuple], Tuple[Move, ...]]],
        state_key: tuple,
    ) -> List[Move]:
        path: List[Move] = []
        current_key = state_key

        while True:
            parent_key, moves = parent_links[current_key]
            if moves:
                path.extend(reversed(moves))
            if parent_key is None:
                break
            current_key = parent_key

        path.reverse()
        return path
