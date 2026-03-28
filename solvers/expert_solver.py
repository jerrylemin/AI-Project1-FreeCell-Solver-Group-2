"""High-power practical FreeCell solver with layered search."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from game.card import Card
from game.moves import Move, apply_move, get_valid_moves, is_safe_to_auto_move
from game.state import GameState
from .base import BaseSolver, SolverResult


FOUNDATION_REMAINING_WEIGHT = 5.6
LOW_RANK_BURIED_WEIGHT = 3.2
NEXT_FOUNDATION_BLOCKED_WEIGHT = 4.8
TRAPPED_TARGET_WEIGHT = 2.3
EMPTY_FREE_CELL_WEIGHT = 2.3
EMPTY_CASCADE_WEIGHT = 6.8
MOVABLE_SEQUENCE_WEIGHT = 1.7
EXPOSED_LOW_CARD_WEIGHT = 3.0
MOBILITY_WEIGHT = 1.5
SEQUENCE_CAPACITY_WEIGHT = 1.35
PRODUCTIVE_MOVE_WEIGHT = 1.45
READY_FOUNDATION_WEIGHT = 2.2
DEAD_END_WEIGHT = 4.8
LOCKED_TOP_WEIGHT = 2.0

SAFE_FOUNDATION_ORDER_BONUS = 80
FOUNDATION_ORDER_BONUS = 22
EXPOSE_ORDER_MULTIPLIER = 6
EMPTY_CASCADE_ORDER_BONUS = 12
SEQUENCE_GAIN_ORDER_MULTIPLIER = 3
FREE_CELL_RELEASE_BONUS = 8
PRODUCTIVE_GAIN_ORDER_MULTIPLIER = 4
DEAD_REDUCTION_ORDER_MULTIPLIER = 5
FREE_CELL_DESTINATION_PENALTY = 8
LOCK_RISK_PENALTY = 5

REOPEN_EPSILON = 1e-9
ENDGAME_REMAINING_CARD_THRESHOLD = 16
ENDGAME_RESOURCE_THRESHOLD = 3
CONCENTRATED_ENDGAME_REMAINING_CARD_THRESHOLD = 34
CONCENTRATED_ENDGAME_SEQUENCE_CAPACITY_THRESHOLD = 8
CONCENTRATED_ENDGAME_FOUNDATION_THRESHOLD = 18


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
    sequence_capacity_bonus: int
    productive_move_bonus: int
    dead_end_penalty: int
    locked_top_penalty: int
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


@dataclass(frozen=True)
class SearchRecord:
    g_cost: int
    priority: float


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    g_weight: float
    order_bias: float
    progress_bias: float
    stagnation_limit: int
    foundation_bias: int
    freecell_penalty: int
    lock_penalty: int


@dataclass(frozen=True)
class PhaseOutcome:
    solved: bool
    status: str
    message: str
    replay_message: str
    solution: List[Move]
    replay_trace: List[Move]
    best_eval: HeuristicBreakdown
    best_trace_length: int
    expanded_nodes: int
    generated_nodes: int
    frontier_size: int
    search_length: int
    current_depth: int


PHASES = (
    PhaseConfig("fast", 0.55, 1.0, 0.42, 18_000, 1, FREE_CELL_DESTINATION_PENALTY, LOCK_RISK_PENALTY),
    PhaseConfig("stable", 0.9, 0.7, 0.28, 36_000, 1, max(3, FREE_CELL_DESTINATION_PENALTY - 3), max(2, LOCK_RISK_PENALTY - 1)),
    PhaseConfig("rescue", 1.25, 0.35, 0.16, 0, 0, max(1, FREE_CELL_DESTINATION_PENALTY - 5), max(1, LOCK_RISK_PENALTY - 3)),
)


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
        self._ensure_caches(reset=True)
        self._reported_best_trace_length = 0

        initial, initial_auto = self._normalize_state(initial)
        initial_eval = self._evaluate_state(initial)

        if initial.is_goal():
            return SolverResult(
                self.name,
                solved=True,
                solution=list(initial_auto),
                replay_trace=list(initial_auto),
                best_trace_length=len(initial_auto),
                replay_label="Replay Solution",
                replay_message="Replaying final solution path.",
                status=self._solved_status(),
            )

        phase_budgets = self._phase_budgets()
        total_expanded = 0
        total_generated = 0
        total_search_length = len(initial_auto)
        total_frontier_size = 1
        global_best_eval = initial_eval
        global_best_trace: List[Move] = list(initial_auto)
        global_best_trace_length = len(initial_auto)
        best_reported_length = len(initial_auto)
        self._reported_best_trace_length = best_reported_length
        last_outcome: Optional[PhaseOutcome] = None

        self._report_progress(
            moves=best_reported_length,
            expanded_nodes=0,
            generated_nodes=1,
            frontier_size=1,
            search_length=len(initial_auto),
            current_depth=len(initial_auto),
            status=self._running_status(),
            force=True,
        )

        for phase, budget in zip(PHASES, phase_budgets):
            if self.stop_requested():
                break
            if budget <= 0:
                continue

            outcome = self._run_phase(initial, initial_auto, initial_eval, phase, budget)
            last_outcome = outcome
            total_expanded += outcome.expanded_nodes
            total_generated += outcome.generated_nodes
            total_search_length = max(total_search_length, outcome.search_length)
            total_frontier_size = outcome.frontier_size
            best_reported_length = max(best_reported_length, outcome.best_trace_length)

            if outcome.best_eval.progress_key > global_best_eval.progress_key:
                global_best_eval = outcome.best_eval
                global_best_trace = list(outcome.replay_trace)
                global_best_trace_length = outcome.best_trace_length
            elif (
                outcome.best_eval.progress_key == global_best_eval.progress_key
                and len(outcome.replay_trace) > len(global_best_trace)
            ):
                global_best_trace = list(outcome.replay_trace)
                global_best_trace_length = max(global_best_trace_length, outcome.best_trace_length)

            if outcome.solved:
                return SolverResult(
                    self.name,
                    solved=True,
                    solution=outcome.solution,
                    replay_trace=list(outcome.solution),
                    best_trace_length=len(outcome.solution),
                    replay_label="Replay Solution",
                    replay_message="Replaying final solution path.",
                    status=self._solved_status(),
                    expanded_nodes=total_expanded,
                    generated_nodes=total_generated,
                    frontier_size=outcome.frontier_size,
                    search_length=outcome.search_length,
                    current_depth=outcome.current_depth,
                )

            self._report_progress(
                moves=max(self._reported_best_trace_length, best_reported_length),
                expanded_nodes=total_expanded,
                generated_nodes=total_generated,
                frontier_size=total_frontier_size,
                search_length=total_search_length,
                current_depth=outcome.current_depth,
                status=self._running_status(),
                message=outcome.message,
                force=True,
            )

            if self.stop_requested():
                break

        if self.stop_requested():
            return SolverResult(
                self.name,
                solved=False,
                replay_trace=global_best_trace,
                best_trace_length=max(best_reported_length, len(global_best_trace)),
                replay_label="Replay Failed Attempt",
                replay_message="Stopped by user, showing best-progress trace.",
                status=self._stop_status(),
                expanded_nodes=total_expanded,
                generated_nodes=total_generated,
                frontier_size=total_frontier_size,
                search_length=total_search_length,
                message=self._stop_msg(),
            )

        node_limited = total_expanded >= self.MAX_NODES
        return SolverResult(
            self.name,
            solved=False,
            replay_trace=global_best_trace,
            best_trace_length=max(best_reported_length, len(global_best_trace)),
            replay_label="Replay Failed Attempt",
            replay_message=(
                "Node limit reached, showing best-progress trace."
                if node_limited
                else "No solution found, showing best-progress trace."
            ),
            status=self._nodelimit_status() if node_limited else self._failure_status(),
            expanded_nodes=total_expanded,
            generated_nodes=total_generated,
            frontier_size=total_frontier_size,
            search_length=total_search_length,
            message=self._nodelimit_msg() if node_limited else (
                last_outcome.message if last_outcome is not None else "No solution found."
            ),
        )

    def _ensure_caches(self, *, reset: bool = False) -> None:
        if reset or not hasattr(self, "_eval_cache"):
            self._eval_cache = {}
        if reset or not hasattr(self, "_legal_moves_cache"):
            self._legal_moves_cache = {}
        if reset or not hasattr(self, "_normalize_cache"):
            self._normalize_cache = {}
        if reset or not hasattr(self, "_dead_penalty_cache"):
            self._dead_penalty_cache = {}
        if reset or not hasattr(self, "_productive_cache"):
            self._productive_cache = {}
        if reset or not hasattr(self, "_locked_top_cache"):
            self._locked_top_cache = {}
        if reset or not hasattr(self, "_sequence_capacity_cache"):
            self._sequence_capacity_cache = {}
        if reset or not hasattr(self, "_endgame_cache"):
            self._endgame_cache = {}
        if reset or not hasattr(self, "_endgame_fail_budget"):
            self._endgame_fail_budget = {}
        if reset or not hasattr(self, "_reported_best_trace_length"):
            self._reported_best_trace_length = 0

    def _phase_budgets(self) -> List[int]:
        fast = max(2_000, int(self.MAX_NODES * 0.28))
        stable = max(4_000, int(self.MAX_NODES * 0.34))
        rescue = max(1, self.MAX_NODES - fast - stable)
        return [fast, stable, rescue]

    def _run_phase(
        self,
        initial: GameState,
        initial_auto: Tuple[Move, ...],
        initial_eval: HeuristicBreakdown,
        phase: PhaseConfig,
        node_budget: int,
    ) -> PhaseOutcome:
        initial_key = initial.canonical_key()
        initial_priority = self._frontier_priority(0, initial_eval, phase)
        counter = 0
        frontier: List[Tuple[float, int, int, int, GameState]] = [
            (
                initial_priority,
                -initial.cards_on_foundation,
                -initial_eval.sequence_capacity_bonus,
                counter,
                initial,
            )
        ]
        parent_links: Dict[tuple, Tuple[Optional[tuple], Tuple[Move, ...]]] = {
            initial_key: (None, initial_auto)
        }
        best_record: Dict[tuple, SearchRecord] = {
            initial_key: SearchRecord(0, initial_priority)
        }
        expanded_best: Dict[tuple, SearchRecord] = {}
        depth_by_key: Dict[tuple, int] = {initial_key: len(initial_auto)}

        local_best_state_key = initial_key
        local_best_eval = initial_eval
        local_best_trace_length = len(initial_auto)
        expanded = 0
        generated = 1
        deepest_depth = len(initial_auto)
        stagnation = 0

        while frontier:
            if self.stop_requested() or expanded >= node_budget:
                break
            if phase.stagnation_limit and stagnation >= phase.stagnation_limit:
                break

            priority, _neg_foundation, _neg_capacity, _counter, state = heapq.heappop(frontier)
            state_key = state.canonical_key()
            record = best_record.get(state_key)
            if record is None or priority > record.priority + REOPEN_EPSILON:
                continue

            current_depth = depth_by_key.get(state_key, 0)
            expanded_record = expanded_best.get(state_key)
            if expanded_record is not None:
                if current_depth > expanded_record.g_cost:
                    continue
                if current_depth == expanded_record.g_cost and priority >= expanded_record.priority - REOPEN_EPSILON:
                    continue
            expanded_best[state_key] = SearchRecord(current_depth, priority)

            expanded += 1
            deepest_depth = max(deepest_depth, current_depth)
            stagnation += 1
            self._maybe_yield()

            if state.is_goal():
                solution = self._reconstruct_links(parent_links, state_key)
                return PhaseOutcome(
                    solved=True,
                    status=self._solved_status(),
                    message="Solved.",
                    replay_message="Replaying final solution path.",
                    solution=solution,
                    replay_trace=list(solution),
                    best_eval=local_best_eval,
                    best_trace_length=len(solution),
                    expanded_nodes=expanded,
                    generated_nodes=generated,
                    frontier_size=len(frontier),
                    search_length=max(deepest_depth, len(solution)),
                    current_depth=current_depth,
                )

            if self._should_enter_endgame(state):
                remaining_budget = min(12_000, max(2_500, node_budget - expanded))
                endgame = self._solve_endgame(state, remaining_budget)
                expanded += endgame[1]
                generated += endgame[2]
                deepest_depth = max(deepest_depth, current_depth + endgame[3])
                if endgame[0] is not None:
                    prefix = self._reconstruct_links(parent_links, state_key)
                    solution = prefix + list(endgame[0])
                    return PhaseOutcome(
                        solved=True,
                        status=self._solved_status(),
                        message="Solved by endgame finisher.",
                        replay_message="Replaying final solution path.",
                        solution=solution,
                        replay_trace=list(solution),
                        best_eval=local_best_eval,
                        best_trace_length=len(solution),
                        expanded_nodes=expanded,
                        generated_nodes=generated,
                        frontier_size=len(frontier),
                        search_length=max(deepest_depth, len(solution)),
                        current_depth=current_depth + endgame[3],
                    )

            parent_key = parent_links[state_key][0]
            current_eval = self._evaluate_state(state)
            ranked_children = self._rank_moves(state, current_depth, current_eval, parent_key, phase)

            for candidate in ranked_children:
                if self.stop_requested():
                    break

                child_key = candidate.child_key
                child_depth = current_depth + len(candidate.move_seq)
                prior = best_record.get(child_key)
                if prior is not None and not self._should_reopen(candidate, child_depth, prior):
                    continue

                best_record[child_key] = SearchRecord(child_depth, candidate.priority)
                parent_links[child_key] = (state_key, candidate.move_seq)
                depth_by_key[child_key] = child_depth
                counter += 1
                generated += 1
                deepest_depth = max(deepest_depth, child_depth)

                heapq.heappush(
                    frontier,
                    (
                        candidate.priority,
                        -candidate.child.cards_on_foundation,
                        -candidate.evaluation.sequence_capacity_bonus,
                        counter,
                        candidate.child,
                    ),
                )

                if self._is_better_progress(candidate.evaluation, local_best_eval):
                    local_best_state_key = child_key
                    local_best_eval = candidate.evaluation
                    local_best_trace_length = max(local_best_trace_length, child_depth)
                    self._reported_best_trace_length = max(
                        self._reported_best_trace_length,
                        local_best_trace_length,
                    )
                    stagnation = 0

                self._report_progress(
                    moves=max(self._reported_best_trace_length, local_best_trace_length),
                    expanded_nodes=expanded,
                    generated_nodes=generated,
                    frontier_size=len(frontier),
                    search_length=deepest_depth,
                    current_depth=child_depth,
                    status=self._running_status(),
                )

        best_trace = self._reconstruct_links(parent_links, local_best_state_key)
        if self.stop_requested():
            status = self._stop_status()
            message = self._stop_msg()
            replay_message = "Stopped by user, showing best-progress trace."
        elif expanded >= node_budget:
            status = self._running_status()
            message = f"Phase {phase.name} budget exhausted, escalating search."
            replay_message = "Best-progress replay is ready."
        else:
            status = self._running_status()
            message = f"Phase {phase.name} stagnated, escalating search."
            replay_message = "Best-progress replay is ready."

        return PhaseOutcome(
            solved=False,
            status=status,
            message=message,
            replay_message=replay_message,
            solution=[],
            replay_trace=best_trace,
            best_eval=local_best_eval,
            best_trace_length=max(local_best_trace_length, len(best_trace)),
            expanded_nodes=expanded,
            generated_nodes=generated,
            frontier_size=len(frontier),
            search_length=max(deepest_depth, len(best_trace)),
            current_depth=deepest_depth,
        )

    def _should_reopen(
        self,
        candidate: RankedChild,
        child_depth: int,
        prior: SearchRecord,
    ) -> bool:
        if child_depth < prior.g_cost:
            return True
        if child_depth == prior.g_cost and candidate.priority < prior.priority - REOPEN_EPSILON:
            return True
        return False

    def _is_better_progress(
        self,
        candidate: HeuristicBreakdown,
        current_best: HeuristicBreakdown,
    ) -> bool:
        return candidate.progress_key > current_best.progress_key

    def _rank_moves(
        self,
        state: GameState,
        depth: int,
        current_eval: HeuristicBreakdown,
        parent_key: Optional[tuple],
        phase: PhaseConfig,
    ) -> List[RankedChild]:
        ranked_by_child: Dict[tuple, RankedChild] = {}
        current_exposed = current_eval.exposed_low_card_reward
        current_sequence = current_eval.movable_sequence_bonus
        current_productive = current_eval.productive_move_bonus
        current_capacity = current_eval.sequence_capacity_bonus
        current_dead = current_eval.dead_end_penalty

        first_empty_fc = next((i for i, card in enumerate(state.free_cells) if card is None), -1)
        first_empty_cascade = next((i for i, column in enumerate(state.cascades) if not column), -1)

        for move in self._get_legal_moves(state):
            if move.dst_type == "freecell" and state.free_cells[move.dst_idx] is None:
                if move.dst_idx != first_empty_fc:
                    continue

            if move.dst_type == "cascade" and not state.cascades[move.dst_idx]:
                if move.dst_idx != first_empty_cascade:
                    continue

            moved_card = self._moved_card_or_none(state, move)
            if moved_card is None:
                continue

            child_raw = apply_move(state, move)
            child, auto_moves = self._normalize_state(child_raw)
            move_seq = (move,) + auto_moves
            child_key = child.canonical_key()

            if parent_key is not None and child_key == parent_key:
                continue

            child_eval = self._evaluate_state(child)
            expose_gain = max(0, child_eval.exposed_low_card_reward - current_exposed)
            sequence_gain = max(0, child_eval.movable_sequence_bonus - current_sequence)
            productive_gain = max(0, child_eval.productive_move_bonus - current_productive)
            capacity_gain = max(0, child_eval.sequence_capacity_bonus - current_capacity)
            dead_reduction = max(0, current_dead - child_eval.dead_end_penalty)
            foundation_gain = candidate_foundation_gain(state, child)
            creates_empty_cascade = int(
                move.src_type == "cascade"
                and len(state.cascades[move.src_idx]) == move.num_cards
                and move.dst_type != "foundation"
            )
            release_freecell = int(move.src_type == "freecell" and move.dst_type != "freecell")
            move_to_freecell = int(move.dst_type == "freecell")
            lock_risk = self._move_lock_risk(state, move, child_eval)
            foundation_order = 0
            if move.dst_type == "foundation":
                foundation_order = FOUNDATION_ORDER_BONUS
                if is_safe_to_auto_move(moved_card, state.foundations):
                    foundation_order = SAFE_FOUNDATION_ORDER_BONUS

            child_priority = self._frontier_priority(depth + len(move_seq), child_eval, phase)
            order_key = (
                foundation_order * phase.foundation_bias,
                (foundation_gain + expose_gain) * EXPOSE_ORDER_MULTIPLIER * phase.order_bias,
                creates_empty_cascade * EMPTY_CASCADE_ORDER_BONUS * phase.order_bias,
                sequence_gain * SEQUENCE_GAIN_ORDER_MULTIPLIER * phase.order_bias,
                productive_gain * PRODUCTIVE_GAIN_ORDER_MULTIPLIER * phase.order_bias,
                dead_reduction * DEAD_REDUCTION_ORDER_MULTIPLIER * phase.order_bias,
                capacity_gain * phase.order_bias,
                release_freecell * FREE_CELL_RELEASE_BONUS * phase.order_bias,
                -move_to_freecell * phase.freecell_penalty,
                -lock_risk * phase.lock_penalty,
                -child_priority,
            )
            candidate = RankedChild(
                move_seq=move_seq,
                child=child,
                child_key=child_key,
                evaluation=child_eval,
                priority=child_priority,
                order_key=order_key,
            )
            existing = ranked_by_child.get(child_key)
            if existing is None or candidate.order_key > existing.order_key:
                ranked_by_child[child_key] = candidate

        ranked = list(ranked_by_child.values())
        ranked.sort(key=lambda item: item.order_key, reverse=True)
        return ranked

    def _frontier_priority(
        self,
        g_cost: int,
        evaluation: HeuristicBreakdown,
        phase: PhaseConfig,
    ) -> float:
        progress_scalar = (
            evaluation.ready_foundation_reward
            + evaluation.productive_move_bonus
            + evaluation.sequence_capacity_bonus
            + evaluation.exposed_low_card_reward
        )
        return (
            phase.g_weight * g_cost
            + evaluation.heuristic_cost
            - phase.progress_bias * progress_scalar
        )

    def _should_enter_endgame(self, state: GameState) -> bool:
        remaining_cards = 52 - state.cards_on_foundation
        empty_resources = state.empty_free_cells + state.empty_cascades
        sequence_capacity = self._sequence_transfer_capacity(state)
        return (
            remaining_cards <= ENDGAME_REMAINING_CARD_THRESHOLD
            or (remaining_cards <= ENDGAME_REMAINING_CARD_THRESHOLD + 4 and empty_resources >= ENDGAME_RESOURCE_THRESHOLD)
            or (
                remaining_cards <= CONCENTRATED_ENDGAME_REMAINING_CARD_THRESHOLD
                and state.cards_on_foundation >= CONCENTRATED_ENDGAME_FOUNDATION_THRESHOLD
                and empty_resources >= ENDGAME_RESOURCE_THRESHOLD
                and sequence_capacity >= CONCENTRATED_ENDGAME_SEQUENCE_CAPACITY_THRESHOLD
            )
        )

    def _solve_endgame(
        self,
        initial: GameState,
        node_budget: int,
    ) -> Tuple[Optional[Tuple[Move, ...]], int, int, int]:
        self._ensure_caches()
        initial_key = initial.canonical_key()
        if initial_key in self._endgame_cache:
            cached = self._endgame_cache[initial_key]
            depth = len(cached) if cached is not None else 0
            return cached, 0, 0, depth
        prior_failed_budget = self._endgame_fail_budget.get(initial_key, -1)
        if prior_failed_budget >= node_budget:
            return None, 0, 0, 0

        phase = PHASES[-1]
        initial_eval = self._evaluate_state(initial)
        counter = 0
        frontier: List[Tuple[float, int, int, GameState]] = [
            (self._frontier_priority(0, initial_eval, phase), 0, counter, initial)
        ]
        parent_links: Dict[tuple, Tuple[Optional[tuple], Tuple[Move, ...]]] = {
            initial_key: (None, tuple())
        }
        best_record: Dict[tuple, SearchRecord] = {
            initial_key: SearchRecord(0, self._frontier_priority(0, initial_eval, phase))
        }
        expanded = 0
        generated = 1
        deepest_depth = 0

        while frontier and expanded < node_budget:
            priority, g_cost, _counter, state = heapq.heappop(frontier)
            state_key = state.canonical_key()
            record = best_record.get(state_key)
            if record is None or g_cost > record.g_cost or priority > record.priority + REOPEN_EPSILON:
                continue

            expanded += 1
            if state.is_goal():
                solution = tuple(self._reconstruct_links(parent_links, state_key))
                self._endgame_cache[initial_key] = solution
                return solution, expanded, generated, len(solution)

            current_eval = self._evaluate_state(state)
            for candidate in self._rank_moves(state, g_cost, current_eval, parent_links[state_key][0], phase):
                child_depth = g_cost + len(candidate.move_seq)
                prior = best_record.get(candidate.child_key)
                if prior is not None and child_depth >= prior.g_cost and candidate.priority >= prior.priority - REOPEN_EPSILON:
                    continue

                best_record[candidate.child_key] = SearchRecord(child_depth, candidate.priority)
                parent_links[candidate.child_key] = (state_key, candidate.move_seq)
                counter += 1
                generated += 1
                deepest_depth = max(deepest_depth, child_depth)
                heapq.heappush(frontier, (candidate.priority, child_depth, counter, candidate.child))

        self._endgame_fail_budget[initial_key] = max(
            self._endgame_fail_budget.get(initial_key, -1),
            node_budget,
        )
        return None, expanded, generated, deepest_depth

    def _evaluate_state(self, state: GameState) -> HeuristicBreakdown:
        self._ensure_caches()
        key = state.canonical_key()
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached

        legal_moves = self._get_legal_moves(state)
        foundation_remaining = 52 - state.cards_on_foundation
        low_rank_buried_penalty = self._low_rank_buried_penalty(state)
        blocked_next_foundation_penalty = self._blocked_next_foundation_penalty(state)
        trapped_target_penalty = self._trapped_target_penalty(state)
        ready_foundation_reward = self._ready_foundation_reward(state, legal_moves)
        movable_sequence_bonus = self._movable_sequence_bonus(state)
        exposed_low_card_reward = self._exposed_low_card_reward(state)
        mobility_bonus = self._mobility_bonus(state, legal_moves)
        sequence_capacity_bonus = self._sequence_transfer_capacity(state)
        productive_move_bonus = self._productive_move_bonus(state, legal_moves)
        dead_end_penalty = self._dead_end_penalty(state, legal_moves)
        locked_top_penalty = self._locked_top_penalty(state)

        heuristic_cost = (
            FOUNDATION_REMAINING_WEIGHT * foundation_remaining
            + LOW_RANK_BURIED_WEIGHT * low_rank_buried_penalty
            + NEXT_FOUNDATION_BLOCKED_WEIGHT * blocked_next_foundation_penalty
            + TRAPPED_TARGET_WEIGHT * trapped_target_penalty
            + DEAD_END_WEIGHT * dead_end_penalty
            + LOCKED_TOP_WEIGHT * locked_top_penalty
            - READY_FOUNDATION_WEIGHT * ready_foundation_reward
            - EMPTY_FREE_CELL_WEIGHT * state.empty_free_cells
            - EMPTY_CASCADE_WEIGHT * state.empty_cascades
            - MOVABLE_SEQUENCE_WEIGHT * movable_sequence_bonus
            - EXPOSED_LOW_CARD_WEIGHT * exposed_low_card_reward
            - MOBILITY_WEIGHT * mobility_bonus
            - SEQUENCE_CAPACITY_WEIGHT * sequence_capacity_bonus
            - PRODUCTIVE_MOVE_WEIGHT * productive_move_bonus
        )
        progress_key = (
            state.cards_on_foundation,
            ready_foundation_reward,
            exposed_low_card_reward,
            productive_move_bonus,
            state.empty_cascades,
            state.empty_free_cells,
            sequence_capacity_bonus,
            movable_sequence_bonus,
            mobility_bonus,
            -dead_end_penalty,
            -blocked_next_foundation_penalty,
            -low_rank_buried_penalty,
            -trapped_target_penalty,
            -locked_top_penalty,
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
            sequence_capacity_bonus=sequence_capacity_bonus,
            productive_move_bonus=productive_move_bonus,
            dead_end_penalty=dead_end_penalty,
            locked_top_penalty=locked_top_penalty,
            heuristic_cost=heuristic_cost,
            progress_key=progress_key,
        )
        self._eval_cache[key] = cached
        return cached

    def _normalize_state(self, state: GameState) -> Tuple[GameState, Tuple[Move, ...]]:
        self._ensure_caches()
        key = self._exact_state_key(state)
        cached = self._normalize_cache.get(key)
        if cached is not None:
            return cached

        current = state
        auto_moves: List[Move] = []
        while True:
            safe_move = self._pick_safe_foundation_move(current)
            if safe_move is None:
                break
            current = apply_move(current, safe_move)
            auto_moves.append(safe_move)

        cached = (current, tuple(auto_moves))
        self._normalize_cache[key] = cached
        return cached

    def _collapse_safe_foundations(self, state: GameState) -> Tuple[GameState, Tuple[Move, ...]]:
        return self._normalize_state(state)

    def _pick_safe_foundation_move(self, state: GameState) -> Optional[Move]:
        best_move: Optional[Move] = None
        best_rank = 99
        for move in self._get_legal_moves(state):
            if move.dst_type != "foundation":
                continue
            moved_card = self._moved_card_or_none(state, move)
            if moved_card is None or not is_safe_to_auto_move(moved_card, state.foundations):
                continue
            if moved_card.rank < best_rank:
                best_rank = moved_card.rank
                best_move = move
        return best_move

    def _get_legal_moves(self, state: GameState) -> Tuple[Move, ...]:
        self._ensure_caches()
        key = self._exact_state_key(state)
        cached = self._legal_moves_cache.get(key)
        if cached is None:
            cached = tuple(get_valid_moves(state))
            self._legal_moves_cache[key] = cached
        return cached

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
                card is not None and card.suit == suit and card.rank == needed_rank
                for card in state.free_cells
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
                if important and depth_from_top > 0:
                    penalty += depth_from_top * max(2, self._low_rank_weight(min(card.rank, 4)))
        return penalty

    def _ready_foundation_reward(self, state: GameState, legal_moves: Tuple[Move, ...]) -> int:
        reward = 0
        for move in legal_moves:
            if move.dst_type != "foundation":
                continue
            moved_card = self._moved_card_or_none(state, move)
            if moved_card is None:
                continue
            reward += 2 if is_safe_to_auto_move(moved_card, state.foundations) else 1
        return reward

    def _movable_sequence_bonus(self, state: GameState) -> int:
        bonus = 0
        capacity = self._sequence_transfer_capacity(state)
        for column in state.cascades:
            run = self._top_run_length(column)
            if run >= 2:
                bonus += run * min(run, capacity)
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

    def _mobility_bonus(self, state: GameState, legal_moves: Tuple[Move, ...]) -> int:
        cascade_targets = sum(1 for move in legal_moves if move.dst_type == "cascade")
        release_freecells = sum(1 for move in legal_moves if move.src_type == "freecell")
        return (
            state.empty_free_cells * 2
            + state.empty_cascades * 4
            + min(18, cascade_targets)
            + min(8, release_freecells)
        )

    def _sequence_transfer_capacity(self, state: GameState) -> int:
        key = self._exact_state_key(state)
        cached = self._sequence_capacity_cache.get(key)
        if cached is None:
            cached = (state.empty_free_cells + 1) * (2 ** state.empty_cascades)
            self._sequence_capacity_cache[key] = cached
        return cached

    def _productive_move_bonus(self, state: GameState, legal_moves: Tuple[Move, ...]) -> int:
        key = state.canonical_key()
        cached = self._productive_cache.get(key)
        if cached is not None:
            return cached

        productive = 0
        for move in legal_moves:
            if move.dst_type == "foundation":
                productive += 2
                continue
            if move.src_type == "freecell" and move.dst_type == "cascade":
                productive += 2
                continue
            if (
                move.src_type == "cascade"
                and len(state.cascades[move.src_idx]) == move.num_cards
                and move.dst_type != "foundation"
            ):
                productive += 2
                continue
            if move.dst_type == "cascade" and state.cascades[move.dst_idx]:
                productive += 1

        self._productive_cache[key] = productive
        return productive

    def _dead_end_penalty(self, state: GameState, legal_moves: Tuple[Move, ...]) -> int:
        key = state.canonical_key()
        cached = self._dead_penalty_cache.get(key)
        if cached is not None:
            return cached

        penalty = 0
        productive = self._productive_move_bonus(state, legal_moves)
        capacity = self._sequence_transfer_capacity(state)
        if state.empty_free_cells == 0 and state.empty_cascades == 0:
            penalty += 6
        if productive <= 1:
            penalty += 6
        if capacity <= 1:
            penalty += 4
        if state.empty_free_cells == 0 and self._blocked_next_foundation_penalty(state) >= 6:
            penalty += 5
        if not any(move.dst_type == "foundation" for move in legal_moves) and productive == 0:
            penalty += 8

        self._dead_penalty_cache[key] = penalty
        return penalty

    def _locked_top_penalty(self, state: GameState) -> int:
        key = state.canonical_key()
        cached = self._locked_top_cache.get(key)
        if cached is not None:
            return cached

        visible_cards = [column[-1] for column in state.cascades if column]
        visible_cards.extend(card for card in state.free_cells if card is not None)
        penalty = 0
        for card in visible_cards:
            if state.foundations[card.suit] >= card.rank:
                continue
            stack_options = sum(
                1
                for other in visible_cards
                if other is not card and card.can_stack_on(other)
            )
            if stack_options == 0 and card.rank > state.foundations[card.suit] + 1:
                penalty += 1

        self._locked_top_cache[key] = penalty
        return penalty

    def _move_lock_risk(self, state: GameState, move: Move, child_eval: HeuristicBreakdown) -> int:
        moved_card = self._moved_card_or_none(state, move)
        if moved_card is None:
            return 0
        risk = 0
        if move.dst_type == "freecell" and moved_card.rank > 8:
            risk += 1
        if move.dst_type == "cascade" and state.cascades[move.dst_idx]:
            dst_top = state.cascades[move.dst_idx][-1]
            if moved_card.is_red == dst_top.is_red:
                risk += 2
        if child_eval.dead_end_penalty > self._dead_end_penalty(state, self._get_legal_moves(state)):
            risk += 1
        return risk

    def _top_run_length(self, column: Tuple[Card, ...]) -> int:
        if not column:
            return 0
        run = 1
        idx = len(column) - 1
        while idx > 0 and column[idx].can_stack_on(column[idx - 1]):
            run += 1
            idx -= 1
        return run

    def _low_rank_weight(self, rank: int) -> int:
        return {1: 12, 2: 9, 3: 6, 4: 4}.get(rank, 1)

    def _moved_card_or_none(self, state: GameState, move: Move) -> Optional[Card]:
        if move.src_type == "cascade":
            column = state.cascades[move.src_idx]
            if len(column) < move.num_cards:
                return None
            return column[-move.num_cards]
        return state.free_cells[move.src_idx]

    def _exact_state_key(self, state: GameState) -> tuple:
        free_cells = tuple(card.to_int() if card is not None else -1 for card in state.free_cells)
        return (state.cascades, free_cells, state.foundations)

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


def candidate_foundation_gain(before: GameState, after: GameState) -> int:
    return after.cards_on_foundation - before.cards_on_foundation
