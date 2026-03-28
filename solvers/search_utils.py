"""Shared safe utilities for the required graph-search solvers.

These helpers are intentionally conservative:
they improve engineering efficiency without changing solver identity.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from game.card import Card
from game.moves import Move, get_valid_moves
from game.state import GameState


StateKey = tuple
ParentLinks = Dict[StateKey, Tuple[Optional[StateKey], Optional[Move]]]
MoveCache = Dict[StateKey, Tuple[Move, ...]]


def exact_state_key(state: GameState) -> StateKey:
    """Return an exact immutable state key.

    Unlike ``GameState.canonical_key()``, this preserves free-cell slot
    positions so graph-search solvers do not merge strategically distinct
    states unless that equivalence is proven separately.
    """

    free_cells = tuple(card.to_int() if card is not None else -1 for card in state.free_cells)
    return (state.cascades, free_cells, state.foundations)


def reconstruct_path(parent_links: ParentLinks, state_key: StateKey) -> List[Move]:
    """Recover a path from exact-key parent links."""

    path: List[Move] = []
    current_key = state_key
    while True:
        parent_key, move = parent_links[current_key]
        if move is None:
            break
        path.append(move)
        if parent_key is None:
            break
        current_key = parent_key
    path.reverse()
    return path


def is_immediate_reverse(candidate: Move, previous: Optional[Move]) -> bool:
    """True if ``candidate`` exactly undoes ``previous`` in one step."""

    if previous is None:
        return False
    return (
        candidate.src_type == previous.dst_type
        and candidate.src_idx == previous.dst_idx
        and candidate.dst_type == previous.src_type
        and candidate.dst_idx == previous.src_idx
        and candidate.num_cards == previous.num_cards
    )


def should_prune_immediate_reverse(
    candidate: Move,
    previous: Optional[Move],
    *,
    use_auto_moves: bool,
) -> bool:
    """Prune only the exact local undo when the visible search is pure."""

    return not use_auto_moves and is_immediate_reverse(candidate, previous)


def ordered_legal_moves(state: GameState, move_cache: MoveCache) -> Tuple[Move, ...]:
    """Get legal moves once per exact state, with safe symmetry filtering.

    The ordering is only a tie-break preference. Solver-specific priority
    semantics still decide which node is expanded next.
    """

    key = exact_state_key(state)
    cached = move_cache.get(key)
    if cached is not None:
        return cached

    filtered = _filter_symmetric_empty_destinations(state, get_valid_moves(state))
    cached = tuple(sorted(filtered, key=lambda move: move_order_key(state, move), reverse=True))
    move_cache[key] = cached
    return cached


def move_order_key(state: GameState, move: Move) -> Tuple[int, int, int, int, int, int, int, int, int]:
    """Order moves by domain-friendly preferences without changing identity."""

    moved_card = _moved_card_or_none(state, move)
    foundation_first = int(move.dst_type == "foundation")
    exposed_low_cards = _exposed_low_rank_reward(state, move)
    creates_empty_cascade = int(
        move.src_type == "cascade" and len(state.cascades[move.src_idx]) == move.num_cards
    )
    sequence_mobility = _sequence_mobility_score(state, move)
    releases_free_cell = int(move.src_type == "freecell" and move.dst_type != "freecell")
    free_cell_last = -int(move.dst_type == "freecell")
    longer_sequence_first = move.num_cards
    lower_rank_first = -(moved_card.rank if moved_card is not None else 99)

    return (
        foundation_first,
        exposed_low_cards,
        creates_empty_cascade,
        sequence_mobility,
        releases_free_cell,
        free_cell_last,
        longer_sequence_first,
        lower_rank_first,
        -move.dst_idx,
    )


def _filter_symmetric_empty_destinations(
    state: GameState,
    moves: Iterable[Move],
) -> List[Move]:
    """Collapse truly equivalent empty-destination moves.

    Empty free cells are interchangeable, as are empty cascades. We keep only
    the first empty destination in each family.
    """

    first_empty_free_cell = next(
        (idx for idx, card in enumerate(state.free_cells) if card is None),
        None,
    )
    first_empty_cascade = next(
        (idx for idx, column in enumerate(state.cascades) if not column),
        None,
    )

    filtered: List[Move] = []
    for move in moves:
        if (
            move.dst_type == "freecell"
            and first_empty_free_cell is not None
            and state.free_cells[move.dst_idx] is None
            and move.dst_idx != first_empty_free_cell
        ):
            continue
        if (
            move.dst_type == "cascade"
            and first_empty_cascade is not None
            and not state.cascades[move.dst_idx]
            and move.dst_idx != first_empty_cascade
        ):
            continue
        filtered.append(move)

    return filtered


def _moved_card_or_none(state: GameState, move: Move) -> Optional[Card]:
    if move.src_type == "cascade":
        column = state.cascades[move.src_idx]
        if len(column) < move.num_cards:
            return None
        return column[-move.num_cards]
    return state.free_cells[move.src_idx]


def _exposed_low_rank_reward(state: GameState, move: Move) -> int:
    if move.src_type != "cascade":
        return 0

    source = state.cascades[move.src_idx]
    if len(source) <= move.num_cards:
        return 0

    exposed = source[-move.num_cards - 1]
    if state.foundations[exposed.suit] >= exposed.rank:
        return 0

    reward = 0
    if exposed.rank <= 4:
        reward += 5 - exposed.rank
    if exposed.rank == state.foundations[exposed.suit] + 1:
        reward += 4
    return reward


def _sequence_mobility_score(state: GameState, move: Move) -> int:
    score = 0
    if move.dst_type == "cascade":
        if state.cascades[move.dst_idx]:
            score += 3
        if move.num_cards > 1:
            score += move.num_cards
    if move.src_type == "freecell" and move.dst_type == "cascade":
        score += 2
    return score
