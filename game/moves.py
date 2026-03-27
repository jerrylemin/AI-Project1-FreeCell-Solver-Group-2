"""Move generation and application for FreeCell.

Move types
----------
Every move is represented as a (source, destination) pair where each
location is one of:
  ('cascade',    col_index)   – top card(s) of a cascade column
  ('freecell',   slot_index)  – a free-cell slot
  ('foundation', suit_index)  – a foundation pile

Sequence moves (multiple cards) are also generated; they are tagged with
the number of cards involved so the GUI can animate them properly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .card import Card
from .state import GameState


# ── Move dataclass ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Move:
    src_type:  str   # 'cascade' | 'freecell'
    src_idx:   int
    dst_type:  str   # 'cascade' | 'freecell' | 'foundation'
    dst_idx:   int
    num_cards: int = 1   # >1 for sequence moves

    def description(self, state: GameState) -> str:
        """Human-readable description of this move."""
        if self.src_type == 'cascade':
            col   = state.cascades[self.src_idx]
            cards = col[-self.num_cards:]
            card_str = ' '.join(str(c) for c in cards)
        else:
            card_str = str(state.free_cells[self.src_idx])

        if self.dst_type == 'foundation':
            return f"Move {card_str} → foundation"
        elif self.dst_type == 'freecell':
            return f"Move {card_str} → free cell {self.dst_idx + 1}"
        else:
            col = state.cascades[self.dst_idx]
            dest = str(col[-1]) if col else "empty column"
            return f"Move {card_str} → column {self.dst_idx + 1} ({dest})"


# ── Helper: max movable sequence length ──────────────────────────────────────

def _max_seq_len(empty_fc: int, empty_cas: int, dest_is_empty: bool) -> int:
    """
    Maximum number of cards that can be moved as one compound sequence under
    standard FreeCell rules (simulating intermediate placements).
    """
    ec = empty_cas - (1 if dest_is_empty else 0)
    ec = max(ec, 0)
    return (empty_fc + 1) * (2 ** ec)


# ── Auto-move logic ───────────────────────────────────────────────────────────

def is_safe_to_auto_move(card: Card, foundations: Tuple[int, ...]) -> bool:
    """
    A card is *safe* to auto-move to its foundation if no card currently
    in play could ever need to be stacked on top of it.  This holds when
    both opposite-coloured suits have their rank ≥ card.rank − 1 already
    on the foundations.
    """
    r = card.rank
    if r <= 2:            # Aces and 2s are always safe
        return True
    # Cards that could be placed on top of this card are rank-1, opposite colour
    opp_suits = [s for s in range(4) if (s in (1, 2)) != card.is_red]
    return all(foundations[s] >= r - 1 for s in opp_suits)


def apply_auto_moves_with_steps(state: GameState) -> Tuple[GameState, List[Move]]:
    """Apply all safe auto-moves and return the final state plus each step."""
    steps: List[Move] = []
    changed = True

    while changed:
        changed = False
        cascades = [list(col) for col in state.cascades]
        free_cells = list(state.free_cells)
        foundations = list(state.foundations)

        for sources in (
            [('cascade', i) for i in range(8)],
            [('freecell', i) for i in range(4)],
        ):
            for src_type, src_idx in sources:
                if src_type == 'cascade':
                    col = cascades[src_idx]
                    if not col:
                        continue
                    card = col[-1]
                else:
                    card = free_cells[src_idx]
                    if card is None:
                        continue

                suit = card.suit
                if (card.rank == foundations[suit] + 1 and
                        is_safe_to_auto_move(card, tuple(foundations))):
                    steps.append(Move(src_type, src_idx, 'foundation', suit))
                    foundations[suit] += 1
                    if src_type == 'cascade':
                        cascades[src_idx].pop()
                    else:
                        free_cells[src_idx] = None
                    changed = True

        state = GameState(
            cascades=tuple(tuple(col) for col in cascades),
            free_cells=tuple(free_cells),
            foundations=tuple(foundations),
        )

    return state, steps


def apply_auto_moves(state: GameState) -> GameState:
    """Repeatedly move safe cards to foundations until no more exist."""
    final_state, _ = apply_auto_moves_with_steps(state)
    return final_state


# ── Move generation ───────────────────────────────────────────────────────────

def get_valid_moves(state: GameState) -> List[Move]:
    """Return all legal moves from *state* (does NOT include auto-moves)."""
    moves: List[Move] = []
    cas    = state.cascades
    fc     = state.free_cells
    fd     = state.foundations
    efc    = state.empty_free_cells
    ecas   = state.empty_cascades

    # ── 1. Cascade top → foundation ───────────────────────────────────────
    for ci, col in enumerate(cas):
        if not col:
            continue
        card = col[-1]
        if card.rank == fd[card.suit] + 1:
            moves.append(Move('cascade', ci, 'foundation', card.suit))

    # ── 2. Free cell → foundation ─────────────────────────────────────────
    for fi, card in enumerate(fc):
        if card is not None and card.rank == fd[card.suit] + 1:
            moves.append(Move('freecell', fi, 'foundation', card.suit))

    # ── 3. Cascade top → free cell ────────────────────────────────────────
    if efc > 0:
        empty_slots = [i for i, c in enumerate(fc) if c is None]
        for ci, col in enumerate(cas):
            if col:
                moves.append(Move('cascade', ci, 'freecell', empty_slots[0]))

    # ── 4. Free cell → cascade ────────────────────────────────────────────
    for fi, card in enumerate(fc):
        if card is None:
            continue
        for ci, col in enumerate(cas):
            if col:
                if card.can_stack_on(col[-1]):
                    moves.append(Move('freecell', fi, 'cascade', ci))
            else:
                moves.append(Move('freecell', fi, 'cascade', ci))

    # ── 5. Cascade → cascade (single card or sequence) ────────────────────
    for src_ci, src_col in enumerate(cas):
        if not src_col:
            continue

        # Find the longest valid sequence at the top of this column
        seq_top = len(src_col) - 1
        seq_start = seq_top
        while seq_start > 0:
            above = src_col[seq_start]
            below = src_col[seq_start - 1]
            if above.can_stack_on(below):
                seq_start -= 1
            else:
                break
        # src_col[seq_start : seq_top+1] is the movable sequence (top = seq_top)

        for dst_ci, dst_col in enumerate(cas):
            if dst_ci == src_ci:
                continue
            dest_is_empty = (len(dst_col) == 0)
            max_len = _max_seq_len(efc, ecas, dest_is_empty)

            for n in range(1, seq_top - seq_start + 2):   # n = sequence length
                if n > max_len:
                    break
                card = src_col[seq_top - n + 1]           # bottom card of sequence
                if dest_is_empty:
                    if n == 1 or True:                     # any single or sequence to empty
                        moves.append(Move('cascade', src_ci, 'cascade', dst_ci, n))
                    break                                  # only one move to an empty column
                else:
                    if card.can_stack_on(dst_col[-1]):
                        moves.append(Move('cascade', src_ci, 'cascade', dst_ci, n))
                        break                             # largest valid sequence to this dest

    return moves


# ── Move application ──────────────────────────────────────────────────────────

def apply_move(state: GameState, move: Move) -> GameState:
    """Return the new GameState after applying *move* to *state*."""
    cascades   = [list(col) for col in state.cascades]
    free_cells = list(state.free_cells)
    foundations = list(state.foundations)

    # Extract cards from source
    if move.src_type == 'cascade':
        col   = cascades[move.src_idx]
        cards = col[-move.num_cards:]
        cascades[move.src_idx] = col[:-move.num_cards]
    else:  # freecell
        cards = [free_cells[move.src_idx]]
        free_cells[move.src_idx] = None

    # Place cards at destination
    if move.dst_type == 'cascade':
        cascades[move.dst_idx].extend(cards)
    elif move.dst_type == 'freecell':
        assert len(cards) == 1
        free_cells[move.dst_idx] = cards[0]
    else:  # foundation
        assert len(cards) == 1
        foundations[cards[0].suit] += 1

    new_state = GameState(
        cascades    = tuple(tuple(c) for c in cascades),
        free_cells  = tuple(free_cells),
        foundations = tuple(foundations),
    )
    return new_state


def apply_move_with_auto(state: GameState, move: Move) -> GameState:
    """Apply *move* then apply all safe auto-moves."""
    return apply_auto_moves(apply_move(state, move))


def apply_move_with_auto_steps(state: GameState, move: Move) -> Tuple[GameState, List[Move]]:
    """Apply *move*, then return the final state plus every visible substep."""
    next_state = apply_move(state, move)
    final_state, auto_steps = apply_auto_moves_with_steps(next_state)
    return final_state, [move, *auto_steps]
