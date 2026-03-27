"""Immutable FreeCell game state."""

from __future__ import annotations
from typing import List, Optional, Tuple

from .card import Card, SUIT_SYMBOLS


class GameState:
    """
    Complete, immutable snapshot of a FreeCell position.

    Attributes
    ----------
    cascades   : tuple[tuple[Card, ...], ...] – 8 columns, bottom→top
    free_cells : tuple[Card | None, ...]      – 4 free-cell slots
    foundations: tuple[int, ...]              – 4 suits; value = highest rank placed (0 = empty)
    """

    __slots__ = ('cascades', 'free_cells', 'foundations', '_hash', '_key')

    def __init__(
        self,
        cascades:    Tuple[Tuple[Optional[Card], ...], ...],
        free_cells:  Tuple[Optional[Card], ...],
        foundations: Tuple[int, ...],
    ):
        self.cascades    = tuple(tuple(col) for col in cascades)
        self.free_cells  = tuple(free_cells)
        self.foundations = tuple(foundations)
        self._hash: Optional[int]   = None
        self._key:  Optional[tuple] = None

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def initial(cls, dealt_cascades: List[List[Card]]) -> 'GameState':
        """Build the starting state from the 8 dealt columns."""
        return cls(
            cascades    = tuple(tuple(col) for col in dealt_cascades),
            free_cells  = (None, None, None, None),
            foundations = (0, 0, 0, 0),
        )

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def empty_free_cells(self) -> int:
        return sum(1 for fc in self.free_cells if fc is None)

    @property
    def empty_cascades(self) -> int:
        return sum(1 for col in self.cascades if len(col) == 0)

    @property
    def cards_on_foundation(self) -> int:
        return sum(self.foundations)

    def is_goal(self) -> bool:
        return all(f == 13 for f in self.foundations)

    # ── Hashing / equality ───────────────────────────────────────────────────

    def canonical_key(self) -> tuple:
        """
        Canonical key for visited-set deduplication.
        Free cells are treated as a multiset (order irrelevant).
        """
        if self._key is None:
            fc_sorted = tuple(sorted(
                (fc.to_int() if fc is not None else -1) for fc in self.free_cells
            ))
            self._key = (self.cascades, fc_sorted, self.foundations)
        return self._key

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.canonical_key())
        return self._hash

    def __eq__(self, other) -> bool:
        if not isinstance(other, GameState):
            return False
        return self.canonical_key() == other.canonical_key()

    # ── Debug repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        fc = ' '.join(str(c) if c else '[ ]' for c in self.free_cells)
        fd = ' '.join(
            f"{self.foundations[s]}{SUIT_SYMBOLS[s]}" for s in range(4)
        )
        rows = [f"FC: {fc}   FD: {fd}"]
        for i, col in enumerate(self.cascades):
            rows.append(f"  C{i+1}: {' '.join(str(c) for c in col)}")
        return '\n'.join(rows)
