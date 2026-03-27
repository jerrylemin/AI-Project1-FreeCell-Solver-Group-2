"""Exact Microsoft FreeCell numbered deals and separate random-board helpers."""

from __future__ import annotations

import random
from typing import Generator, Iterable, List, Sequence

from .card import Card


def ms_rand_stream(seed: int) -> Generator[int, None, None]:
    """Yield values from the classic 15-bit Microsoft C rand() sequence."""
    state = seed & 0x7FFFFFFF
    while True:
        state = (214013 * state + 2531011) % (1 << 31)
        yield (state >> 16) & 0x7FFF


def microsoft_deck() -> List[Card]:
    """Return the standard Microsoft FreeCell deck order."""
    return [Card(rank, suit) for rank in range(1, 14) for suit in range(4)]


def ms_numbered_deal(seed: int) -> List[List[Card]]:
    """
    Generate the exact Microsoft FreeCell numbered deal for ``seed``.

    Algorithm:
    - Start with the standard Microsoft deck order:
      AC AD AH AS, 2C 2D 2H 2S, ..., KC KD KH KS
    - Use the classic Microsoft 15-bit rand() LCG seeded by the deal number
    - Repeatedly select one card from the shrinking active deck
    - Deal selected cards row-wise across the eight cascades
    """
    deck = microsoft_deck()
    dealt_cards: List[Card] = []
    rand_stream = ms_rand_stream(seed)

    for remaining_length in range(len(deck), 0, -1):
        r = next(rand_stream) % remaining_length
        selected = deck[r]
        deck[r], deck[remaining_length - 1] = deck[remaining_length - 1], deck[r]
        dealt_cards.append(selected)

    cascades: List[List[Card]] = [[] for _ in range(8)]
    for index, card in enumerate(dealt_cards):
        cascades[index % 8].append(card)
    return cascades


def random_board(seed: int | None = None) -> List[List[Card]]:
    """Generate a true random shuffled board, separate from Microsoft deals."""
    rng = random.Random(seed)
    deck = microsoft_deck()
    rng.shuffle(deck)

    cascades: List[List[Card]] = [[] for _ in range(8)]
    for index, card in enumerate(deck):
        cascades[index % 8].append(card)
    return cascades


def get_visual_top_row(cascades: Sequence[Sequence[Card]]) -> List[str]:
    """Return the first visible row of cards from left to right."""
    return [str(column[0]) for column in cascades if column]


def validate_deal(cascades: Iterable[Iterable[Card]]) -> bool:
    """Return True if ``cascades`` contains exactly one standard 52-card deck."""
    all_cards = [card for column in cascades for card in column]
    if len(all_cards) != 52:
        return False

    seen = set()
    for card in all_cards:
        key = (card.rank, card.suit)
        if key in seen:
            return False
        seen.add(key)
    return True
