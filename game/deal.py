"""Compatibility wrappers around the exact Microsoft deal module."""

from __future__ import annotations

from .ms_deals import (
    get_visual_top_row,
    microsoft_deck,
    ms_numbered_deal,
    ms_rand_stream,
    random_board,
    validate_deal,
)


def ms_deal(game_number: int):
    return ms_numbered_deal(game_number)


def random_deal(seed: int | None = None):
    return random_board(seed)


__all__ = [
    "get_visual_top_row",
    "microsoft_deck",
    "ms_deal",
    "ms_numbered_deal",
    "ms_rand_stream",
    "random_board",
    "random_deal",
    "validate_deal",
]
