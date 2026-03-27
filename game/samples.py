"""Named sample boards for quick solver and UI checks."""

from __future__ import annotations

from game.card import Card
from game.state import GameState


def _easy_demo() -> GameState:
    return GameState(
        cascades=((), (), (), (), (), (), (), ()),
        free_cells=(Card(13, 0), None, None, None),
        foundations=(12, 13, 13, 13),
    )


def _medium_demo() -> GameState:
    return GameState(
        cascades=(
            (Card(13, 0), Card(12, 0), Card(11, 0)),
            (Card(13, 1), Card(12, 1), Card(11, 1)),
            (Card(13, 2), Card(12, 2), Card(11, 2)),
            (Card(13, 3), Card(12, 3), Card(11, 3)),
            (),
            (),
            (),
            (),
        ),
        free_cells=(None, None, None, None),
        foundations=(10, 10, 10, 10),
    )


SAMPLE_BOARDS = {
    "easy_demo": _easy_demo,
    "medium_demo": _medium_demo,
}

SAMPLE_BOARD_NAMES = tuple(SAMPLE_BOARDS.keys())


def get_sample_board(name: str) -> GameState:
    if name not in SAMPLE_BOARDS:
        raise KeyError(name)
    return SAMPLE_BOARDS[name]()
