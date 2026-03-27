import unittest
from unittest import mock

import gui.app as app_module
from game.deal import get_visual_top_row, ms_deal, validate_deal


RANKS = "A23456789TJQK"
SUITS = "CDHS"


def card_text(card) -> str:
    return f"{RANKS[card.rank - 1]}{SUITS[card.suit]}"


def board_text(cascades):
    return [[card_text(card) for card in column] for column in cascades]


DEAL_1 = [
    ["JD", "KD", "2S", "4C", "3S", "6D", "6S"],
    ["2D", "KC", "KS", "5C", "TD", "8S", "9C"],
    ["9H", "9S", "9D", "TS", "4S", "8D", "2H"],
    ["JC", "5S", "QD", "QH", "TH", "QS", "6H"],
    ["5D", "AD", "JS", "4H", "8H", "6C"],
    ["7H", "QC", "AS", "AC", "2C", "3D"],
    ["7C", "KH", "AH", "4D", "JH", "8C"],
    ["5H", "3H", "3C", "7S", "7D", "TC"],
]

DEAL_617 = [
    ["7D", "TD", "TH", "KD", "4C", "4S", "JD"],
    ["AD", "7S", "QC", "5H", "QS", "TS", "KS"],
    ["5C", "QD", "3H", "9S", "9C", "2H", "KC"],
    ["3S", "AC", "9D", "3C", "9H", "5D", "4H"],
    ["5S", "6D", "6S", "8S", "7C", "JC"],
    ["8C", "8H", "8D", "7H", "6H", "6C"],
    ["2D", "AS", "3D", "4D", "2C", "JH"],
    ["AH", "KH", "TC", "JS", "2S", "QH"],
]


class _DummyDialog:
    def __init__(self, parent, result, on_play):
        self.parent = parent
        self.result = result
        self.on_play = on_play


class MicrosoftDealTests(unittest.TestCase):
    def test_deal_1_matches_known_microsoft_layout(self):
        self.assertEqual(board_text(ms_deal(1)), DEAL_1)

    def test_deal_617_matches_known_microsoft_layout(self):
        self.assertEqual(board_text(ms_deal(617)), DEAL_617)

    def test_deal_164_top_row_is_deterministic(self):
        row1 = get_visual_top_row(ms_deal(164))
        row2 = get_visual_top_row(ms_deal(164))
        self.assertEqual(row1, row2)
        self.assertEqual(len(row1), 8)

    def test_same_numbered_deal_loaded_twice_is_identical(self):
        self.assertEqual(board_text(ms_deal(164)), board_text(ms_deal(164)))

    def test_numbered_deal_is_valid_deck(self):
        self.assertTrue(validate_deal(ms_deal(1)))

    def test_restart_restores_exact_numbered_deal(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = app_module.FreeCellApp()
            app.withdraw()
            try:
                app._load_microsoft_deal(617)
                initial = board_text(app._initial_cascades)
                app._load_random_board(seed=7)
                app._load_microsoft_deal(617)
                self.assertEqual(board_text(app._initial_cascades), initial)
                app._restart()
                self.assertEqual(board_text(app._initial_cascades), initial)
            finally:
                app.destroy()

    def test_numbered_deal_and_random_mode_do_not_share_label_or_path(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = app_module.FreeCellApp()
            app.withdraw()
            try:
                app._load_microsoft_deal(164)
                numbered_label = app._board_source_var.get()
                numbered_board = board_text(app._initial_cascades)
                app._load_random_board(seed=164)
                random_label = app._board_source_var.get()
                random_board = board_text(app._initial_cascades)
                self.assertEqual(numbered_label, "Microsoft Deal #164")
                self.assertEqual(random_label, "Random Shuffled Board")
                self.assertNotEqual(numbered_board, random_board)
            finally:
                app.destroy()


if __name__ == "__main__":
    unittest.main()
