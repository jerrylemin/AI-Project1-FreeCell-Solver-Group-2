import unittest

from game.card import Card
from game.moves import Move, apply_move, get_valid_moves
from game.state import GameState
from gui.app import FreeCellApp
from gui.app import MIN_VISIBLE_CASCADE_STRIP, compute_cascade_spacing


class LayoutAndRulesTests(unittest.TestCase):
    def test_dense_cascade_spacing_preserves_visible_identity_strip(self):
        spacing = compute_cascade_spacing(
            column_height=720 - 168 - 18,
            card_count=20,
            card_height=110,
            min_overlap_visible=MIN_VISIBLE_CASCADE_STRIP,
        )

        self.assertGreaterEqual(spacing, MIN_VISIBLE_CASCADE_STRIP)

    def test_rules_engine_keeps_move_legality_consistent(self):
        state = GameState(
            cascades=((Card(2, 0),), (), (), (), (), (), (), ()),
            free_cells=(Card(1, 0), None, None, None),
            foundations=(0, 0, 0, 0),
        )

        moves = get_valid_moves(state)

        self.assertIn(Move("freecell", 0, "foundation", 0), moves)
        self.assertIn(Move("cascade", 0, "freecell", 1), moves)

        next_state = apply_move(state, Move("freecell", 0, "foundation", 0))
        self.assertEqual(next_state.foundations, (1, 0, 0, 0))
        self.assertEqual(state.foundations, (0, 0, 0, 0))

    def test_canvas_cards_include_mirrored_corner_markers(self):
        app = FreeCellApp()
        app.withdraw()
        try:
            items = app._card_items[0]
            self.assertIn("top_rank_right", items)
            self.assertIn("top_suit_right", items)
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
