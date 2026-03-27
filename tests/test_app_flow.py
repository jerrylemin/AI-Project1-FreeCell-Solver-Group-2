import time
import unittest
from unittest import mock

import gui.app as app_module
from game.deal import validate_deal
from game.moves import apply_auto_moves, apply_move, get_valid_moves
from solvers.base import SolverResult


class _DummyDialog:
    def __init__(self, parent, result, on_play):
        self.parent = parent
        self.result = result
        self.on_play = on_play


class _CaptureSolver:
    def __init__(self):
        self.name = "Capture"
        self.MAX_NODES = 0
        self.seen_state = None

    def request_stop(self) -> None:
        return None

    def solve(self, state):
        self.seen_state = state
        return SolverResult(
            algorithm=self.name,
            solved=False,
            replay_trace=[],
            best_trace_length=0,
            status="Failed, no solution found",
            message="capture",
        )


class FreeCellAppFlowTests(unittest.TestCase):
    def _make_app(self):
        app = app_module.FreeCellApp()
        app.withdraw()
        return app

    def _wait_for_solver(self, app, timeout_s: float = 2.0):
        deadline = time.time() + timeout_s
        while (app._solving or not app._solver_progress_queue.empty()) and time.time() < deadline:
            app.update()
            time.sleep(0.01)
        app.update()

    def test_blank_new_game_loads_random_microsoft_deal(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog), \
             mock.patch.object(app_module.simpledialog, "askstring", return_value=""), \
             mock.patch.object(app_module.random, "randint", return_value=164):
            app = self._make_app()
            try:
                app._new_game_dialog()
                self.assertEqual(app._deal_number, 164)
                self.assertEqual(app._board_source_var.get(), "Microsoft Deal #164")
                self.assertTrue(validate_deal(app._initial_cascades))
            finally:
                app.destroy()

    def test_startup_board_source_labeling(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = self._make_app()
            try:
                self.assertEqual(app._board_source_var.get(), "Microsoft Deal #1")
                self.assertEqual(app._deal_number, 1)
            finally:
                app.destroy()

    def test_restart_restores_current_game_initial_board(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = self._make_app()
            try:
                app._load_random_board(seed=7)
                initial_key = app._state.canonical_key()
                move = get_valid_moves(app._state)[0]
                app._state = apply_auto_moves(apply_move(app._state, move))
                self.assertNotEqual(app._state.canonical_key(), initial_key)
                app._restart()
                self.assertEqual(app._state.canonical_key(), initial_key)
                self.assertEqual(app._board_source_var.get(), "Random Shuffled Board")
            finally:
                app.destroy()

    def test_random_board_path_is_explicitly_distinct_from_microsoft_deals(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog), \
             mock.patch.object(app_module.simpledialog, "askstring", return_value="RANDOM"):
            app = self._make_app()
            try:
                app._new_game_dialog()
                self.assertIsNone(app._deal_number)
                self.assertEqual(app._board_source_var.get(), "Random Shuffled Board")
                self.assertTrue(validate_deal(app._initial_cascades))
            finally:
                app.destroy()

    def test_sample_board_source_labeling(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = self._make_app()
            try:
                app._load_sample_board("easy_demo")
                self.assertEqual(app._board_source_var.get(), "Sample Board: easy_demo")
                self.assertEqual(app._state.foundations, (12, 13, 13, 13))
            finally:
                app.destroy()

    def test_solver_uses_current_visible_board_not_startup_board(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = self._make_app()
            try:
                move = get_valid_moves(app._state)[0]
                app._state = apply_auto_moves(apply_move(app._state, move))
                expected_key = app._state.canonical_key()
                capture = _CaptureSolver()
                app._create_solver = lambda algo, progress_callback: capture
                app._max_nodes_var.set("25")
                app._run_solver("BFS")
                self._wait_for_solver(app)
                self.assertIsNotNone(capture.seen_state)
                self.assertEqual(capture.seen_state.canonical_key(), expected_key)
            finally:
                app.destroy()


if __name__ == "__main__":
    unittest.main()
