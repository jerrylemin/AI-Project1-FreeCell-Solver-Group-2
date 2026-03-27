import time
import unittest
from unittest import mock

import gui.app as app_module
from game.card import Card
from game.moves import Move
from game.state import GameState
from solvers.base import SolverResult


class _DummyDialog:
    def __init__(self, parent, result, on_play):
        self.parent = parent
        self.result = result
        self.on_play = on_play


class ReplayControlTests(unittest.TestCase):
    def _make_app(self):
        app = app_module.FreeCellApp()
        app.withdraw()
        return app

    def _pump(self, app, timeout_s: float = 2.0, predicate=None):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            app.update()
            if predicate is not None and predicate():
                return
            time.sleep(0.01)
        app.update()

    def _set_custom_base_state(self, app):
        state = GameState(
            cascades=((), (), (), (), (), (), (), ()),
            free_cells=(Card(12, 0), Card(13, 0), None, None),
            foundations=(11, 13, 13, 13),
        )
        app._initial_state = state
        app._state = state
        app._initial_cascades = [list(col) for col in state.cascades]
        app._render()
        return state

    def test_success_replay_trace_is_committed(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = self._make_app()
            try:
                self._set_custom_base_state(app)
                result = SolverResult(
                    algorithm="BFS",
                    solved=True,
                    solution=[Move("freecell", 0, "foundation", 0)],
                    replay_trace=[Move("freecell", 0, "foundation", 0)],
                    best_trace_length=1,
                    status="Solved",
                )
                app._solver_done(result)
                self.assertEqual(len(app._replay_moves), 1)
                self.assertEqual(app._btn_play.cget("state"), "normal")
                self.assertEqual(app._btn_step.cget("state"), "normal")
            finally:
                app.destroy()

    def test_stop_replay_trace_is_committed_and_controls_enable(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = self._make_app()
            try:
                self._set_custom_base_state(app)
                result = SolverResult(
                    algorithm="A*",
                    solved=False,
                    replay_trace=[Move("freecell", 0, "foundation", 0)],
                    best_trace_length=1,
                    replay_label="Replay Failed Attempt",
                    replay_message="Stopped by user, showing best-progress trace.",
                    status="Stopped by user",
                )
                app._solver_done(result)
                self.assertEqual(len(app._replay_moves), 1)
                self.assertEqual(app._btn_play.cget("state"), "normal")
                self.assertEqual(app._btn_step.cget("state"), "normal")
            finally:
                app.destroy()

    def test_back_decrements_replay_cursor(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = self._make_app()
            try:
                self._set_custom_base_state(app)
                trace = [
                    Move("freecell", 0, "foundation", 0),
                    Move("freecell", 1, "foundation", 0),
                ]
                app._load_replay_trace(trace, autoplay=False, is_solution_trace=True)
                app._step_replay()
                self._pump(app, predicate=lambda: app._replay_cursor == 1 and app._active_animation is None)
                self.assertEqual(app._replay_cursor, 1)
                app._back_replay()
                self.assertEqual(app._replay_cursor, 0)
            finally:
                app.destroy()

    def test_pause_preserves_replay_cursor(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = self._make_app()
            try:
                self._set_custom_base_state(app)
                trace = [
                    Move("freecell", 0, "foundation", 0),
                    Move("freecell", 1, "foundation", 0),
                ]
                app._load_replay_trace(trace, autoplay=False, is_solution_trace=True)
                app._play_replay()
                self._pump(app, predicate=lambda: app._replay_cursor >= 1)
                app._pause_replay()
                cursor = app._replay_cursor
                self._pump(app, timeout_s=0.3)
                self.assertEqual(app._replay_cursor, cursor)
            finally:
                app.destroy()

    def test_restart_clears_replay_state_safely(self):
        with mock.patch.object(app_module, "ResultsDialog", _DummyDialog):
            app = self._make_app()
            try:
                self._set_custom_base_state(app)
                app._load_replay_trace([Move("freecell", 0, "foundation", 0)], autoplay=False)
                self.assertEqual(len(app._replay_moves), 1)
                app._restart()
                self.assertEqual(len(app._replay_moves), 0)
                self.assertEqual(app._replay_cursor, 0)
                self.assertEqual(app._btn_play.cget("state"), "disabled")
            finally:
                app.destroy()


if __name__ == "__main__":
    unittest.main()
