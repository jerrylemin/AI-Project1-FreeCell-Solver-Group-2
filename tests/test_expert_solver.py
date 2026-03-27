import unittest
from unittest import mock

from game.card import Card
from game.deal import ms_deal
from game.moves import apply_move, get_valid_moves
from game.samples import get_sample_board
from game.state import GameState
import solvers.expert_solver as expert_module
from solvers.expert_solver import ExpertSolver


class ExpertSolverTests(unittest.TestCase):
    def _prepare_private_caches(self, solver: ExpertSolver):
        solver._eval_cache = {}
        solver._legal_moves_cache = {}
        solver._closure_cache = {}

    def _assert_trace_legal(self, initial: GameState, trace):
        state = initial
        for move in trace:
            self.assertIn(move, get_valid_moves(state))
            state = apply_move(state, move)
        return state

    def test_expert_solver_returns_legal_move_sequences(self):
        initial = get_sample_board("medium_demo")
        solver = ExpertSolver(use_auto_moves=False)
        solver.MAX_NODES = 10_000

        result = solver.solve(initial)

        self.assertTrue(result.solved)
        final_state = self._assert_trace_legal(initial, result.solution)
        self.assertTrue(final_state.is_goal())
        self.assertEqual(result.replay_trace, result.solution)

    def test_expert_solver_solves_near_goal(self):
        near_goal = GameState(
            cascades=((), (), (), (), (), (), (), ()),
            free_cells=(Card(13, 0), None, None, None),
            foundations=(12, 13, 13, 13),
        )
        solver = ExpertSolver(use_auto_moves=False)
        solver.MAX_NODES = 100

        result = solver.solve(near_goal)

        self.assertTrue(result.solved)
        self.assertEqual(result.status, "Expert Solver solved")
        self.assertEqual(result.solution_length, 1)
        self.assertTrue(self._assert_trace_legal(near_goal, result.solution).is_goal())

    def test_expert_solver_solves_easy_demo(self):
        initial = get_sample_board("easy_demo")
        solver = ExpertSolver(use_auto_moves=False)
        solver.MAX_NODES = 100

        result = solver.solve(initial)

        self.assertTrue(result.solved)
        self.assertEqual(result.status, "Expert Solver solved")
        self.assertEqual(result.solution_length, 1)
        self.assertTrue(self._assert_trace_legal(initial, result.solution).is_goal())

    def test_expert_solver_keeps_replayable_best_progress_trace_on_stop(self):
        initial = GameState.initial(ms_deal(164))
        holder = {}

        def callback(snapshot):
            if snapshot.moves > 0:
                holder["solver"].request_stop()

        solver = ExpertSolver(use_auto_moves=False, progress_callback=callback)
        holder["solver"] = solver
        solver.MAX_NODES = 25_000
        solver.PROGRESS_INTERVAL_S = 0.0

        result = solver.solve(initial)

        self.assertFalse(result.solved)
        self.assertEqual(result.status, "Expert Solver stopped")
        self.assertTrue(result.replay_available)
        self.assertGreater(result.replay_length, 0)
        self.assertEqual(result.replay_label, "Replay Failed Attempt")
        self.assertIn("best-progress", result.replay_message)
        self._assert_trace_legal(initial, result.replay_trace)

    def test_expert_solver_uses_shared_rules_engine(self):
        initial = get_sample_board("medium_demo")
        solver = ExpertSolver(use_auto_moves=False)
        solver.MAX_NODES = 10_000

        with mock.patch.object(
            expert_module,
            "get_valid_moves",
            wraps=expert_module.get_valid_moves,
        ) as mocked_moves, mock.patch.object(
            expert_module,
            "apply_move",
            wraps=expert_module.apply_move,
        ) as mocked_apply:
            result = solver.solve(initial)

        self.assertTrue(result.solved)
        self.assertGreater(mocked_moves.call_count, 0)
        self.assertGreater(mocked_apply.call_count, 0)

    def test_expert_solver_does_not_mutate_states_in_place(self):
        initial = get_sample_board("medium_demo")
        original_key = initial.canonical_key()
        original_repr = repr(initial)
        original_cascades = initial.cascades
        original_free_cells = initial.free_cells
        original_foundations = initial.foundations

        solver = ExpertSolver(use_auto_moves=False)
        solver.MAX_NODES = 10_000
        solver.solve(initial)

        self.assertEqual(initial.canonical_key(), original_key)
        self.assertEqual(repr(initial), original_repr)
        self.assertIs(initial.cascades, original_cascades)
        self.assertIs(initial.free_cells, original_free_cells)
        self.assertIs(initial.foundations, original_foundations)

    def test_expert_solver_legal_move_cache_is_freecell_slot_sensitive(self):
        ace = Card(1, 3)
        state_a = GameState(
            cascades=((), (), (), (), (), (), (), ()),
            free_cells=(ace, None, None, None),
            foundations=(0, 0, 0, 0),
        )
        state_b = GameState(
            cascades=((), (), (), (), (), (), (), ()),
            free_cells=(None, ace, None, None),
            foundations=(0, 0, 0, 0),
        )

        self.assertEqual(state_a.canonical_key(), state_b.canonical_key())

        solver = ExpertSolver(use_auto_moves=False)
        self._prepare_private_caches(solver)

        moves_a = solver._get_legal_moves(state_a)
        moves_b = solver._get_legal_moves(state_b)

        self.assertTrue(any(move.src_type == "freecell" and move.src_idx == 0 for move in moves_a))
        self.assertFalse(any(move.src_type == "freecell" and move.src_idx == 1 for move in moves_a))
        self.assertTrue(any(move.src_type == "freecell" and move.src_idx == 1 for move in moves_b))
        self.assertFalse(any(move.src_type == "freecell" and move.src_idx == 0 for move in moves_b))

    def test_expert_solver_safe_foundation_cache_is_freecell_slot_sensitive(self):
        ace = Card(1, 3)
        state_a = GameState(
            cascades=((), (), (), (), (), (), (), ()),
            free_cells=(ace, None, None, None),
            foundations=(0, 0, 0, 0),
        )
        state_b = GameState(
            cascades=((), (), (), (), (), (), (), ()),
            free_cells=(None, ace, None, None),
            foundations=(0, 0, 0, 0),
        )

        self.assertEqual(state_a.canonical_key(), state_b.canonical_key())

        solver = ExpertSolver(use_auto_moves=False)
        self._prepare_private_caches(solver)

        collapsed_a, auto_moves_a = solver._collapse_safe_foundations(state_a)
        collapsed_b, auto_moves_b = solver._collapse_safe_foundations(state_b)

        self.assertEqual(len(auto_moves_a), 1)
        self.assertEqual(len(auto_moves_b), 1)
        self.assertEqual(auto_moves_a[0].src_type, "freecell")
        self.assertEqual(auto_moves_b[0].src_type, "freecell")
        self.assertEqual(auto_moves_a[0].src_idx, 0)
        self.assertEqual(auto_moves_b[0].src_idx, 1)
        self.assertEqual(collapsed_a.foundations[3], 1)
        self.assertEqual(collapsed_b.foundations[3], 1)
        self.assertEqual(collapsed_a.free_cells, (None, None, None, None))
        self.assertEqual(collapsed_b.free_cells, (None, None, None, None))


if __name__ == "__main__":
    unittest.main()
