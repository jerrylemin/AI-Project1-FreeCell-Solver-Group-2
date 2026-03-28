import unittest
from unittest import mock

import solvers.astar as astar_module
import solvers.bfs as bfs_module
import solvers.dfs as dfs_module
import solvers.search_utils as search_utils
import solvers.ucs as ucs_module
from game.card import Card
from game.moves import Move, apply_move, get_valid_moves
from game.state import GameState
from solvers.astar import AStarSolver
from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.search_utils import exact_state_key, ordered_legal_moves, should_prune_immediate_reverse
from solvers.ucs import UCSSolver


class FakeState:
    _MARKERS = {
        "root": Card(13, 0),
        "a": Card(12, 1),
        "b": Card(11, 2),
        "c": Card(10, 3),
        "loop": Card(9, 0),
        "cheap": Card(8, 1),
        "expensive_goal": Card(7, 2),
        "cheap_goal": Card(6, 3),
        "good": Card(5, 0),
        "bad": Card(4, 1),
        "good_goal": Card(3, 2),
        "bad_goal": Card(2, 3),
    }

    def __init__(self, name: str, *, goal: bool = False):
        self.name = name
        self._goal = goal
        marker = self._MARKERS[name]
        self.cascades = ((marker,), (), (), (), (), (), (), ())
        self.free_cells = (None, None, None, None)
        self.foundations = (13, 13, 13, 13) if goal else (0, 0, 0, 0)

    @property
    def empty_free_cells(self) -> int:
        return 4

    @property
    def empty_cascades(self) -> int:
        return 7

    @property
    def cards_on_foundation(self) -> int:
        return sum(self.foundations)

    def is_goal(self) -> bool:
        return self._goal

    def __repr__(self) -> str:
        return f"FakeState({self.name!r})"


def near_goal_state() -> GameState:
    return GameState(
        cascades=((), (), (), (), (), (), (), ()),
        free_cells=(Card(13, 0), None, None, None),
        foundations=(12, 13, 13, 13),
    )


class RequiredSolverOptimizationTests(unittest.TestCase):
    def _play_trace(self, state: GameState, trace):
        current = state
        for move in trace:
            self.assertIn(move, get_valid_moves(current))
            next_state = apply_move(current, move)
            self.assertIsNot(next_state, current)
            current = next_state
        return current

    def test_exact_state_key_is_stable_and_slot_sensitive(self):
        state_a = GameState(
            cascades=((), (), (), (), (), (), (), ()),
            free_cells=(Card(1, 0), Card(2, 1), None, None),
            foundations=(0, 0, 0, 0),
        )
        state_b = GameState(
            cascades=((), (), (), (), (), (), (), ()),
            free_cells=(Card(2, 1), Card(1, 0), None, None),
            foundations=(0, 0, 0, 0),
        )

        self.assertEqual(exact_state_key(state_a), exact_state_key(state_a))
        self.assertNotEqual(exact_state_key(state_a), exact_state_key(state_b))
        self.assertEqual(state_a.canonical_key(), state_b.canonical_key())

    def test_apply_move_returns_fresh_state(self):
        state = GameState(
            cascades=((Card(2, 0),), (), (), (), (), (), (), ()),
            free_cells=(Card(1, 0), None, None, None),
            foundations=(0, 0, 0, 0),
        )

        moved = apply_move(state, Move("freecell", 0, "foundation", 0))

        self.assertIsNot(moved, state)
        self.assertEqual(state.foundations, (0, 0, 0, 0))
        self.assertEqual(moved.foundations, (1, 0, 0, 0))

    def test_bfs_still_behaves_as_fifo_graph_search(self):
        root = FakeState("root")
        a = FakeState("a")
        b = FakeState("b")
        c = FakeState("c")
        goal = FakeState("cheap_goal", goal=True)

        to_a = Move("cascade", 0, "cascade", 1)
        to_b = Move("cascade", 0, "freecell", 0)
        to_c = Move("cascade", 0, "cascade", 2)
        to_goal = Move("cascade", 0, "foundation", 0)

        graph = {
            "root": [to_a, to_b],
            "a": [to_c],
            "b": [to_goal],
            "c": [],
        }
        transitions = {
            ("root", to_a): a,
            ("root", to_b): b,
            ("a", to_c): c,
            ("b", to_goal): goal,
        }
        expanded = []

        def fake_moves(state):
            expanded.append(state.name)
            return graph.get(state.name, [])

        def fake_apply(state, move):
            return transitions[(state.name, move)]

        solver = BFSSolver(use_auto_moves=False)
        solver.PROGRESS_INTERVAL_S = 0.0
        with mock.patch.object(search_utils, "get_valid_moves", side_effect=fake_moves), \
             mock.patch.object(bfs_module, "apply_move", side_effect=fake_apply):
            result = solver.solve(root)

        self.assertTrue(result.solved)
        self.assertEqual(result.solution, [to_b, to_goal])
        self.assertEqual(expanded[:3], ["root", "a", "b"])

    def test_ids_still_behaves_as_iterative_deepening_with_cycle_checks(self):
        root = FakeState("root")
        loop = FakeState("loop")
        goal = FakeState("cheap_goal", goal=True)

        to_loop = Move("cascade", 0, "cascade", 1)
        back_to_root = Move("cascade", 1, "cascade", 0)
        to_goal = Move("cascade", 0, "foundation", 0)

        graph = {
            "root": [to_loop],
            "loop": [back_to_root, to_goal],
        }
        transitions = {
            ("root", to_loop): loop,
            ("loop", back_to_root): root,
            ("loop", to_goal): goal,
        }
        expanded = []

        def fake_moves(state):
            expanded.append(state.name)
            return graph.get(state.name, [])

        def fake_apply(state, move):
            return transitions[(state.name, move)]

        solver = DFSSolver(use_auto_moves=False)
        solver.MAX_DEPTH = 5
        solver.PROGRESS_INTERVAL_S = 0.0
        with mock.patch.object(search_utils, "get_valid_moves", side_effect=fake_moves), \
             mock.patch.object(dfs_module, "apply_move", side_effect=fake_apply), \
             mock.patch.object(solver, "_dls", wraps=solver._dls) as wrapped_dls:
            result = solver.solve(root)

        root_limits = [
            call.kwargs["limit"]
            for call in wrapped_dls.call_args_list
            if call.args[0] is root
        ]
        self.assertTrue(result.solved)
        self.assertEqual(result.solution, [to_loop, to_goal])
        self.assertEqual(root_limits[:2], [1, 2])

    def test_ucs_still_prioritizes_by_path_cost_g(self):
        root = FakeState("root")
        cheap = FakeState("cheap")
        expensive_goal = FakeState("expensive_goal", goal=True)
        cheap_goal = FakeState("cheap_goal", goal=True)

        to_expensive_goal = Move("cascade", 0, "foundation", 0)
        to_cheap = Move("cascade", 0, "cascade", 1)
        cheap_to_goal = Move("cascade", 1, "foundation", 0)

        graph = {
            "root": [to_expensive_goal, to_cheap],
            "cheap": [cheap_to_goal],
        }
        transitions = {
            ("root", to_expensive_goal): expensive_goal,
            ("root", to_cheap): cheap,
            ("cheap", cheap_to_goal): cheap_goal,
        }
        costs = {
            to_expensive_goal: 5,
            to_cheap: 1,
            cheap_to_goal: 1,
        }

        def fake_moves(state):
            return graph.get(state.name, [])

        def fake_apply(state, move):
            return transitions[(state.name, move)]

        solver = UCSSolver(use_auto_moves=False)
        solver.PROGRESS_INTERVAL_S = 0.0
        with mock.patch.object(search_utils, "get_valid_moves", side_effect=fake_moves), \
             mock.patch.object(ucs_module, "apply_move", side_effect=fake_apply), \
             mock.patch.object(ucs_module, "move_cost", side_effect=lambda move: costs[move]):
            result = solver.solve(root)

        self.assertTrue(result.solved)
        self.assertEqual(result.solution, [to_cheap, cheap_to_goal])

    def test_astar_still_prioritizes_by_f_equals_g_plus_h(self):
        root = FakeState("root")
        bad = FakeState("bad")
        good = FakeState("good")
        bad_goal = FakeState("bad_goal", goal=True)
        good_goal = FakeState("good_goal", goal=True)

        to_bad = Move("cascade", 0, "freecell", 0)
        to_good = Move("cascade", 0, "cascade", 1)
        bad_to_goal = Move("freecell", 0, "foundation", 0)
        good_to_goal = Move("cascade", 1, "foundation", 0)

        graph = {
            "root": [to_bad, to_good],
            "bad": [bad_to_goal],
            "good": [good_to_goal],
        }
        transitions = {
            ("root", to_bad): bad,
            ("root", to_good): good,
            ("bad", bad_to_goal): bad_goal,
            ("good", good_to_goal): good_goal,
        }
        heuristic_values = {
            "root": 2,
            "bad": 10,
            "good": 1,
            "bad_goal": 0,
            "good_goal": 0,
        }
        expanded = []

        def fake_moves(state):
            expanded.append(state.name)
            return graph.get(state.name, [])

        def fake_apply(state, move):
            return transitions[(state.name, move)]

        solver = AStarSolver(use_auto_moves=False)
        solver.PROGRESS_INTERVAL_S = 0.0
        with mock.patch.object(search_utils, "get_valid_moves", side_effect=fake_moves), \
             mock.patch.object(astar_module, "apply_move", side_effect=fake_apply), \
             mock.patch.object(astar_module, "heuristic", side_effect=lambda state: heuristic_values[state.name]):
            result = solver.solve(root)

        self.assertTrue(result.solved)
        self.assertEqual(result.solution, [to_good, good_to_goal])
        self.assertEqual(expanded[:2], ["root", "good"])

    def test_immediate_reverse_pruning_keeps_valid_progress_moves(self):
        state = GameState(
            cascades=((Card(2, 1),), (), (), (), (), (), (), ()),
            free_cells=(Card(1, 0), None, None, None),
            foundations=(0, 0, 0, 0),
        )
        previous = Move("cascade", 1, "freecell", 0)
        moves = [
            move
            for move in ordered_legal_moves(state, {})
            if not should_prune_immediate_reverse(move, previous, use_auto_moves=False)
        ]

        self.assertIn(Move("freecell", 0, "foundation", 0), moves)
        self.assertNotIn(Move("freecell", 0, "cascade", 1), moves)

    def test_symmetry_filter_for_empty_destinations_is_safe(self):
        state = GameState(
            cascades=((Card(12, 1),), (), (), (), (), (), (), ()),
            free_cells=(Card(11, 0), None, None, None),
            foundations=(0, 0, 0, 0),
        )

        raw_moves = get_valid_moves(state)
        filtered_moves = ordered_legal_moves(state, {})

        raw_empty_destinations = sorted(
            move.dst_idx
            for move in raw_moves
            if move.src_type == "freecell" and move.dst_type == "cascade" and not state.cascades[move.dst_idx]
        )
        filtered_empty_destinations = sorted(
            move.dst_idx
            for move in filtered_moves
            if move.src_type == "freecell" and move.dst_type == "cascade" and not state.cascades[move.dst_idx]
        )

        self.assertGreater(len(raw_empty_destinations), 1)
        self.assertEqual(filtered_empty_destinations, [1])
        self.assertIn(Move("freecell", 0, "cascade", 0), filtered_moves)

    def test_required_solvers_solve_near_goal(self):
        for solver_cls in (BFSSolver, DFSSolver, UCSSolver, AStarSolver):
            solver = solver_cls(use_auto_moves=False)
            solver.MAX_NODES = 100

            result = solver.solve(near_goal_state())

            self.assertTrue(result.solved, solver.name)
            self.assertEqual(result.solution_length, 1, solver.name)

    def test_required_solver_replay_reconstructs_valid_trace(self):
        for solver_cls in (BFSSolver, DFSSolver, UCSSolver, AStarSolver):
            initial = near_goal_state()
            solver = solver_cls(use_auto_moves=False)
            solver.MAX_NODES = 100

            result = solver.solve(initial)
            final_state = self._play_trace(initial, result.replay_trace)

            self.assertTrue(result.solved, solver.name)
            self.assertEqual(result.replay_trace, result.solution, solver.name)
            self.assertTrue(final_state.is_goal(), solver.name)


if __name__ == "__main__":
    unittest.main()
