import unittest

from game.card import Card
from game.deal import ms_deal
from game.moves import apply_auto_moves
from game.state import GameState
from solvers.astar import AStarSolver
from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.ucs import UCSSolver


SOLVER_CLASSES = (BFSSolver, DFSSolver, UCSSolver, AStarSolver)


class SolverBehaviorTests(unittest.TestCase):
    def test_solvers_reset_cleanly_between_runs_on_near_goal(self):
        near_goal = GameState(
            cascades=((), (), (), (), (), (), (), ()),
            free_cells=(Card(13, 0), None, None, None),
            foundations=(12, 13, 13, 13),
        )

        for solver_cls in SOLVER_CLASSES:
            solver = solver_cls(use_auto_moves=False)
            solver.MAX_NODES = 100

            first = solver.solve(near_goal)
            second = solver.solve(near_goal)

            self.assertTrue(first.solved, solver.name)
            self.assertTrue(second.solved, solver.name)
            self.assertEqual(first.solution_length, 1, solver.name)
            self.assertEqual(second.solution_length, 1, solver.name)

    def test_best_trace_length_is_monotonic_during_search(self):
        state = apply_auto_moves(GameState.initial(ms_deal(1)))

        for solver_cls in SOLVER_CLASSES:
            moves = []
            solver = solver_cls(
                use_auto_moves=True,
                progress_callback=lambda snapshot, seen=moves: seen.append(snapshot.moves),
            )
            solver.MAX_NODES = 200
            solver.PROGRESS_INTERVAL_S = 0.0

            solver.solve(state)

            self.assertGreaterEqual(len(moves), 1, solver.name)
            self.assertTrue(
                all(left <= right for left, right in zip(moves, moves[1:])),
                solver.name,
            )


if __name__ == "__main__":
    unittest.main()
