import unittest

from game.card import Card
from game.deal import ms_deal
from game.state import GameState
from solvers.bfs import BFSSolver


class SolverReplayTests(unittest.TestCase):
    def test_failed_bfs_still_returns_replay_trace(self):
        solver = BFSSolver(use_auto_moves=True)
        solver.MAX_NODES = 5

        result = solver.solve(GameState.initial(ms_deal(1)))

        self.assertFalse(result.solved)
        self.assertTrue(result.replay_available)
        self.assertIsNotNone(result.replay_trace)
        self.assertEqual(result.replay_label, "Replay Failed Attempt")
        self.assertGreaterEqual(result.replay_length, 1)

    def test_solved_bfs_replay_matches_solution(self):
        near_goal = GameState(
            cascades=((), (), (), (), (), (), (), ()),
            free_cells=(Card(13, 0), None, None, None),
            foundations=(12, 13, 13, 13),
        )
        solver = BFSSolver(use_auto_moves=True)

        result = solver.solve(near_goal)

        self.assertTrue(result.solved)
        self.assertEqual(result.replay_trace, result.solution)
        self.assertEqual(result.replay_label, "Replay Solution")


if __name__ == "__main__":
    unittest.main()
