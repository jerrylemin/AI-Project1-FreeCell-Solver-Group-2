"""Small benchmark runner for ExpertSolver on Microsoft deals."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.deal import ms_deal
from game.state import GameState
from solvers.expert_solver import ExpertSolver


OUTPUT_CSV = ROOT / "report_data" / "expert_solver_benchmark.csv"


def run() -> None:
    rng = random.Random(20260327)
    deals = []
    while len(deals) < 10:
        deal = rng.randint(1, 500)
        if deal not in deals:
            deals.append(deal)

    rows = []
    solved_any = False
    for deal in deals:
        solver = ExpertSolver(use_auto_moves=False)
        solver.MAX_NODES = 15_000
        result = solver.solve(GameState.initial(ms_deal(deal)))
        rows.append(
            {
                "deal": deal,
                "solved": int(result.solved),
                "status": result.status,
                "search_time_s": round(result.search_time, 6),
                "expanded_nodes": result.expanded_nodes,
                "generated_nodes": result.generated_nodes,
                "solution_length": result.solution_length,
                "replay_length": result.replay_length,
                "best_trace_length": result.best_trace_length,
            }
        )
        print(
            f"deal={deal:3d} solved={result.solved!s:5} "
            f"time={result.search_time:7.3f}s expanded={result.expanded_nodes:6d} "
            f"solution={result.solution_length:3d} replay={result.replay_length:3d}"
        )
        if result.solved:
            solved_any = True

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote benchmark results to {OUTPUT_CSV}")
    print(f"Solved at least one deal: {solved_any}")


if __name__ == "__main__":
    run()
