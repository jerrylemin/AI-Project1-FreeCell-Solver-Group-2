"""Run reproducible experiments for the CSC14003 FreeCell report.

This script benchmarks the required four algorithms on a fixed board set
using the current repository code and writes:
- raw trial results CSV
- per-case summary CSV
- environment metadata JSON
- comparison charts for the report
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.card import Card
from game.deal import ms_deal
from game.samples import get_sample_board
from game.state import GameState
from solvers.astar import AStarSolver
from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.ucs import UCSSolver


OUTPUT_DIR = ROOT / "report_data"
CHART_DIR = ROOT / "report_assets"
RAW_CSV = OUTPUT_DIR / "experiment_results_raw.csv"
SUMMARY_CSV = OUTPUT_DIR / "experiment_results_summary.csv"
ENV_JSON = OUTPUT_DIR / "experiment_environment.json"
TRIALS = 3
MAX_NODES = 5_000


@dataclass(frozen=True)
class BoardSpec:
    name: str
    group: str
    description: str
    factory: Callable[[], GameState]


SOLVERS = {
    "BFS": BFSSolver,
    "DFS (IDS)": DFSSolver,
    "UCS": UCSSolver,
    "A*": AStarSolver,
}


def near_goal_board() -> GameState:
    return GameState(
        cascades=((), (), (), (), (), (), (), ()),
        free_cells=(Card(13, 0), None, None, None),
        foundations=(12, 13, 13, 13),
    )


def two_suit_finish_board() -> GameState:
    return GameState(
        cascades=((Card(13, 0), Card(12, 0)), (Card(13, 1), Card(12, 1)), (), (), (), (), (), ()),
        free_cells=(None, None, None, None),
        foundations=(11, 11, 13, 13),
    )


BOARD_SPECS: List[BoardSpec] = [
    BoardSpec(
        name="near_goal",
        group="core",
        description="Hand-crafted single-move finish state used in solver tests.",
        factory=near_goal_board,
    ),
    BoardSpec(
        name="easy_demo",
        group="core",
        description="Built-in sample board easy_demo.",
        factory=lambda: get_sample_board("easy_demo"),
    ),
    BoardSpec(
        name="two_suit_finish",
        group="core",
        description="Hand-crafted four-move finish with two active suits.",
        factory=two_suit_finish_board,
    ),
    BoardSpec(
        name="medium_demo",
        group="core",
        description="Built-in sample board medium_demo.",
        factory=lambda: get_sample_board("medium_demo"),
    ),
    BoardSpec(
        name="ms_deal_1",
        group="stress",
        description="Full Microsoft deal #1 from the current repository generator.",
        factory=lambda: GameState.initial(ms_deal(1)),
    ),
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)


def run_trials() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for board_spec in BOARD_SPECS:
        for trial in range(1, TRIALS + 1):
            for solver_name, solver_cls in SOLVERS.items():
                state = board_spec.factory()
                solver = solver_cls(use_auto_moves=False)
                solver.MAX_NODES = MAX_NODES
                result = solver.solve(state)
                rows.append(
                    {
                        "board": board_spec.name,
                        "board_group": board_spec.group,
                        "board_description": board_spec.description,
                        "trial": trial,
                        "algorithm": solver_name,
                        "solved": int(result.solved),
                        "status": result.status,
                        "search_time_s": round(result.search_time, 6),
                        "peak_memory_kb": round(result.memory_kb, 3),
                        "expanded_nodes": result.expanded_nodes,
                        "generated_nodes": result.generated_nodes,
                        "frontier_size": result.frontier_size,
                        "search_length": result.search_length,
                        "solution_length": result.solution_length,
                        "best_trace_length": result.best_trace_length,
                        "replay_length": result.replay_length,
                        "node_limit": MAX_NODES,
                    }
                )
                print(
                    f"{board_spec.name:16} | trial {trial} | {solver_name:9} | "
                    f"solved={result.solved!s:5} | time={result.search_time:7.3f}s | "
                    f"mem={result.memory_kb:9.1f} KB | expanded={result.expanded_nodes:6d} | "
                    f"solution={result.solution_length:3d}"
                )
    return rows


def write_raw_csv(rows: List[Dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with RAW_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((row["board"], row["algorithm"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (board, algorithm), group_rows in grouped.items():
        first = group_rows[0]

        def med(field: str) -> float:
            values = [float(r[field]) for r in group_rows]
            return statistics.median(values)

        solved_runs = sum(int(r["solved"]) for r in group_rows)
        statuses = [str(r["status"]) for r in group_rows]
        common_status = max(set(statuses), key=statuses.count)
        summary_rows.append(
            {
                "board": board,
                "board_group": first["board_group"],
                "board_description": first["board_description"],
                "algorithm": algorithm,
                "trials": len(group_rows),
                "solved_runs": solved_runs,
                "solve_rate": round(solved_runs / len(group_rows), 3),
                "status_mode": common_status,
                "median_search_time_s": round(med("search_time_s"), 6),
                "median_peak_memory_kb": round(med("peak_memory_kb"), 3),
                "median_expanded_nodes": int(round(med("expanded_nodes"))),
                "median_generated_nodes": int(round(med("generated_nodes"))),
                "median_frontier_size": int(round(med("frontier_size"))),
                "median_search_length": int(round(med("search_length"))),
                "median_solution_length": int(round(med("solution_length"))),
                "median_best_trace_length": int(round(med("best_trace_length"))),
                "node_limit": first["node_limit"],
            }
        )

    summary_rows.sort(key=lambda row: (row["board_group"], row["board"], row["algorithm"]))
    return summary_rows


def write_summary_csv(summary_rows: List[Dict[str, object]]) -> None:
    fieldnames = list(summary_rows[0].keys())
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_environment_metadata() -> None:
    payload = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "max_nodes": MAX_NODES,
        "trials": TRIALS,
        "solver_setting_use_auto_moves": False,
    }
    ENV_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _chart_rows(summary_rows: List[Dict[str, object]], field: str, title: str, ylabel: str, filename: str) -> None:
    core_rows = [row for row in summary_rows if row["board_group"] == "core"]
    boards = [spec.name for spec in BOARD_SPECS if spec.group == "core"]
    algorithms = list(SOLVERS.keys())

    values = np.array(
        [
            [
                float(
                    next(
                        row[field]
                        for row in core_rows
                        if row["board"] == board and row["algorithm"] == algorithm
                    )
                )
                for algorithm in algorithms
            ]
            for board in boards
        ]
    )

    x = np.arange(len(boards))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    colors = ["#355C7D", "#6C5B7B", "#C06C84", "#F67280"]

    for idx, algorithm in enumerate(algorithms):
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, values[:, idx], width=width, label=algorithm, color=colors[idx])
        if field == "median_solution_length":
            for bar, board in zip(bars, boards):
                row = next(
                    item
                    for item in core_rows
                    if item["board"] == board and item["algorithm"] == algorithm
                )
                if float(row["solve_rate"]) < 1.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        max(bar.get_height(), 0.2) + 0.1,
                        "F",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="#222222",
                        fontweight="bold",
                    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(boards)
    ax.legend(ncols=4, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=200)
    plt.close(fig)


def build_charts(summary_rows: List[Dict[str, object]]) -> None:
    _chart_rows(
        summary_rows,
        "median_search_time_s",
        "Median Search Time on Core Benchmark Set",
        "Seconds",
        "chart_search_time.png",
    )
    _chart_rows(
        summary_rows,
        "median_peak_memory_kb",
        "Median Peak Memory on Core Benchmark Set",
        "Peak memory (KB)",
        "chart_peak_memory.png",
    )
    _chart_rows(
        summary_rows,
        "median_expanded_nodes",
        "Median Expanded Nodes on Core Benchmark Set",
        "Expanded nodes",
        "chart_expanded_nodes.png",
    )
    _chart_rows(
        summary_rows,
        "median_solution_length",
        "Median Solution Length on Core Benchmark Set",
        "Moves (F = failed to solve)",
        "chart_solution_length.png",
    )


def main() -> None:
    ensure_dirs()
    rows = run_trials()
    write_raw_csv(rows)
    summary_rows = summarize(rows)
    write_summary_csv(summary_rows)
    write_environment_metadata()
    build_charts(summary_rows)
    print(f"\nWrote raw results to {RAW_CSV}")
    print(f"Wrote summary results to {SUMMARY_CSV}")
    print(f"Wrote environment metadata to {ENV_JSON}")
    print(f"Wrote charts to {CHART_DIR}")


if __name__ == "__main__":
    main()
