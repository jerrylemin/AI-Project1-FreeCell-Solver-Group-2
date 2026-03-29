"""Run the fixed CSC14003 FreeCell experiment suite.

This script performs the full experiment pipeline required by the project brief:
- materialize a fixed testcase suite under experiments/testcases/
- write experiments/test_manifest.json
- run BFS, DFS (implemented as IDS), UCS, and A* on the fixed main suite
- run the same four algorithms on the fixed stress suite
- record actual metrics from the current solver implementations
- write CSV outputs and comparison plots
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

EXPERIMENTS_DIR = ROOT / "experiments"
TESTCASES_DIR = EXPERIMENTS_DIR / "testcases"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"
MAIN_RESULTS_CSV = EXPERIMENTS_DIR / "results_main.csv"
STRESS_RESULTS_CSV = EXPERIMENTS_DIR / "results_stress.csv"
MANIFEST_PATH = EXPERIMENTS_DIR / "test_manifest.json"
ENVIRONMENT_PATH = EXPERIMENTS_DIR / "experiment_environment.json"

MAX_NODES = 150_000
USE_AUTO_MOVES = False

RANKS = "A23456789TJQK"
SUITS = "CDHS"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    category: str
    expected_difficulty: str
    source_type: str
    description: str
    factory: Callable[[], GameState]
    deal_number: Optional[int] = None
    sample_name: Optional[str] = None
    use_in_main_comparison: bool = False
    use_in_stress_comparison: bool = False


SOLVERS = {
    "BFS": BFSSolver,
    "DFS (IDS)": DFSSolver,
    "UCS": UCSSolver,
    "A*": AStarSolver,
}


def near_goal_state() -> GameState:
    return GameState(
        cascades=((), (), (), (), (), (), (), ()),
        free_cells=(Card(13, 0), None, None, None),
        foundations=(12, 13, 13, 13),
    )


CASE_SPECS: List[CaseSpec] = [
    CaseSpec(
        name="near_goal",
        category="main",
        expected_difficulty="trivial",
        source_type="custom_state",
        description="Single-move finish state used to validate basic solver correctness.",
        factory=near_goal_state,
        use_in_main_comparison=True,
    ),
    CaseSpec(
        name="easy_demo",
        category="main",
        expected_difficulty="easy",
        source_type="sample_board",
        description="Built-in sample board easy_demo from the repository.",
        factory=lambda: get_sample_board("easy_demo"),
        sample_name="easy_demo",
        use_in_main_comparison=True,
    ),
    CaseSpec(
        name="medium_demo",
        category="main",
        expected_difficulty="medium",
        source_type="sample_board",
        description="Built-in sample board medium_demo from the repository.",
        factory=lambda: get_sample_board("medium_demo"),
        sample_name="medium_demo",
        use_in_main_comparison=True,
    ),
    CaseSpec(
        name="ms_deal_1",
        category="main",
        expected_difficulty="full_deal",
        source_type="microsoft_deal",
        description="Exact Microsoft numbered deal 1.",
        factory=lambda: GameState.initial(ms_deal(1)),
        deal_number=1,
        use_in_main_comparison=True,
    ),
    CaseSpec(
        name="ms_deal_164",
        category="main",
        expected_difficulty="full_deal",
        source_type="microsoft_deal",
        description="Exact Microsoft numbered deal 164.",
        factory=lambda: GameState.initial(ms_deal(164)),
        deal_number=164,
        use_in_main_comparison=True,
    ),
    CaseSpec(
        name="ms_deal_617",
        category="main",
        expected_difficulty="full_deal",
        source_type="microsoft_deal",
        description="Exact Microsoft numbered deal 617.",
        factory=lambda: GameState.initial(ms_deal(617)),
        deal_number=617,
        use_in_main_comparison=True,
    ),
    CaseSpec(
        name="ms_deal_1941",
        category="stress",
        expected_difficulty="hard_full_deal",
        source_type="microsoft_deal",
        description="Stress testcase based on Microsoft numbered deal 1941.",
        factory=lambda: GameState.initial(ms_deal(1941)),
        deal_number=1941,
        use_in_stress_comparison=True,
    ),
    CaseSpec(
        name="ms_deal_10692",
        category="stress",
        expected_difficulty="hard_full_deal",
        source_type="microsoft_deal",
        description="Stress testcase based on Microsoft numbered deal 10692.",
        factory=lambda: GameState.initial(ms_deal(10692)),
        deal_number=10692,
        use_in_stress_comparison=True,
    ),
    CaseSpec(
        name="ms_deal_11982",
        category="stress",
        expected_difficulty="known_unsolvable",
        source_type="microsoft_deal",
        description="Known unsolvable Microsoft numbered deal 11982.",
        factory=lambda: GameState.initial(ms_deal(11982)),
        deal_number=11982,
        use_in_stress_comparison=True,
    ),
    CaseSpec(
        name="ms_deal_146692",
        category="stress",
        expected_difficulty="known_unsolvable",
        source_type="microsoft_deal",
        description="Known unsolvable Microsoft numbered deal 146692.",
        factory=lambda: GameState.initial(ms_deal(146692)),
        deal_number=146692,
        use_in_stress_comparison=True,
    ),
    CaseSpec(
        name="ms_deal_781948",
        category="stress",
        expected_difficulty="known_unsolvable",
        source_type="microsoft_deal",
        description="Known unsolvable Microsoft numbered deal 781948.",
        factory=lambda: GameState.initial(ms_deal(781948)),
        deal_number=781948,
        use_in_stress_comparison=True,
    ),
]


def card_to_code(card: Card) -> str:
    return f"{RANKS[card.rank - 1]}{SUITS[card.suit]}"


def code_to_card(code: str) -> Card:
    rank = RANKS.index(code[0]) + 1
    suit = SUITS.index(code[1])
    return Card(rank, suit)


def serialize_state(state: GameState) -> Dict[str, object]:
    return {
        "cascades": [[card_to_code(card) for card in column] for column in state.cascades],
        "free_cells": [card_to_code(card) if card is not None else None for card in state.free_cells],
        "foundations": list(state.foundations),
    }


def deserialize_state(payload: Dict[str, object]) -> GameState:
    cascades = tuple(
        tuple(code_to_card(code) for code in column)
        for column in payload["cascades"]
    )
    free_cells = tuple(
        code_to_card(code) if code is not None else None
        for code in payload["free_cells"]
    )
    foundations = tuple(int(value) for value in payload["foundations"])
    return GameState(cascades=cascades, free_cells=free_cells, foundations=foundations)


def ensure_directories() -> None:
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    TESTCASES_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def _board_source_label(spec: CaseSpec) -> str:
    if spec.source_type == "microsoft_deal":
        return f"Microsoft Deal #{spec.deal_number}"
    if spec.source_type == "sample_board":
        return f"Sample Board: {spec.sample_name}"
    return "Custom fixed state"


def materialize_testcases() -> List[Dict[str, object]]:
    manifest_entries: List[Dict[str, object]] = []
    for spec in CASE_SPECS:
        state = spec.factory()
        payload = {
            "name": spec.name,
            "category": spec.category,
            "expected_difficulty": spec.expected_difficulty,
            "source_type": spec.source_type,
            "description": spec.description,
            "deal_number": spec.deal_number,
            "sample_name": spec.sample_name,
            "use_in_main_comparison": spec.use_in_main_comparison,
            "use_in_stress_comparison": spec.use_in_stress_comparison,
            "board_source": _board_source_label(spec),
            "state": serialize_state(state),
        }
        case_path = TESTCASES_DIR / f"{spec.name}.json"
        case_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        manifest_entries.append(
            {
                "name": spec.name,
                "file": f"experiments/testcases/{spec.name}.json",
                "category": spec.category,
                "expected_difficulty": spec.expected_difficulty,
                "source_type": spec.source_type,
                "board_source": payload["board_source"],
                "deal_number": spec.deal_number,
                "sample_name": spec.sample_name,
                "description": spec.description,
                "use_in_main_comparison": spec.use_in_main_comparison,
                "use_in_stress_comparison": spec.use_in_stress_comparison,
            }
        )

    manifest_payload = {
        "node_limit": MAX_NODES,
        "use_auto_moves": USE_AUTO_MOVES,
        "algorithms": list(SOLVERS.keys()),
        "cases": manifest_entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return manifest_entries


def load_case(case_path: Path) -> Dict[str, object]:
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    payload["loaded_state"] = deserialize_state(payload["state"])
    return payload


def write_environment() -> None:
    payload = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "node_limit": MAX_NODES,
        "use_auto_moves": USE_AUTO_MOVES,
        "cwd": str(ROOT),
    }
    ENVIRONMENT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def classify_stop_condition(status: str) -> str:
    lowered = status.lower()
    if "node limit" in lowered:
        return "node_limit"
    if "stopped" in lowered:
        return "stopped"
    if "solved" in lowered:
        return "solved"
    return "failed"


def run_suite(cases: Iterable[Dict[str, object]], suite_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        case_path = ROOT / str(case["file"])
        loaded = load_case(case_path)
        for algorithm, solver_cls in SOLVERS.items():
            state = loaded["loaded_state"]
            solver = solver_cls(use_auto_moves=USE_AUTO_MOVES)
            solver.MAX_NODES = MAX_NODES
            result = solver.solve(state)
            rows.append(
                {
                    "suite": suite_name,
                    "testcase": loaded["name"],
                    "board_source": loaded["board_source"],
                    "source_type": loaded["source_type"],
                    "deal_number": loaded["deal_number"] or "",
                    "sample_name": loaded["sample_name"] or "",
                    "category": loaded["category"],
                    "expected_difficulty": loaded["expected_difficulty"],
                    "algorithm": algorithm,
                    "solved": int(result.solved),
                    "status": result.status,
                    "search_time_s": f"{result.search_time:.6f}",
                    "peak_memory_kb": f"{result.memory_kb:.3f}",
                    "expanded_nodes": result.expanded_nodes,
                    "generated_nodes": result.generated_nodes,
                    "frontier_size": result.frontier_size,
                    "search_length": result.search_length,
                    "solution_length": result.solution_length,
                    "best_trace_length": result.best_trace_length,
                    "replay_length": result.replay_length,
                    "node_limit": MAX_NODES,
                    "use_auto_moves": int(USE_AUTO_MOVES),
                    "stop_condition": classify_stop_condition(result.status),
                }
            )
            print(
                f"{suite_name:6} | {loaded['name']:14} | {algorithm:9} | "
                f"solved={result.solved!s:5} | status={result.status:18} | "
                f"time={result.search_time:7.3f}s | mem={result.memory_kb:9.1f} KB | "
                f"expanded={result.expanded_nodes:6d} | solution={result.solution_length:3d} | "
                f"best={result.best_trace_length:3d}"
            )
    return rows


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_suite(rows: List[Dict[str, object]], suite_name: str) -> None:
    case_order = [
        case.name
        for case in CASE_SPECS
        if (suite_name == "main" and case.use_in_main_comparison)
        or (suite_name == "stress" and case.use_in_stress_comparison)
    ]
    metric_configs = [
        ("search_time_s", "Search Time (s)", f"{suite_name.title()} Suite Search Time", f"{suite_name}_time_comparison.png"),
        ("peak_memory_kb", "Peak Memory (KB)", f"{suite_name.title()} Suite Peak Memory", f"{suite_name}_memory_comparison.png"),
        ("expanded_nodes", "Expanded Nodes", f"{suite_name.title()} Suite Expanded Nodes", f"{suite_name}_expanded_nodes_comparison.png"),
        ("solution_length", "Solution Length", f"{suite_name.title()} Suite Solution Length", f"{suite_name}_solution_length_comparison.png"),
    ]

    by_case_alg = {(row["testcase"], row["algorithm"]): row for row in rows}
    algorithms = list(SOLVERS.keys())
    colors = {
        "BFS": "#355C7D",
        "DFS (IDS)": "#6C5B7B",
        "UCS": "#F08A5D",
        "A*": "#43AA8B",
    }

    for field, ylabel, title, filename in metric_configs:
        figure, axis = plt.subplots(figsize=(13, 5.8))
        positions = list(range(len(case_order)))
        width = 0.18

        for alg_index, algorithm in enumerate(algorithms):
            offset = (alg_index - 1.5) * width
            values: List[float] = []
            solved_flags: List[bool] = []
            for case_name in case_order:
                row = by_case_alg[(case_name, algorithm)]
                solved = bool(int(row["solved"]))
                solved_flags.append(solved)
                if field == "solution_length" and not solved:
                    values.append(0.0)
                else:
                    values.append(float(row[field]))

            bars = axis.bar(
                [position + offset for position in positions],
                values,
                width=width,
                label=algorithm,
                color=colors[algorithm],
                edgecolor="#222222",
                linewidth=0.5,
            )

            for bar, solved, value in zip(bars, solved_flags, values):
                if not solved:
                    bar.set_hatch("//")
                    if field == "solution_length":
                        axis.text(
                            bar.get_x() + bar.get_width() / 2,
                            max(0.15, value + 0.12),
                            "F",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            color="#B00020",
                            fontweight="bold",
                        )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions)
        axis.set_xticklabels(case_order, rotation=20, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.legend(ncols=4, fontsize=9)
        if field == "solution_length":
            axis.set_ylim(bottom=0)
        figure.tight_layout()
        figure.savefig(PLOTS_DIR / filename, dpi=220)
        plt.close(figure)


def main() -> None:
    ensure_directories()
    manifest_entries = materialize_testcases()
    write_environment()

    main_cases = [case for case in manifest_entries if case["use_in_main_comparison"]]
    stress_cases = [case for case in manifest_entries if case["use_in_stress_comparison"]]

    main_rows = run_suite(main_cases, "main")
    stress_rows = run_suite(stress_cases, "stress")

    write_rows(MAIN_RESULTS_CSV, main_rows)
    write_rows(STRESS_RESULTS_CSV, stress_rows)
    plot_suite(main_rows, "main")
    plot_suite(stress_rows, "stress")

    print(f"\nWrote manifest: {MANIFEST_PATH}")
    print(f"Wrote environment: {ENVIRONMENT_PATH}")
    print(f"Wrote main results: {MAIN_RESULTS_CSV}")
    print(f"Wrote stress results: {STRESS_RESULTS_CSV}")
    print(f"Wrote plots under: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
