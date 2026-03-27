"""Build Report.pdf from the generated experiment artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT_PDF = ROOT / "Report.pdf"
SUMMARY_CSV = ROOT / "report_data" / "experiment_results_summary.csv"
ENV_JSON = ROOT / "report_data" / "experiment_environment.json"
CHART_DIR = ROOT / "report_assets"


def load_summary() -> List[Dict[str, str]]:
    with SUMMARY_CSV.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_env() -> Dict[str, str]:
    return json.loads(ENV_JSON.read_text(encoding="utf-8"))


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(doc.leftMargin, 0.45 * inch, "CSC14003 Project 1 - FreeCell Solver Report")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def make_table(data: List[List[object]], col_widths: List[float], header_bg="#DCE6F1", font_size=8) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#777777")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FBFD")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_bullets(story, styles, items: Iterable[str]) -> None:
    bullet_style = styles["Body"]
    for item in items:
        story.append(Paragraph(f"- {item}", bullet_style))
    story.append(Spacer(1, 6))


def main() -> None:
    summary_rows = load_summary()
    env = load_env()
    core_rows = [row for row in summary_rows if row["board_group"] == "core"]
    stress_rows = [row for row in summary_rows if row["board_group"] == "stress"]

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitlePage",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=18,
            textColor=colors.HexColor("#17365D"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor("#17365D"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=8,
            spaceAfter=5,
            textColor=colors.HexColor("#1F497D"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            spaceAfter=4,
        )
    )

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.7 * inch,
        title="CSC14003 Project 1 Report",
        author="OpenAI Codex",
    )

    story: List[object] = []

    story.append(Spacer(1, 1.1 * inch))
    story.append(Paragraph("CSC14003 Project 1 Report", styles["TitlePage"]))
    story.append(Paragraph("FreeCell Solver", styles["TitlePage"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Course: CSC14003 - Introduction to Artificial Intelligence", styles["Body"]))
    story.append(Paragraph("Deliverable: Section 3.6 Report", styles["Body"]))
    story.append(Paragraph("Date: 2026-03-27", styles["Body"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Team members", styles["SubSection"]))
    story.append(
        make_table(
            [
                ["Member", "Student ID", "Role"],
                ["TODO Member 1", "TODO ID 1", "TODO"],
                ["TODO Member 2", "TODO ID 2", "TODO"],
                ["TODO Member 3", "TODO ID 3", "TODO"],
            ],
            [2.1 * inch, 1.4 * inch, 2.3 * inch],
            font_size=9,
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Note: Team member names, student IDs, task distribution, and completion percentages must be filled in by the team before submission.",
            styles["Small"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("1. Introduction", styles["Section"]))
    story.append(
        Paragraph(
            "This project implements a Python desktop FreeCell application with a tkinter GUI, a shared rules engine, and multiple search solvers. The current repository supports manual play, Microsoft numbered deals, random shuffled boards, built-in sample boards, background solver execution, live search metrics, and replay of either the final solution or the best-progress trace.",
            styles["Body"],
        )
    )
    add_bullets(story, styles, ["Required algorithms present: BFS, DFS implemented as IDS, UCS, and A*.", "An additional Expert Solver exists in the repository but is not used as a substitute for the required algorithms in the main evaluation."])

    story.append(Paragraph("2. Project Planning and Task Distribution", styles["Section"]))
    story.append(
        Paragraph(
            "Section 3.6 of the project brief requires each team member's responsibilities and completion percentage. The repository did not contain the actual team allocation data, so the table below is an explicit placeholder for the team to complete before submission.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Member", "Student ID", "Assigned tasks", "Completion %", "Notes"],
                ["TODO Member 1", "TODO ID 1", "TODO", "TODO", "TODO"],
                ["TODO Member 2", "TODO ID 2", "TODO", "TODO", "TODO"],
                ["TODO Member 3", "TODO ID 3", "TODO", "TODO", "TODO"],
            ],
            [1.3 * inch, 1.1 * inch, 2.1 * inch, 0.95 * inch, 1.15 * inch],
            font_size=8,
        )
    )
    story.append(
        Paragraph(
            "Brief example from page 5: if the group score is 9.0 and a student's completion percentage is 90%, the individual score is 9.0 x 90% = 8.1.",
            styles["Small"],
        )
    )

    story.append(Paragraph("3. System Overview", styles["Section"]))
    story.append(
        Paragraph(
            "The entry point is main.py, which launches gui.app.FreeCellApp. The codebase is organized into GUI, game-logic, solver, and test modules. The GUI handles drawing, user actions, solver controls, and replay. The shared game layer defines cards, immutable states, move generation, move application, Microsoft deals, and sample boards. The solver layer contains a shared BaseSolver plus BFS, DFS (IDS), UCS, A*, and an extra Expert Solver. Replay and live metrics flow through SolverResult, ProgressSnapshot, and the FreshQueue progress queue.",
            styles["Body"],
        )
    )
    add_bullets(
        story,
        styles,
        [
            "GUI modules: gui/app.py, gui/progress_queue.py",
            "Game logic: game/card.py, game/state.py, game/moves.py, game/deal.py, game/ms_deals.py, game/samples.py",
            "Required solvers: solvers/bfs.py, solvers/dfs.py, solvers/ucs.py, solvers/astar.py",
            "Metrics and replay infrastructure: solvers/base.py and gui/app.py",
            "Tests: tests/*.py",
        ],
    )

    story.append(Paragraph("4. Search Problem Formulation", styles["Section"]))
    story.append(
        Paragraph(
            "The search state is game.state.GameState with eight cascades, four free cells, and four foundation counters. Legal actions are generated by game.moves.get_valid_moves() and include moves between cascades, free cells, and foundations, including sequence moves. apply_move() creates a new immutable GameState for every transition. The goal test is GameState.is_goal(), which checks that every foundation has reached rank 13. Repeated-state handling uses GameState.canonical_key(), which keeps the cascades exact, sorts free cells as a multiset, and includes the foundation tuple.",
            styles["Body"],
        )
    )
    add_bullets(
        story,
        styles,
        [
            "BFS and IDS use implicit unit depth as their search depth measure.",
            "UCS uses a custom move cost: foundation = 0, free cell = 3, cascade = 1.",
            "A* uses unit move cost g plus a heuristic h.",
            "BFS suppresses duplicates through the parent dictionary; UCS and A* use best-cost tables; IDS uses path-based cycle checking only.",
        ],
    )

    story.append(Paragraph("5. Algorithm Description", styles["Section"]))
    for title, text in [
        (
            "5.1 BFS",
            "solvers/bfs.py implements BFS with a collections.deque frontier. States are popped in FIFO order using popleft(). Repeated states are avoided through the parent dictionary keyed by canonical_key(). The solver checks the goal before the loop and immediately after generating each child. Paths are reconstructed by BaseSolver._reconstruct(). BFS is complete over the explored state space but uses the most memory on difficult boards because it stores a large frontier.",
        ),
        (
            "5.2 DFS or IDS",
            "The repository's DFS requirement is implemented as iterative deepening DFS in solvers/dfs.py, and the solver name is explicitly 'DFS (IDS)'. The search increases the depth limit from 1 upward and runs a recursive depth-limited search. It avoids cycles only along the current path using path_keys, so work is repeated across depth iterations. This keeps memory low but causes significantly higher runtime on harder boards.",
        ),
        (
            "5.3 UCS",
            "solvers/ucs.py implements uniform-cost search with a heapq priority queue storing (g, counter, state). The actual path-cost function in the code assigns cost 0 to foundation moves, 3 to free-cell moves, and 1 to cascade moves. Repeated states are handled with a best_g table. UCS therefore optimizes the repository's custom move cost, not strictly the fewest number of moves.",
        ),
        (
            "5.4 A*",
            "solvers/astar.py implements A* with a heapq frontier over f = g + h. The heuristic combines three terms: cards not yet on foundations, blocker depth for the next needed foundation cards, and a 0.5 penalty for occupied free cells. A direct repository counterexample shows the heuristic can exceed the true remaining cost, so the implementation should be described as heuristic best-first search with f = g + h but without a guaranteed optimality proof.",
        ),
    ]:
        story.append(Paragraph(title, styles["SubSection"]))
        story.append(Paragraph(text, styles["Body"]))

    story.append(Paragraph("6. Experiments", styles["Section"]))
    story.append(
        Paragraph(
            "The benchmark runner is report_tools/run_report_experiments.py. Every required algorithm was executed on the same fixed board set with the same settings: max nodes = 5,000, use_auto_moves = False, and 3 trials per algorithm-board pair. Medians are reported. The performance metrics are taken directly from BaseSolver.solve(), which measures elapsed search time and peak memory via tracemalloc and returns expanded nodes and solution length through SolverResult.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Item", "Value"],
                ["Python version", env["python_version"]],
                ["Python implementation", env["python_implementation"]],
                ["Platform", env["platform"]],
                ["Processor string", env["processor"]],
                ["Machine", env["machine"]],
                ["Trials per case", str(env["trials"])],
                ["Equal node cap", str(env["max_nodes"])],
                ["use_auto_moves", str(env["solver_setting_use_auto_moves"])],
            ],
            [2.2 * inch, 4.4 * inch],
            font_size=9,
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        make_table(
            [
                ["Benchmark set", "Board", "Purpose"],
                ["Core", "near_goal", "Single-move finish state used in solver tests"],
                ["Core", "easy_demo", "Built-in sample board from the repository"],
                ["Core", "two_suit_finish", "Hand-crafted 4-move finish state"],
                ["Core", "medium_demo", "Harder built-in sample board"],
                ["Stress", "ms_deal_1", "Full Microsoft deal #1 under the same node cap"],
            ],
            [0.85 * inch, 1.45 * inch, 4.3 * inch],
            font_size=8,
        )
    )

    story.append(Paragraph("7. Experimental Results", styles["Section"]))
    core_table = [["Board", "Algorithm", "Solved", "Time (s)", "Peak KB", "Expanded", "Solution", "Status"]]
    for row in sorted(core_rows, key=lambda item: (item["board"], item["algorithm"])):
        core_table.append(
            [
                row["board"],
                row["algorithm"],
                "Yes" if float(row["solve_rate"]) == 1.0 else "No",
                row["median_search_time_s"],
                row["median_peak_memory_kb"],
                row["median_expanded_nodes"],
                row["median_solution_length"],
                row["status_mode"],
            ]
        )
    story.append(make_table(core_table, [0.95 * inch, 0.9 * inch, 0.55 * inch, 0.65 * inch, 0.8 * inch, 0.72 * inch, 0.65 * inch, 1.15 * inch], font_size=7.5))
    story.append(Spacer(1, 0.12 * inch))

    stress_table = [["Board", "Algorithm", "Solved", "Time (s)", "Peak KB", "Expanded", "Solution", "Status"]]
    for row in sorted(stress_rows, key=lambda item: item["algorithm"]):
        stress_table.append(
            [
                row["board"],
                row["algorithm"],
                "Yes" if float(row["solve_rate"]) == 1.0 else "No",
                row["median_search_time_s"],
                row["median_peak_memory_kb"],
                row["median_expanded_nodes"],
                row["median_solution_length"],
                row["status_mode"],
            ]
        )
    story.append(make_table(stress_table, [0.95 * inch, 0.9 * inch, 0.55 * inch, 0.65 * inch, 0.8 * inch, 0.72 * inch, 0.65 * inch, 1.15 * inch], header_bg="#EAD1DC", font_size=7.5))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "Key observation: BFS and IDS solve the small finish states but both exhaust the equal 5,000-node budget on medium_demo. UCS and A* solve all core boards, but all four required algorithms fail on the full Microsoft stress case under the same budget.",
            styles["Body"],
        )
    )

    for index, (chart_title, filename) in enumerate([
        ("Search time comparison", "chart_search_time.png"),
        ("Peak memory comparison", "chart_peak_memory.png"),
        ("Expanded nodes comparison", "chart_expanded_nodes.png"),
        ("Solution length comparison", "chart_solution_length.png"),
    ]):
        if index > 0:
            story.append(PageBreak())
        story.append(Paragraph(chart_title, styles["Section"]))
        story.append(Image(str(CHART_DIR / filename), width=7.0 * inch, height=4.1 * inch))
        story.append(Spacer(1, 0.12 * inch))
        if filename == "chart_solution_length.png":
            story.append(Paragraph("In the solution-length chart, F marks algorithms that failed to solve the board within the equal node cap.", styles["Small"]))

    story.append(Paragraph("8. Discussion", styles["Section"]))
    story.append(
        Paragraph(
            "Among the four required algorithms, UCS and A* are the most practical in the current repository. They solve every core benchmark board under the equal 5,000-node budget, while BFS and IDS both fail on medium_demo. BFS uses the most memory because it maintains a large frontier. IDS keeps memory very low but pays with repeated work and the slowest runtime on the harder sample board. UCS and A* are close on the tested core set; both solve medium_demo in about 0.27 seconds and expand 256 nodes. A* should still be described carefully because the current heuristic is not formally admissible.",
            styles["Body"],
        )
    )
    add_bullets(
        story,
        styles,
        [
            "BFS: simple and complete, but memory-heavy.",
            "DFS (IDS): memory-light, but expensive in runtime because of repeated iterations.",
            "UCS: strong under the repository's custom move cost.",
            "A*: strong practical performance on the tested core set, but without a valid optimality guarantee under the current heuristic.",
            "Expert Solver exists as an extra implementation note, but it is outside the required four-algorithm comparison.",
        ],
    )

    story.append(Paragraph("9. Conclusion", styles["Section"]))
    story.append(
        Paragraph(
            "The FreeCell Solver repository satisfies the main structural requirement of implementing handwritten search algorithms inside a playable Python GUI. The shared rules engine, immutable state model, and solver metrics make the codebase suitable for reproducible evaluation. Under the equal 5,000-node budget used in this report, BFS and IDS are limited to small finish states, while UCS and A* also solve the harder medium_demo sample board. In the current implementation, UCS and A* are the most practical among the four required algorithms.",
            styles["Body"],
        )
    )

    story.append(Paragraph("10. References", styles["Section"]))
    refs = [
        "1. CSC14003 - Project 1.pdf, Section 3.6 'Report', page 5. Local course brief PDF found at C:\\MEGA\\co so ttnt\\CSC14003 - Project 1.pdf.",
        "2. Repository source code used as primary implementation evidence: main.py, gui/app.py, gui/progress_queue.py, game/state.py, game/moves.py, game/deal.py, game/ms_deals.py, solvers/base.py, solvers/bfs.py, solvers/dfs.py, solvers/ucs.py, solvers/astar.py.",
        "3. Repository test modules used to validate current behavior: tests/test_app_flow.py, tests/test_solver_behavior.py, tests/test_solver_replay.py, tests/test_replay_controls.py, tests/test_ms_deals.py, tests/test_layout_and_rules.py.",
    ]
    add_bullets(story, styles, refs)

    story.append(Paragraph("11. Appendix", styles["Section"]))
    story.append(Paragraph("11.1 Generated experiment artifacts", styles["SubSection"]))
    add_bullets(
        story,
        styles,
        [
            "report_data/experiment_results_raw.csv",
            "report_data/experiment_results_summary.csv",
            "report_data/experiment_environment.json",
            "report_assets/chart_search_time.png",
            "report_assets/chart_peak_memory.png",
            "report_assets/chart_expanded_nodes.png",
            "report_assets/chart_solution_length.png",
        ],
    )
    story.append(Paragraph("11.2 Commands used", styles["SubSection"]))
    add_bullets(
        story,
        styles,
        [
            "python report_tools/run_report_experiments.py",
            "python -m unittest discover -s tests",
            "python main.py",
        ],
    )
    story.append(Paragraph("11.3 Generative AI usage", styles["SubSection"]))
    story.append(
        Paragraph(
            "Generative AI was used in this report-generation session to audit the repository, extract implementation details, run reproducible experiments, summarize the results, generate charts, draft the report text, and build the final PDF. All numeric values in the report come from actual CSV outputs generated from the current repository, and the final text was checked manually against those artifacts.",
            styles["Body"],
        )
    )
    story.append(Paragraph("11.4 Prompt record", styles["SubSection"]))
    prompt_table = [
        ["Prompt / instruction source", "What it was used for"],
        ["Expert Solver patch request", "Added the Expert Solver and related tests earlier in this session."],
        ["README / requirements audit request", "Produced up-to-date run instructions and dependency notes earlier in this session."],
        ["Section 3.6 report generation request", "Used for this report-generation run."],
        ["TODO: add other team-used prompts", "Fill in before submission if applicable."],
    ]
    story.append(make_table(prompt_table, [2.8 * inch, 3.4 * inch], header_bg="#FCE5CD", font_size=8))

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
