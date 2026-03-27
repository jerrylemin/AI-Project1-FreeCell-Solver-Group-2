# FreeCell Solver

## Overview
This repository contains a Python desktop FreeCell game with a tkinter GUI and multiple built-in search solvers. You can play FreeCell manually, load Microsoft numbered deals or random boards, and run the implemented solvers directly from the interface.

The current codebase includes these solver implementations:
- BFS
- DFS via iterative deepening (`DFS (IDS)`)
- UCS
- A*
- Expert Solver

## Features
- Manual FreeCell play in a desktop GUI
- `New Game` dialog for Microsoft numbered deals
- Blank `New Game` input to load a random Microsoft deal
- `RANDOM` input to load a separately shuffled random board
- `Sample` dialog for built-in sample boards (`easy_demo`, `medium_demo`)
- `Restart` to restore the current board to its initial state
- `Undo` for manual moves
- BFS solver button
- DFS button that runs iterative deepening DFS
- UCS solver button
- A* solver button
- Expert Solver button for a stronger practical solver
- `Stop` button for an in-progress solver
- Replay controls: `Play`, `Pause`, `Step`, `Back`
- Replay of the full solution path after success
- Replay of the best-progress trace after stop or failure when available
- Live metrics bar showing algorithm, board source, status, moves, time, memory, expanded nodes, generated nodes, frontier size, and search length
- Optional `Autoplay` checkbox for visible safe auto-foundation steps after a manual move

## Project Structure
```text
AI-Project1-FreeCell-Solver/
+-- main.py                  # GUI entry point
+-- game/
|   +-- card.py              # Card model and rank/suit helpers
|   +-- state.py             # Immutable FreeCell game state
|   +-- moves.py             # Legal move generation, move application, safe auto-moves
|   +-- ms_deals.py          # Exact Microsoft deal generation and random board helpers
|   +-- deal.py              # Deal-generation wrappers
|   `-- samples.py           # Built-in sample boards
+-- gui/
|   +-- app.py               # Main tkinter application and replay controls
|   `-- progress_queue.py    # Queue used for live solver progress updates
+-- solvers/
|   +-- base.py              # Shared solver result/progress infrastructure
|   +-- bfs.py               # Breadth-first search solver
|   +-- dfs.py               # Iterative-deepening DFS solver
|   +-- ucs.py               # Uniform-cost search solver
|   +-- astar.py             # A* solver
|   `-- expert_solver.py     # Practical high-power weighted best-first solver
+-- tests/
|   +-- test_app_flow.py
|   +-- test_expert_solver.py
|   +-- test_layout_and_rules.py
|   +-- test_ms_deals.py
|   +-- test_progress_queue.py
|   +-- test_replay_controls.py
|   +-- test_solver_behavior.py
|   `-- test_solver_replay.py
+-- requirements.txt         # Pip requirements for this repository
`-- README.md
```

## Requirements
- Python 3.10 or newer
- A Python installation that includes `tkinter` / Tk support

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Installation
Create and activate a virtual environment if you want an isolated setup, then install the requirements file.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## How to Run
Start the GUI with:

```bash
python main.py
```

## How to Use the App
1. Launch the program with `python main.py`.
2. Use `New Game` to load a board.
   - Enter a positive integer to load that Microsoft FreeCell deal.
   - Leave the input blank to load a random Microsoft deal.
   - Enter `RANDOM` to load a separately shuffled random board.
3. Use `Sample` and enter `easy_demo` or `medium_demo` to load a built-in sample board.
4. Play manually by clicking a source card or stack, then clicking a valid destination. Right-click clears the current selection.
5. Use `Restart` to reset the current board to its initial layout. Use `Undo` to revert the last manual move.
6. Run a solver with one of the solver buttons: `BFS`, `DFS`, `UCS`, `A*`, or `Expert Solver`.
7. While a solver is running, use `Stop` to request cancellation. The app keeps the best available replay trace when possible.
8. After a run finishes, use the result dialog or the replay controls (`Play`, `Pause`, `Step`, `Back`) to inspect the returned trace.
9. Watch the metrics bar at the top of the window:
   - `Board` shows whether the current board came from a Microsoft deal, a random shuffled board, or a sample board.
   - `Status` shows the live solver state.
   - `Moves` shows the best trace length while searching, then the final solution length or final best trace length after completion.
   - `Time`, `Memory`, `Expanded`, `Generated`, `Frontier`, and `Search length` update as the search progresses.
10. If the `Autoplay` checkbox is enabled, manual moves animate any safe auto-foundation follow-up steps after the move.

## Solver Notes
- `BFS`: breadth-first search.
- `DFS`: the button label is `DFS`, but the solver reports itself as `DFS (IDS)` because it uses iterative deepening.
- `UCS`: uniform-cost search.
- `A*`: heuristic search with replay support.
- `Expert Solver`: a practical weighted best-first solver intended to be stronger than the simple A* on many real deals.

All solver buttons use the same live metrics area, the same stop mechanism, and the same replay system.

## Testing
This repository includes an automated test suite.

Run it with:

```bash
python -m unittest discover -s tests
```

The tests cover solver behavior, replay controls, GUI flow, Microsoft deal generation, layout/rules consistency, and the Expert Solver integration.

## Notes and Limitations
- Hard full deals can take a long time or hit the configured node limit before a solver finishes.
- A stopped or failed solver replay is a best-progress trace, not a guaranteed complete solution.
- Microsoft numbered deals and random shuffled boards are separate board sources and are labeled separately in the GUI.
- The application is a tkinter desktop app, so it requires a Python build with working Tk support.
- Sample boards are built into the source code; no external card data files are required.

## Submission-Oriented Note
This repository includes `README.md` and `requirements.txt` so the source code can be installed and run in the expected project submission format.
