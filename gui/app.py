"""FreeCell GUI - tkinter Canvas-based implementation."""

from __future__ import annotations

import os
import random
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Callable, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game.card import Card, SUIT_SYMBOLS
from game.deal import ms_deal, random_deal
from game.moves import (
    Move,
    apply_move,
    apply_move_with_auto_steps,
    get_valid_moves,
)
from game.samples import SAMPLE_BOARD_NAMES, get_sample_board
from game.state import GameState
from gui.progress_queue import FreshQueue
from solvers.base import SolverProgress, SolverResult


CW, CH = 80, 110
COL_W = CW + 10
OVERLAP = 28
DEFAULT_CASCADE_SPACING = 30
MIN_VISIBLE_CASCADE_STRIP = 18
BOARD_BOTTOM_MARGIN = 8

CANVAS_W = 960
CANVAS_H = 720

FC_Y = 20
FD_Y = 20
CAS_Y = 168

FC_X = [10 + i * COL_W for i in range(4)]
FD_X = [CANVAS_W - 4 * COL_W - 10 + i * COL_W for i in range(4)]
CAS_X = [(CANVAS_W - 8 * COL_W) // 2 + i * COL_W for i in range(8)]

BG_FELT = "#1a6b3a"
BG_EMPTY = "#155a30"
CLR_WHITE = "#fafafa"
CLR_RED = "#cc1111"
CLR_BLACK = "#111122"
CLR_SEL = "#ffe44d"
CLR_HINT = "#44ddaa"
CLR_LABEL = "#aaddbb"
CLR_SLOT = "#336633"

FONT_RANK_SM = ("Helvetica", 11, "bold")
FONT_CORNER = ("Helvetica", 10, "bold")
FONT_SUIT_LG = ("Helvetica", 26)
FONT_BTN = ("Helvetica", 10, "bold")
FONT_STATUS = ("Helvetica", 10)
FONT_LABEL = ("Helvetica", 9)

ANIMATION_FRAME_MS = 16
SOLVER_POLL_MS = 50
MOVE_DURATION_MS = 220
BETWEEN_MOVES_MS = 40
MIN_SPEED = 0.5
MAX_SPEED = 3.0


def compute_cascade_spacing(
    column_height: int,
    card_count: int,
    card_height: int,
    min_overlap_visible: int,
    normal_spacing: int = DEFAULT_CASCADE_SPACING,
) -> int:
    """Pick a readable cascade spacing that fits while preserving card identity."""
    if card_count <= 1:
        return 0

    usable = max(column_height - card_height, min_overlap_visible)
    fitted = usable // (card_count - 1)
    return max(min_overlap_visible, min(normal_spacing, fitted))


class Selection:
    """Tracks the currently selected source stack."""

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.src_type: Optional[str] = None
        self.src_idx = -1
        self.num_cards = 0

    def is_active(self) -> bool:
        return self.src_type is not None

    def set(self, src_type: str, src_idx: int, num_cards: int = 1) -> None:
        self.src_type = src_type
        self.src_idx = src_idx
        self.num_cards = num_cards


class ResultsDialog(tk.Toplevel):
    def __init__(self, parent, result: SolverResult, on_play: Callable[[SolverResult], None]):
        super().__init__(parent)
        self.title(f"Solver Result - {result.algorithm}")
        self.resizable(False, False)
        self.grab_set()

        text = tk.Text(
            self,
            font=("Courier", 11),
            width=48,
            height=12,
            bg="#1e2030",
            fg="#c8d8e8",
            relief="flat",
            padx=12,
            pady=8,
        )
        text.insert("1.0", result.summary())
        text.config(state="disabled")
        text.pack(padx=16, pady=(16, 8))

        btn_frame = tk.Frame(self, bg="#2a2a3a")
        btn_frame.pack(fill="x", padx=16, pady=(0, 12))

        if result.replay_available:
            tk.Button(
                btn_frame,
                text=result.replay_label,
                font=FONT_BTN,
                bg="#2d8c54",
                fg="white",
                activebackground="#3aaa66",
                relief="flat",
                padx=10,
                pady=5,
                command=lambda: (self.destroy(), on_play(result)),
            ).pack(side="left", padx=(0, 8))

        tk.Button(
            btn_frame,
            text="Close",
            font=FONT_BTN,
            bg="#555577",
            fg="white",
            relief="flat",
            padx=10,
            pady=5,
            command=self.destroy,
        ).pack(side="left")


class FreeCellApp(tk.Tk):
    """Main FreeCell application window."""

    def __init__(self):
        super().__init__()
        self.title("FreeCell - AI Solver")
        self.resizable(False, False)
        self.configure(bg="#1e2030")

        self._deal_number: Optional[int] = None
        self._initial_cascades: Optional[List[List[Card]]] = None
        self._initial_state: Optional[GameState] = None
        self._state: Optional[GameState] = None
        self._preview_state: Optional[GameState] = None
        self._history: List[GameState] = []
        self._selection = Selection()

        self._solving = False
        self._active_solver = None
        self._solver_thread: Optional[threading.Thread] = None
        self._solver_progress_queue: FreshQueue[SolverProgress] = FreshQueue(maxsize=24)
        self._solver_progress_job: Optional[str] = None

        self._replay_moves: List[Move] = []
        self._replay_states: List[GameState] = []
        self._replay_cursor = 0
        self._playing_solution = False
        self._pending_play_job: Optional[str] = None
        self._pending_animation_job: Optional[str] = None
        self._active_animation: Optional[dict] = None
        self._sequence_state: Optional[GameState] = None
        self._sequence_queue: List[Tuple[Move, str]] = []
        self._sequence_done: Optional[Callable[[], None]] = None
        self._sequence_commit_state = True

        self._speed_var = tk.DoubleVar(value=1.35)
        self._autoplay_var = tk.BooleanVar(value=False)
        self._max_nodes_var = tk.StringVar(value="150000")
        self._board_source_var = tk.StringVar(value="Microsoft Deal #1")
        self._live_solver_name = tk.StringVar(value="Idle")
        self._live_solver_status = tk.StringVar(value="Idle")
        self._live_moves_var = tk.StringVar(value="0")
        self._live_time_var = tk.StringVar(value="0.00 s")
        self._live_memory_var = tk.StringVar(value="0 KB")
        self._live_nodes_var = tk.StringVar(value="0")
        self._live_generated_var = tk.StringVar(value="0")
        self._live_frontier_var = tk.StringVar(value="0")
        self._live_search_var = tk.StringVar(value="0")
        self._status_var = tk.StringVar(value="Microsoft Deal #1 - Click a card to select it.")
        self._replay_label = "Replay Solution"
        self._replay_message = ""
        self._replay_is_solution = True
        self._replay_kind = ""

        self._card_items: Dict[int, Dict[str, object]] = {}
        self._visible_card_ids: Set[int] = set()
        self._freecell_slots: List[Dict[str, int]] = []
        self._foundation_slots: List[Dict[str, int]] = []
        self._cascade_slots: List[Dict[str, int]] = []
        self._foundation_hint_labels: List[int] = []
        self._cascade_hint_rects: List[int] = []
        self._win_overlay_items: List[int] = []

        self._build_ui()
        self._new_game_number(1)

    def _build_ui(self) -> None:
        ctrl = tk.Frame(self, bg="#1e2030", pady=6)
        ctrl.pack(fill="x", padx=8)

        def btn(parent, text, cmd, color="#3a5f8a", **kw):
            widget = tk.Button(
                parent,
                text=text,
                command=cmd,
                font=FONT_BTN,
                bg=color,
                fg="white",
                activebackground="#6688bb",
                relief="flat",
                padx=9,
                pady=4,
                **kw,
            )
            widget.pack(side="left", padx=3)
            return widget

        self._btn_new_game = btn(ctrl, "New Game", self._new_game_dialog)
        self._btn_sample = btn(ctrl, "Sample", self._sample_board_dialog)
        self._btn_restart = btn(ctrl, "Restart", self._restart)
        self._btn_undo = btn(ctrl, "Undo", self._undo)

        tk.Frame(ctrl, bg="#1e2030", width=30).pack(side="left")

        self._btn_bfs = btn(ctrl, "BFS", lambda: self._run_solver("BFS"), "#5a3a8a")
        self._btn_dfs = btn(ctrl, "DFS", lambda: self._run_solver("DFS"), "#5a3a8a")
        self._btn_ucs = btn(ctrl, "UCS", lambda: self._run_solver("UCS"), "#5a3a8a")
        self._btn_astar = btn(ctrl, "A*", lambda: self._run_solver("ASTAR"), "#5a3a8a")
        self._btn_expert = btn(
            ctrl,
            "Expert Solver",
            lambda: self._run_solver("EXPERT"),
            "#7a4a1f",
        )

        tk.Frame(ctrl, bg="#1e2030", width=30).pack(side="left")

        self._btn_play = btn(ctrl, "Play", self._play_replay, "#2d7a4a")
        self._btn_play.config(state="disabled")
        self._btn_pause = btn(ctrl, "Pause", self._pause_replay, "#2d7a4a")
        self._btn_pause.config(state="disabled")
        self._btn_step = btn(ctrl, "Step", self._step_replay, "#2d7a4a")
        self._btn_step.config(state="disabled")
        self._btn_back = btn(ctrl, "Back", self._back_replay, "#2d7a4a")
        self._btn_step.config(state="disabled")
        self._btn_back.config(state="disabled")
        self._btn_stop = btn(ctrl, "Stop", self._stop_solver, "#8a3a3a")
        self._btn_stop.config(state="disabled")

        tk.Label(ctrl, text="Max nodes", font=FONT_LABEL, bg="#1e2030", fg="#aabbcc").pack(
            side="left", padx=(12, 4)
        )
        self._max_nodes_entry = tk.Entry(
            ctrl,
            textvariable=self._max_nodes_var,
            width=9,
            font=FONT_STATUS,
            bg="#0f1320",
            fg="#e4eef8",
            insertbackground="#e4eef8",
            relief="flat",
            justify="right",
        )
        self._max_nodes_entry.pack(side="left")

        tk.Label(ctrl, text="Speed", font=FONT_LABEL, bg="#1e2030", fg="#aabbcc").pack(
            side="left", padx=(12, 4)
        )
        self._speed_scale = ttk.Scale(
            ctrl,
            from_=MIN_SPEED,
            to=MAX_SPEED,
            variable=self._speed_var,
            orient="horizontal",
            length=120,
        )
        self._speed_scale.pack(side="left")
        tk.Checkbutton(
            ctrl,
            text="Autoplay",
            variable=self._autoplay_var,
            font=FONT_LABEL,
            bg="#1e2030",
            fg="#aabbcc",
            activebackground="#1e2030",
            activeforeground="#e4eef8",
            selectcolor="#101624",
            relief="flat",
            highlightthickness=0,
        ).pack(side="left", padx=(10, 0))

        info = tk.Frame(self, bg="#151520", padx=10, pady=6)
        info.pack(fill="x", padx=8, pady=(0, 4))

        def stat(parent, title, variable):
            box = tk.Frame(parent, bg="#151520")
            box.pack(side="left", padx=12)
            tk.Label(box, text=title, font=FONT_LABEL, bg="#151520", fg="#7fa4c4").pack(anchor="w")
            tk.Label(box, textvariable=variable, font=FONT_BTN, bg="#151520", fg="#e4eef8").pack(
                anchor="w"
            )

        stat(info, "Algorithm", self._live_solver_name)
        stat(info, "Board", self._board_source_var)
        stat(info, "Status", self._live_solver_status)
        stat(info, "Moves", self._live_moves_var)
        stat(info, "Time", self._live_time_var)
        stat(info, "Memory", self._live_memory_var)
        stat(info, "Expanded", self._live_nodes_var)
        stat(info, "Generated", self._live_generated_var)
        stat(info, "Frontier", self._live_frontier_var)
        stat(info, "Search length", self._live_search_var)

        tk.Label(
            self,
            textvariable=self._status_var,
            font=FONT_STATUS,
            bg="#151520",
            fg="#aabbcc",
            anchor="w",
            padx=8,
        ).pack(fill="x", side="bottom")

        self._canvas = tk.Canvas(self, width=CANVAS_W, height=CANVAS_H, bg=BG_FELT, highlightthickness=0)
        self._canvas.pack(padx=8, pady=(0, 4))
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<Button-3>", self._on_right_click)

        self._locked_during_solver = (
            self._btn_new_game,
            self._btn_sample,
            self._btn_restart,
            self._btn_undo,
            self._btn_bfs,
            self._btn_dfs,
            self._btn_ucs,
            self._btn_astar,
            self._btn_expert,
            self._max_nodes_entry,
        )

        self._init_canvas_scene()

    def _init_canvas_scene(self) -> None:
        canvas = self._canvas
        canvas.create_text(
            FC_X[0] + CW // 2,
            FC_Y - 12,
            text="FREE CELLS",
            fill=CLR_LABEL,
            font=("Helvetica", 9, "bold"),
            anchor="center",
        )
        canvas.create_text(
            FD_X[0] + CW // 2,
            FD_Y - 12,
            text="FOUNDATIONS",
            fill=CLR_LABEL,
            font=("Helvetica", 9, "bold"),
            anchor="center",
        )

        self._freecell_slots = [
            self._create_slot_items(FC_X[idx], FC_Y, f"FC {idx + 1}") for idx in range(4)
        ]

        self._foundation_slots = []
        self._foundation_hint_labels = []
        for suit in range(4):
            self._foundation_slots.append(self._create_slot_items(FD_X[suit], FD_Y, SUIT_SYMBOLS[suit]))
            self._foundation_hint_labels.append(
                canvas.create_text(
                    FD_X[suit] + CW // 2,
                    FD_Y + CH // 2 + 28,
                    text="",
                    fill=CLR_HINT,
                    font=FONT_LABEL,
                    anchor="center",
                    state="hidden",
                )
            )

        self._cascade_slots = []
        self._cascade_hint_rects = []
        for idx in range(8):
            x = CAS_X[idx]
            canvas.create_text(x + CW // 2, CAS_Y - 14, text=f"C{idx + 1}", fill=CLR_LABEL, font=FONT_LABEL)
            self._cascade_slots.append(self._create_slot_items(x, CAS_Y, "Empty"))
            self._cascade_hint_rects.append(
                canvas.create_rectangle(
                    x,
                    CAS_Y,
                    x + CW,
                    CAS_Y + CH,
                    outline=CLR_HINT,
                    width=3,
                    fill="",
                    state="hidden",
                )
            )

        for card_id in range(52):
            self._card_items[card_id] = self._create_card_items(card_id)

        self._win_overlay_items = [
            canvas.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill="#000000", stipple="gray50", state="hidden"),
            canvas.create_text(
                CANVAS_W // 2,
                CANVAS_H // 2 - 30,
                text="YOU WIN!",
                fill="#ffe44d",
                font=("Helvetica", 42, "bold"),
                anchor="center",
                state="hidden",
            ),
            canvas.create_text(
                CANVAS_W // 2,
                CANVAS_H // 2 + 30,
                text="All 52 cards moved to foundations.",
                fill="#ffffff",
                font=("Helvetica", 16),
                anchor="center",
                state="hidden",
            ),
        ]

    def _create_slot_items(self, x: int, y: int, label: str) -> Dict[str, int]:
        rect = self._canvas.create_rectangle(
            x,
            y,
            x + CW,
            y + CH,
            fill=BG_EMPTY,
            outline=CLR_SLOT,
            width=2,
            dash=(6, 4),
        )
        text = self._canvas.create_text(
            x + CW // 2,
            y + CH // 2,
            text=label,
            fill=CLR_LABEL,
            font=FONT_LABEL,
            anchor="center",
        )
        return {"rect": rect, "label": text}

    def _create_card_items(self, card_id: int) -> Dict[str, object]:
        card = Card.from_int(card_id)
        text_color = CLR_RED if card.is_red else CLR_BLACK
        tag = f"card_{card_id}"
        shadow = self._canvas.create_rectangle(0, 0, 0, 0, fill="#000", outline="", state="hidden", tags=(tag,))
        body = self._canvas.create_rectangle(
            0,
            0,
            0,
            0,
            fill=CLR_WHITE,
            outline="#888",
            width=1,
            state="hidden",
            tags=(tag,),
        )
        top_rank = self._canvas.create_text(
            0,
            0,
            text=f"{card.rank_str}{card.suit_symbol}",
            fill=text_color,
            font=FONT_CORNER,
            anchor="nw",
            state="hidden",
            tags=(tag,),
        )
        top_suit = self._canvas.create_text(
            0,
            0,
            text="",
            fill=text_color,
            font=FONT_RANK_SM,
            anchor="nw",
            state="hidden",
            tags=(tag,),
        )
        center_suit = self._canvas.create_text(
            0,
            0,
            text=card.suit_symbol,
            fill=text_color,
            font=FONT_SUIT_LG,
            anchor="center",
            state="hidden",
            tags=(tag,),
        )
        top_rank_right = self._canvas.create_text(
            0,
            0,
            text=f"{card.rank_str}{card.suit_symbol}",
            fill=text_color,
            font=FONT_CORNER,
            anchor="ne",
            state="hidden",
            tags=(tag,),
        )
        top_suit_right = self._canvas.create_text(
            0,
            0,
            text="",
            fill=text_color,
            font=FONT_RANK_SM,
            anchor="ne",
            state="hidden",
            tags=(tag,),
        )
        bottom_suit = self._canvas.create_text(
            0,
            0,
            text=card.suit_symbol,
            fill=text_color,
            font=FONT_RANK_SM,
            anchor="se",
            state="hidden",
            tags=(tag,),
        )
        bottom_rank = self._canvas.create_text(
            0,
            0,
            text=card.rank_str,
            fill=text_color,
            font=FONT_RANK_SM,
            anchor="se",
            state="hidden",
            tags=(tag,),
        )
        return {
            "tag": tag,
            "all": (
                shadow,
                body,
                top_rank,
                top_suit,
                center_suit,
                top_rank_right,
                top_suit_right,
                bottom_suit,
                bottom_rank,
            ),
            "body": body,
            "shadow": shadow,
            "top_rank": top_rank,
            "top_suit": top_suit,
            "center_suit": center_suit,
            "top_rank_right": top_rank_right,
            "top_suit_right": top_suit_right,
            "bottom_suit": bottom_suit,
            "bottom_rank": bottom_rank,
        }

    def _configure_slot(self, slot: Dict[str, int], label: str, hint: bool) -> None:
        outline = CLR_HINT if hint else CLR_SLOT
        width = 3 if hint else 2
        self._canvas.itemconfigure(slot["rect"], outline=outline, width=width)
        self._canvas.itemconfigure(slot["label"], text=label, fill=CLR_LABEL, state="normal")

    def _place_card_item(self, card_id: int, x: int, y: int, highlight: str = "") -> None:
        items = self._card_items[card_id]
        outline = highlight or "#888"
        width = 3 if highlight else 1

        self._canvas.coords(items["shadow"], x + 3, y + 3, x + CW + 3, y + CH + 3)
        self._canvas.coords(items["body"], x, y, x + CW, y + CH)
        self._canvas.coords(items["top_rank"], x + 7, y + 6)
        self._canvas.coords(items["top_suit"], x + 7, y + 20)
        self._canvas.coords(items["center_suit"], x + CW // 2, y + CH // 2)
        self._canvas.coords(items["top_rank_right"], x + CW - 7, y + 6)
        self._canvas.coords(items["top_suit_right"], x + CW - 7, y + 20)
        self._canvas.coords(items["bottom_suit"], x + CW - 7, y + CH - 20)
        self._canvas.coords(items["bottom_rank"], x + CW - 7, y + CH - 7)
        self._canvas.itemconfigure(items["body"], outline=outline, width=width)

        for item_id in items["all"]:
            self._canvas.itemconfigure(item_id, state="normal")

    def _hide_card_item(self, card_id: int) -> None:
        for item_id in self._card_items[card_id]["all"]:
            self._canvas.itemconfigure(item_id, state="hidden")

    def _set_win_overlay(self, visible: bool) -> None:
        state = "normal" if visible else "hidden"
        for item_id in self._win_overlay_items:
            self._canvas.itemconfigure(item_id, state=state)
            if visible:
                self._canvas.tag_raise(item_id)

    def _cancel_pending_playback(self) -> None:
        self._playing_solution = False

        if self._pending_play_job is not None:
            self.after_cancel(self._pending_play_job)
            self._pending_play_job = None

        if self._pending_animation_job is not None:
            self.after_cancel(self._pending_animation_job)
            self._pending_animation_job = None

        self._active_animation = None
        self._sequence_state = None
        self._sequence_queue = []
        self._sequence_done = None
        self._preview_state = None
        self._sync_playback_controls()

    def _clear_solution(self) -> None:
        self._playing_solution = False
        self._replay_moves = []
        self._replay_states = []
        self._replay_cursor = 0
        self._replay_label = "Replay Solution"
        self._replay_message = ""
        self._replay_is_solution = True
        self._replay_kind = ""
        self._sync_playback_controls()

    def _sync_playback_controls(self) -> None:
        trace_len = len(self._replay_moves)
        has_trace = trace_len > 0
        can_play = (
            has_trace
            and self._replay_cursor < trace_len
            and not self._playing_solution
            and self._active_animation is None
            and not self._solving
        )
        can_pause = has_trace and self._playing_solution
        can_step = (
            has_trace
            and self._replay_cursor < trace_len
            and not self._playing_solution
            and self._active_animation is None
            and not self._solving
        )
        can_back = (
            has_trace
            and self._replay_cursor > 0
            and not self._playing_solution
            and self._active_animation is None
            and not self._solving
        )

        self._btn_play.config(state="normal" if can_play else "disabled")
        self._btn_pause.config(state="normal" if can_pause else "disabled")
        self._btn_step.config(state="normal" if can_step else "disabled")
        self._btn_back.config(state="normal" if can_back else "disabled")

    def _set_solver_controls(self, solving: bool) -> None:
        state = "disabled" if solving else "normal"
        for widget in self._locked_during_solver:
            widget.config(state=state)
        self._btn_stop.config(state="normal" if solving else "disabled", text="Stop")

    def _build_initial_state(self, cascades: List[List[Card]]) -> GameState:
        return GameState.initial([list(col) for col in cascades])

    def _reset_board(self, state: GameState) -> None:
        self._state = state
        self._history = []
        self._selection.clear()
        self._preview_state = None
        self._render()

    def _card_key(self, card: Card) -> int:
        return card.to_int()

    def _layout_positions(self, state: GameState) -> Dict[int, Tuple[int, int]]:
        positions: Dict[int, Tuple[int, int]] = {}

        for idx, card in enumerate(state.free_cells):
            if card is not None:
                positions[self._card_key(card)] = (FC_X[idx], FC_Y)

        for suit, top_rank in enumerate(state.foundations):
            if top_rank > 0:
                positions[self._card_key(Card(top_rank, suit))] = (FD_X[suit], FD_Y)

        for col_idx, col in enumerate(state.cascades):
            spacing = self._cascade_spacing(len(col))
            for row_idx, card in enumerate(col):
                positions[self._card_key(card)] = (CAS_X[col_idx], CAS_Y + row_idx * spacing)

        return positions

    def _cascade_spacing(self, card_count: int) -> int:
        return compute_cascade_spacing(
            CANVAS_H - CAS_Y - BOARD_BOTTOM_MARGIN,
            card_count,
            CH,
            MIN_VISIBLE_CASCADE_STRIP,
        )

    def _build_step_queue(self, state: GameState, move: Move, prefix: str = "") -> List[Tuple[Move, str]]:
        if self._autoplay_var.get():
            _, steps = apply_move_with_auto_steps(state, move)
        else:
            steps = [move]
        queue: List[Tuple[Move, str]] = []
        current = state

        for idx, step in enumerate(steps):
            if idx == 0:
                status = f"{prefix}{step.description(current)}".strip()
            else:
                status = f"{prefix}Auto: {step.description(current)}".strip()
            queue.append((step, status))
            current = apply_move(current, step)

        return queue

    def _move_duration_ms(
        self,
        start_positions: Dict[int, Tuple[int, int]],
        end_positions: Dict[int, Tuple[int, int]],
        card_ids: List[int],
    ) -> int:
        distance = 0
        for card_id in card_ids:
            sx, sy = start_positions[card_id]
            ex, ey = end_positions[card_id]
            distance = max(distance, abs(ex - sx) + abs(ey - sy))

        speed = max(self._speed_var.get(), MIN_SPEED)
        raw = MOVE_DURATION_MS + distance * 0.25 + max(0, len(card_ids) - 1) * 24
        return int(max(90, min(380, raw)) / speed)

    def _start_move_sequence(
        self,
        base_state: GameState,
        queue: List[Tuple[Move, str]],
        on_complete: Optional[Callable[[], None]] = None,
        commit_state: bool = True,
    ) -> None:
        if not queue:
            if on_complete is not None:
                on_complete()
            return

        self._sequence_state = base_state
        self._sequence_queue = list(queue)
        self._sequence_done = on_complete
        self._sequence_commit_state = commit_state
        self._sync_playback_controls()
        self._run_next_sequence_step()

    def _run_next_sequence_step(self) -> None:
        if self._sequence_state is None:
            return

        if not self._sequence_queue:
            done = self._sequence_done
            self._sequence_state = None
            self._sequence_done = None
            self._sync_playback_controls()
            if done is not None:
                done()
            return

        move, status = self._sequence_queue.pop(0)
        self._animate_single_move(self._sequence_state, move, status)

    def _animate_single_move(self, before_state: GameState, move: Move, status: str) -> None:
        if move.src_type == "cascade":
            cards = list(before_state.cascades[move.src_idx][-move.num_cards :])
        else:
            card = before_state.free_cells[move.src_idx]
            cards = [card] if card is not None else []

        if not cards:
            return

        after_state = apply_move(before_state, move)
        start_positions = self._layout_positions(before_state)
        end_positions = self._layout_positions(after_state)
        card_ids = [self._card_key(card) for card in cards]
        max_distance = 0
        for card_id in card_ids:
            sx, sy = start_positions[card_id]
            ex, ey = end_positions[card_id]
            max_distance = max(max_distance, abs(ex - sx) + abs(ey - sy))

        duration_s = self._move_duration_ms(start_positions, end_positions, card_ids) / 1000.0
        if not self._sequence_commit_state:
            duration_s *= 0.62

        self._active_animation = {
            "before_state": before_state,
            "after_state": after_state,
            "commit_state": self._sequence_commit_state,
            "status": status,
            "move": move,
            "cards": cards,
            "card_ids": card_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "lift_px": max(18, min(56, int(max_distance * 0.09) or 18)),
            "started_at": time.perf_counter(),
            "duration_s": duration_s,
        }
        if status:
            self._set_status(status)
        self._tick_animation()

    def _tick_animation(self) -> None:
        animation = self._active_animation
        self._pending_animation_job = None
        if animation is None:
            return

        elapsed = time.perf_counter() - animation["started_at"]
        duration = max(animation["duration_s"], 0.001)
        progress = min(1.0, elapsed / duration)
        eased = progress * progress * (3.0 - 2.0 * progress)

        before_state = animation["before_state"]
        move = animation["move"]
        cards = animation["cards"]
        card_ids = animation["card_ids"]
        start_positions = animation["start_positions"]
        end_positions = animation["end_positions"]
        lift_px = animation["lift_px"]

        self._render(before_state, hidden_card_ids=set(card_ids), forced_dst=(move.dst_type, move.dst_idx))

        for card in cards:
            card_id = self._card_key(card)
            sx, sy = start_positions[card_id]
            ex, ey = end_positions[card_id]
            x = round(sx + (ex - sx) * eased)
            arc = 4 * progress * (1 - progress)
            y = round(sy + (ey - sy) * eased - lift_px * arc)
            self._place_card_item(card_id, x, y, highlight=CLR_SEL)
            self._canvas.tag_raise(self._card_items[card_id]["tag"])

        if progress < 1.0:
            self._pending_animation_job = self.after(ANIMATION_FRAME_MS, self._tick_animation)
            return

        if animation["commit_state"]:
            self._preview_state = None
            self._state = animation["after_state"]
        else:
            self._preview_state = animation["after_state"]

        self._sequence_state = animation["after_state"]
        self._active_animation = None
        self._render()
        self._run_next_sequence_step()

    def _load_board(
        self,
        cascades: List[List[Card]],
        *,
        board_label: str,
        deal_number: Optional[int],
        status_message: str,
    ) -> None:
        self._cancel_pending_playback()
        self._deal_number = deal_number
        self._initial_cascades = [list(col) for col in cascades]
        self._initial_state = self._build_initial_state(self._initial_cascades)
        self._board_source_var.set(board_label)
        self._clear_solution()
        self._reset_board(self._initial_state)
        self._set_status(status_message)

    def _load_microsoft_deal(self, number: int) -> None:
        self._load_board(
            ms_deal(number),
            board_label=f"Microsoft Deal #{number}",
            deal_number=number,
            status_message=f"Microsoft Deal #{number} - Click a card to select it.",
        )

    def _load_random_board(self, seed: Optional[int] = None) -> None:
        self._load_board(
            random_deal(seed),
            board_label="Random Shuffled Board",
            deal_number=None,
            status_message="Random Shuffled Board - Click a card to select it.",
        )

    def _load_sample_board(self, name: str) -> None:
        state = get_sample_board(name)
        self._cancel_pending_playback()
        self._deal_number = None
        self._initial_cascades = [list(col) for col in state.cascades]
        self._initial_state = state
        self._board_source_var.set(f"Sample Board: {name}")
        self._clear_solution()
        self._reset_board(state)
        self._set_status(f"Sample Board: {name} - Click a card to select it.")

    def _new_game_number(self, number: int) -> None:
        self._load_microsoft_deal(number)

    def _sample_board_dialog(self) -> None:
        answer = simpledialog.askstring(
            "Sample Board",
            f"Enter sample name: {', '.join(SAMPLE_BOARD_NAMES)}",
            parent=self,
        )
        if answer is None:
            return

        sample_name = answer.strip()
        if sample_name not in SAMPLE_BOARD_NAMES:
            messagebox.showerror("Invalid sample", f"Choose one of: {', '.join(SAMPLE_BOARD_NAMES)}")
            return

        self._load_sample_board(sample_name)

    def _new_game_dialog(self) -> None:
        answer = simpledialog.askstring(
            "New Game",
            "Enter an MS FreeCell deal number (1-1000000),\nleave blank for a random Microsoft deal,\nor type RANDOM for a shuffled random board:",
            parent=self,
        )
        if answer is None:
            return

        answer = answer.strip()
        if not answer:
            self._load_microsoft_deal(random.randint(1, 1_000_000))
            return

        if answer.upper().startswith("SAMPLE:"):
            sample_name = answer.split(":", 1)[1].strip()
            if sample_name not in SAMPLE_BOARD_NAMES:
                messagebox.showerror("Invalid sample", f"Choose one of: {', '.join(SAMPLE_BOARD_NAMES)}")
                return
            self._load_sample_board(sample_name)
            return

        if answer.upper() == "RANDOM":
            self._load_random_board()
            return

        try:
            deal_number = int(answer)
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a positive integer.")
            return

        if deal_number < 1:
            messagebox.showerror("Invalid input", "Please enter a positive integer.")
            return

        self._new_game_number(deal_number)

    def _restart(self) -> None:
        if self._initial_state is None:
            return

        self._cancel_pending_playback()
        self._clear_solution()
        self._reset_board(self._initial_state)
        self._set_status(f"{self._board_source_var.get()} - restarted.")

    def _undo(self) -> None:
        if not self._history:
            self._set_status("Nothing to undo.")
            return

        self._cancel_pending_playback()
        self._clear_solution()
        self._state = self._history.pop()
        self._selection.clear()
        self._render()
        self._set_status("Undo complete.")

    def _render(
        self,
        state: Optional[GameState] = None,
        hidden_card_ids: Optional[Set[int]] = None,
        forced_dst: Optional[Tuple[str, int]] = None,
    ) -> None:
        display_state = state or self._preview_state or self._state
        if display_state is None:
            return

        hidden = hidden_card_ids or set()
        for card_id in hidden:
            self._hide_card_item(card_id)

        selection = self._selection if not hidden else Selection()
        dst_highlights: Dict[Tuple[str, int], bool] = {}
        if forced_dst is not None:
            dst_highlights[forced_dst] = True
        elif selection.is_active():
            for move in get_valid_moves(display_state):
                if move.src_type == selection.src_type and move.src_idx == selection.src_idx:
                    dst_highlights[(move.dst_type, move.dst_idx)] = True

        visible_ids: Set[int] = set()
        draw_order: List[int] = []

        for idx, card in enumerate(display_state.free_cells):
            hint = ("freecell", idx) in dst_highlights
            card_id = self._card_key(card) if card is not None else None
            visible = card is not None and card_id not in hidden
            self._configure_slot(self._freecell_slots[idx], f"FC {idx + 1}", hint and not visible)
            if not visible or card_id is None:
                continue

            highlight = ""
            if selection.src_type == "freecell" and selection.src_idx == idx:
                highlight = CLR_SEL
            elif hint:
                highlight = CLR_HINT
            self._place_card_item(card_id, FC_X[idx], FC_Y, highlight=highlight)
            visible_ids.add(card_id)
            draw_order.append(card_id)

        for suit, top_rank in enumerate(display_state.foundations):
            hint = ("foundation", suit) in dst_highlights
            self._configure_slot(self._foundation_slots[suit], SUIT_SYMBOLS[suit], hint and top_rank == 0)
            hint_label = self._foundation_hint_labels[suit]

            if top_rank > 0:
                card = Card(top_rank, suit)
                card_id = self._card_key(card)
                if card_id not in hidden:
                    self._place_card_item(card_id, FD_X[suit], FD_Y, highlight=CLR_HINT if hint else "")
                    visible_ids.add(card_id)
                    draw_order.append(card_id)

            if hint and 0 < top_rank < 13:
                self._canvas.itemconfigure(hint_label, text=f"Next {top_rank + 1}", state="normal")
            else:
                self._canvas.itemconfigure(hint_label, state="hidden")

        for col_idx, col in enumerate(display_state.cascades):
            x = CAS_X[col_idx]
            hint_col = ("cascade", col_idx) in dst_highlights
            visible_cards = [card for card in col if self._card_key(card) not in hidden]
            self._configure_slot(self._cascade_slots[col_idx], "Empty", hint_col and not visible_cards)

            selected_start = len(col)
            if selection.is_active() and selection.src_type == "cascade" and selection.src_idx == col_idx:
                selected_start = len(col) - selection.num_cards

            if not visible_cards:
                self._canvas.itemconfigure(self._cascade_hint_rects[col_idx], state="hidden")
                continue

            for row_idx, card in enumerate(visible_cards):
                original_row_idx = len(col) - len(visible_cards) + row_idx
                spacing = self._cascade_spacing(len(col))
                y = CAS_Y + row_idx * spacing
                highlight = CLR_SEL if original_row_idx >= selected_start else ""
                card_id = self._card_key(card)
                self._place_card_item(card_id, x, y, highlight=highlight)
                visible_ids.add(card_id)
                draw_order.append(card_id)

            if hint_col:
                y_top = CAS_Y + (len(visible_cards) - 1) * self._cascade_spacing(len(col))
                hint_rect = self._cascade_hint_rects[col_idx]
                self._canvas.coords(hint_rect, x, y_top, x + CW, y_top + CH)
                self._canvas.itemconfigure(hint_rect, state="normal")
            else:
                self._canvas.itemconfigure(self._cascade_hint_rects[col_idx], state="hidden")

        for card_id in self._visible_card_ids - visible_ids:
            self._hide_card_item(card_id)
        self._visible_card_ids = visible_ids

        for card_id in draw_order:
            self._canvas.tag_raise(self._card_items[card_id]["tag"])

        for hint_label in self._foundation_hint_labels:
            self._canvas.tag_raise(hint_label)
        for hint_rect in self._cascade_hint_rects:
            self._canvas.tag_raise(hint_rect)

        self._set_win_overlay(display_state.is_goal())

    def _on_click(self, event) -> None:
        if self._solving or self._active_animation is not None or self._state is None or self._state.is_goal():
            return

        loc = self._hit_test(event.x, event.y)
        if loc is None:
            self._selection.clear()
            self._render()
            return

        self._handle_location_click(loc)

    def _on_right_click(self, _event) -> None:
        if self._solving or self._active_animation is not None:
            return
        self._selection.clear()
        self._render()

    def _hit_test(self, x: int, y: int) -> Optional[Tuple]:
        if self._state is None:
            return None

        for i in range(4):
            if FC_X[i] <= x <= FC_X[i] + CW and FC_Y <= y <= FC_Y + CH:
                return ("freecell", i)

        for i in range(4):
            if FD_X[i] <= x <= FD_X[i] + CW and FD_Y <= y <= FD_Y + CH:
                return ("foundation", i)

        for ci in range(8):
            cx = CAS_X[ci]
            if not (cx <= x <= cx + CW):
                continue
            col = self._state.cascades[ci]
            spacing = self._cascade_spacing(len(col))
            if not col:
                if CAS_Y <= y <= CAS_Y + CH:
                    return ("cascade_empty", ci)
                continue

            for ri in range(len(col) - 1, -1, -1):
                y_card = CAS_Y + ri * spacing
                y_bottom = y_card + (spacing if ri < len(col) - 1 else CH)
                if y_card <= y <= y_bottom:
                    return ("cascade_card", ci, ri)

        return None

    def _handle_location_click(self, loc: Tuple) -> None:
        if self._state is None:
            return

        state = self._state
        selection = self._selection
        loc_type = loc[0]

        if loc_type == "freecell":
            idx = loc[1]
            card = state.free_cells[idx]
            if selection.is_active():
                self._try_move_to("freecell", idx)
            elif card is not None:
                selection.set("freecell", idx, 1)
                self._render()
            return

        if loc_type == "foundation":
            if selection.is_active():
                self._try_move_to("foundation", loc[1])
            return

        if loc_type == "cascade_empty":
            if selection.is_active():
                self._try_move_to("cascade", loc[1])
            return

        if loc_type != "cascade_card":
            return

        col_idx, row_idx = loc[1], loc[2]
        col = state.cascades[col_idx]
        num_cards_from_here = len(col) - row_idx

        if (
            selection.is_active()
            and selection.src_type == "cascade"
            and selection.src_idx == col_idx
            and row_idx >= len(col) - selection.num_cards
        ):
            selection.clear()
            self._render()
            return

        if selection.is_active():
            self._try_move_to("cascade", col_idx)
            return

        max_seq = num_cards_from_here
        for offset in range(1, num_cards_from_here):
            above = col[row_idx + offset]
            below = col[row_idx + offset - 1]
            if not above.can_stack_on(below):
                max_seq = offset
                break

        from game.moves import _max_seq_len

        movable = _max_seq_len(state.empty_free_cells, state.empty_cascades, False)
        selection.set("cascade", col_idx, max(1, min(max_seq, movable)))
        self._render()

    def _try_move_to(self, dst_type: str, dst_idx: int) -> None:
        if self._state is None:
            return

        selection = self._selection
        move = Move(
            src_type=selection.src_type,
            src_idx=selection.src_idx,
            dst_type=dst_type,
            dst_idx=dst_idx,
            num_cards=selection.num_cards if dst_type == "cascade" else 1,
        )

        matched = None
        for legal_move in get_valid_moves(self._state):
            if (
                legal_move.src_type == move.src_type
                and legal_move.src_idx == move.src_idx
                and legal_move.dst_type == move.dst_type
                and legal_move.dst_idx == move.dst_idx
            ):
                matched = legal_move
                break

        if matched is None:
            selection.clear()
            self._set_status("Invalid move.")
            self._render()
            return

        base_state = self._state
        self._cancel_pending_playback()
        self._clear_solution()
        self._history.append(base_state)
        selection.clear()
        queue = self._build_step_queue(base_state, matched)

        def done() -> None:
            self._sync_playback_controls()
            if self._state is not None and self._state.is_goal():
                self._set_status("Congratulations - you solved it!")

        self._start_move_sequence(base_state, queue, done)

    def _commit_replay_trace(
        self,
        trace: List[Move],
        *,
        replay_label: str = "Replay Solution",
        replay_message: str = "",
        is_solution_trace: bool = False,
        trace_kind: str = "",
    ) -> None:
        if self._initial_state is None:
            return

        self._cancel_pending_playback()
        self._history = []
        self._selection.clear()
        self._replay_moves = list(trace)
        self._replay_states = [self._initial_state]
        state = self._initial_state
        for move in self._replay_moves:
            state = apply_move(state, move)
            self._replay_states.append(state)

        self._replay_cursor = 0
        self._state = self._initial_state
        self._replay_label = replay_label
        self._replay_message = replay_message
        self._replay_is_solution = is_solution_trace
        self._replay_kind = trace_kind or ("solution" if is_solution_trace else "best-progress")
        self._render()
        self._sync_playback_controls()

    def _load_replay_trace(
        self,
        trace: List[Move],
        *,
        autoplay: bool = True,
        replay_label: str = "Replay Solution",
        replay_message: str = "",
        is_solution_trace: bool = False,
    ) -> None:
        self._commit_replay_trace(
            trace,
            replay_label=replay_label,
            replay_message=replay_message,
            is_solution_trace=is_solution_trace,
        )
        if replay_message:
            self._set_status(replay_message)
        else:
            self._set_status(f"Replay loaded: {len(trace)} moves. Press Play for replay.")

        if autoplay and self._replay_moves:
            self._play_replay()

    def _schedule_next_replay_move(self, delay_ms: int) -> None:
        if self._pending_play_job is not None:
            self.after_cancel(self._pending_play_job)
        self._pending_play_job = self.after(delay_ms, self._play_replay_move)

    def _play_replay(self) -> None:
        if not self._replay_moves or self._replay_cursor >= len(self._replay_moves):
            return

        self._playing_solution = True
        self._sync_playback_controls()
        if self._active_animation is None:
            self._schedule_next_replay_move(0)

    def _pause_replay(self) -> None:
        if not self._replay_moves:
            return

        self._playing_solution = False
        if self._pending_play_job is not None:
            self.after_cancel(self._pending_play_job)
            self._pending_play_job = None
        self._sync_playback_controls()
        if self._active_animation is None:
            self._set_status(
                f"Paused at step {self._replay_cursor} of {len(self._replay_moves)}."
            )

    def _step_replay(self) -> None:
        if not self._replay_moves or self._active_animation is not None:
            return

        self._playing_solution = False
        if self._pending_play_job is not None:
            self.after_cancel(self._pending_play_job)
            self._pending_play_job = None
        self._sync_playback_controls()
        self._play_replay_move()

    def _back_replay(self) -> None:
        if self._replay_cursor <= 0 or self._active_animation is not None or self._solving:
            return

        self._playing_solution = False
        if self._pending_play_job is not None:
            self.after_cancel(self._pending_play_job)
            self._pending_play_job = None
        self._replay_cursor -= 1
        self._state = self._replay_states[self._replay_cursor]
        self._preview_state = None
        self._render()
        self._sync_playback_controls()
        self._set_status(
            f"Paused at step {self._replay_cursor} of {len(self._replay_moves)}."
        )

    def _play_replay_move(self) -> None:
        self._pending_play_job = None

        if self._state is None or self._active_animation is not None:
            return

        if self._replay_cursor >= len(self._replay_moves):
            self._playing_solution = False
            self._sync_playback_controls()
            if not self._replay_is_solution and not (self._state and self._state.is_goal()):
                self._set_status("Best-progress replay complete.")
            else:
                self._set_status("Solution complete - all moves played.")
            return

        move = self._replay_moves[self._replay_cursor]
        step_no = self._replay_cursor + 1
        total_steps = len(self._replay_moves)
        base_state = self._replay_states[self._replay_cursor]
        queue = [(move, f"Replaying step {step_no} of {total_steps}")]

        def done() -> None:
            self._replay_cursor += 1
            if self._replay_cursor < len(self._replay_states):
                self._state = self._replay_states[self._replay_cursor]
            self._sync_playback_controls()
            if self._state is None:
                return
            if self._state.is_goal():
                self._playing_solution = False
                self._sync_playback_controls()
                self._set_status("Solved, replay complete.")
                return
            if self._replay_cursor >= len(self._replay_moves):
                self._playing_solution = False
                self._sync_playback_controls()
                if not self._replay_is_solution and not self._state.is_goal():
                    self._set_status("Best-progress replay complete.")
                else:
                    self._set_status("Solution complete - all moves played.")
                return
            if self._playing_solution:
                delay = int(BETWEEN_MOVES_MS / max(self._speed_var.get(), MIN_SPEED))
                self._schedule_next_replay_move(max(20, delay))
            else:
                self._set_status(
                    f"Paused at step {self._replay_cursor} of {len(self._replay_moves)}."
                )

        self._start_move_sequence(base_state, queue, done)

    def _set_live_stats(
        self,
        algorithm: str,
        status: str,
        moves: int,
        elapsed_time: float,
        memory_kb: float,
        expanded_nodes: int,
        generated_nodes: int,
        frontier_size: int,
        search_length: int,
    ) -> None:
        self._live_solver_name.set(algorithm)
        self._live_solver_status.set(status)
        self._live_moves_var.set(f"{moves:,}")
        self._live_time_var.set(f"{elapsed_time:.2f} s")
        self._live_memory_var.set(f"{memory_kb:,.0f} KB")
        self._live_nodes_var.set(f"{expanded_nodes:,}")
        self._live_generated_var.set(f"{generated_nodes:,}")
        self._live_frontier_var.set(f"{frontier_size:,}")
        self._live_search_var.set(f"{search_length:,}")

    def _create_solver(self, algo: str, progress_callback):
        from solvers.astar import AStarSolver
        from solvers.bfs import BFSSolver
        from solvers.dfs import DFSSolver
        from solvers.expert_solver import ExpertSolver
        from solvers.ucs import UCSSolver

        solver_classes = {
            "BFS": BFSSolver,
            "DFS": DFSSolver,
            "UCS": UCSSolver,
            "ASTAR": AStarSolver,
            "EXPERT": ExpertSolver,
        }
        return solver_classes[algo](use_auto_moves=False, progress_callback=progress_callback)

    def _get_max_nodes_limit(self) -> Optional[int]:
        raw = self._max_nodes_var.get().strip().replace(",", "")
        if not raw:
            messagebox.showerror("Invalid MAX_NODES", "Enter a positive integer for MAX_NODES.")
            return None

        try:
            value = int(raw)
        except ValueError:
            messagebox.showerror("Invalid MAX_NODES", "MAX_NODES must be a positive integer.")
            return None

        if value < 1:
            messagebox.showerror("Invalid MAX_NODES", "MAX_NODES must be greater than 0.")
            return None

        self._max_nodes_var.set(str(value))
        return value

    def _schedule_solver_progress_poll(self) -> None:
        if self._solver_progress_job is None:
            self._solver_progress_job = self.after(SOLVER_POLL_MS, self._poll_solver_progress)

    def _poll_solver_progress(self) -> None:
        self._solver_progress_job = None
        snapshots = self._solver_progress_queue.drain()

        latest: Optional[SolverProgress] = None
        final_result: Optional[SolverResult] = None

        for snapshot in snapshots:
            latest = snapshot
            if snapshot.done and snapshot.result is not None:
                final_result = snapshot.result

        if latest is not None:
            self._set_live_stats(
                latest.algorithm,
                latest.status,
                latest.moves,
                latest.elapsed_time,
                latest.memory_kb,
                latest.expanded_nodes,
                latest.generated_nodes,
                latest.frontier_size,
                latest.search_length,
            )

        if final_result is not None:
            self._solver_done(final_result)

        if self._solving or not self._solver_progress_queue.empty():
            self._schedule_solver_progress_poll()

    def _stop_solver(self) -> None:
        if not self._solving or self._active_solver is None:
            return

        self._active_solver.request_stop()
        self._btn_stop.config(state="disabled", text="Stopping...")
        self._set_status(f"Stopping {self._active_solver.name}...")

    def _run_solver(self, algo: str) -> None:
        if self._solving:
            messagebox.showinfo("Busy", "A solver is already running.")
            return

        if self._active_animation is not None:
            messagebox.showinfo("Busy", "Wait for the current animation to finish.")
            return

        if self._state is None:
            return

        max_nodes_limit = self._get_max_nodes_limit()
        if max_nodes_limit is None:
            return

        self._cancel_pending_playback()
        self._clear_solution()
        self._preview_state = None
        self._solver_progress_queue = FreshQueue(maxsize=24)

        solver = self._create_solver(algo, self._solver_progress_queue.put)
        solver.MAX_NODES = max_nodes_limit
        self._active_solver = solver
        base_state = self._state

        self._solving = True
        self._set_solver_controls(True)
        self._sync_playback_controls()
        self._set_live_stats(solver.name, f"{solver.name} running", 0, 0.0, 0.0, 0, 0, 0, 0)
        self._set_status(
            f"Running {solver.name} on {self._board_source_var.get()}. "
            f"Moves shows best trace length while searching."
        )
        self._schedule_solver_progress_poll()

        def run() -> None:
            try:
                result = solver.solve(base_state)
            except Exception as exc:
                result = SolverResult(
                    algorithm=solver.name,
                    solved=False,
                    replay_trace=None,
                    replay_label="Replay Failed Attempt",
                    replay_message="Solver crashed before a useful trace was available.",
                    status="Failed",
                    message=f"Solver crashed: {exc}",
                )

            self._solver_progress_queue.put(
                SolverProgress(
                    algorithm=result.algorithm or solver.name,
                    status=result.status,
                    moves=result.display_moves,
                    elapsed_time=result.search_time,
                    memory_kb=result.memory_kb,
                    expanded_nodes=result.expanded_nodes,
                    generated_nodes=result.generated_nodes,
                    frontier_size=result.frontier_size,
                    search_length=result.search_length,
                    done=True,
                    solved=result.solved,
                    message=result.message,
                    result=result,
                )
            )

        self._solver_thread = threading.Thread(target=run, daemon=True)
        self._solver_thread.start()

    def _solver_done(self, result: SolverResult) -> None:
        self._solving = False
        self._active_solver = None
        self._solver_thread = None
        self._set_solver_controls(False)

        self._preview_state = None
        if result.replay_available:
            self._commit_replay_trace(
                result.replay_trace or [],
                replay_label=result.replay_label,
                replay_message=result.replay_message,
                is_solution_trace=result.solved,
            )
        else:
            self._render()
            self._sync_playback_controls()

        self._set_live_stats(
            result.algorithm,
            result.status,
            result.display_moves,
            result.search_time,
            result.memory_kb,
            result.expanded_nodes,
            result.generated_nodes,
            result.frontier_size,
            result.search_length,
        )

        move_label = "solution length" if result.solved else "best trace length"
        if result.replay_available and result.replay_length > 0:
            ready_text = "Solved, replay ready" if result.solved else f"{result.status}, best-progress replay ready"
        else:
            ready_text = "Solved" if result.solved else result.status or "No solution found"

        if result.replay_available and result.replay_length > 0:
            detail = ready_text
        else:
            detail = result.message or ready_text
        self._set_status(
            f"{result.algorithm}: {detail} | "
            f"{result.display_moves} {move_label} | "
            f"{result.search_time:.2f}s | "
            f"{result.expanded_nodes:,} nodes | "
            f"{result.memory_kb:.0f} KB"
        )

        ResultsDialog(self, result, lambda replay_result: self._load_replay_trace(
            replay_result.replay_trace or [],
            autoplay=True,
            replay_label=replay_result.replay_label,
            replay_message=replay_result.replay_message,
            is_solution_trace=replay_result.solved,
        ))

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)
