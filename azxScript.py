import ctypes
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import keyboard
import mss
import numpy as np
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication


VERSION = "v7-fast-detection"
TARGET_SUM = 10

# TEMPLATE MATCHING THRESHOLD - adjust this if needed
MATCH_THRESHOLD = 0.6  # Only accept matches above this confidence

# Grid geometry (cells)
rows = 16
columns = 10

# Pixels
offset_x = 51
offset_y = 52
top_start = 221
left_start = 708

# Capture size inside a tile (used for scanning)
capture_area_w = 44
capture_area_h = 45

# Highlight box uses the capture-size (most accurate to what you see),
# stepped by offset_x/offset_y.
tile_w = capture_area_w
tile_h = capture_area_h


@dataclass(frozen=True)
class Rect:
    r1: int
    c1: int
    r2: int
    c2: int


class OverlayController(QtCore.QObject):
    state_changed = QtCore.pyqtSignal(object, int, int, str)  # highlight, count, index, status


class Overlay(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.highlight: Optional[Rect] = None
        self.solution_count: int = 0
        self.solution_index: int = -1
        self.status_line: str = ""

        self.setWindowTitle("AZX Overlay")

        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool |
            QtCore.Qt.BypassWindowManagerHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        screen = QApplication.primaryScreen()
        geo = screen.geometry() if screen else QtCore.QRect(0, 0, 1920, 1080)
        self.setGeometry(geo)

        self.show()
        self.raise_()

        self._hwnd = None
        self._apply_winapi_styles()

        # Re-assert topmost periodically (some apps fight it)
        self._topmost_timer = QtCore.QTimer(self)
        self._topmost_timer.timeout.connect(self._enforce_topmost)
        self._topmost_timer.start(300)

        print(f"[overlay] Overlay window: {geo.x()},{geo.y()} {geo.width()}x{geo.height()}")

    def _apply_winapi_styles(self) -> None:
        try:
            self._hwnd = int(self.winId())
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOOLWINDOW = 0x00000080

            ex_style = ctypes.windll.user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                self._hwnd,
                GWL_EXSTYLE,
                ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
            )
        except Exception as exc:
            print(f"[overlay] WinAPI style tweak failed: {exc}")

    def _enforce_topmost(self) -> None:
        self.raise_()
        if self._hwnd is None:
            return

        try:
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            ctypes.windll.user32.SetWindowPos(
                self._hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
            )
        except Exception:
            pass

    @QtCore.pyqtSlot(object, int, int, str)
    def set_state(self, highlight_obj: object, solution_count: int, solution_index: int, status_line: str) -> None:
        self.highlight = highlight_obj if isinstance(highlight_obj, Rect) else None
        self.solution_count = solution_count
        self.solution_index = solution_index
        self.status_line = status_line
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Always-visible marker
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 255, 0, 220))
        painter.drawEllipse(10, 10, 10, 10)

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 220), 1))
        painter.setFont(QtGui.QFont("Consolas", 10))

        if self.solution_count > 0:
            idx = self.solution_index + 1 if self.solution_index >= 0 else 0
            painter.drawText(28, 20, f"OVERLAY ON | {VERSION} | solutions: {self.solution_count} | showing: {idx}/{self.solution_count}")
        else:
            painter.drawText(28, 20, f"OVERLAY ON | {VERSION} | solutions: 0")

        if self.status_line:
            painter.drawText(28, 36, self.status_line)

        # Highlight (big fill + thick border)
        if self.highlight is None:
            return

        rect = self.highlight
        x = left_start + rect.c1 * offset_x
        y = top_start + rect.r1 * offset_y
        w = (rect.c2 - rect.c1) * offset_x + tile_w
        h = (rect.r2 - rect.r1) * offset_y + tile_h

        painter.setBrush(QtGui.QColor(0, 255, 0, 55))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 255), 8))
        painter.drawRect(x, y, w, h)

        # Corner marks (very visible)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 255), 10))
        painter.drawPoint(x, y)
        painter.drawPoint(x + w, y)
        painter.drawPoint(x, y + h)
        painter.drawPoint(x + w, y + h)


# Screen-capture area of a single cell
start_area_template = {
    "top": top_start,
    "left": left_start,
    "width": capture_area_w,
    "height": capture_area_h
}

# runtime state
grid: List[List[int]] = [[0 for _ in range(columns)] for _ in range(rows)]
eliminated: List[List[bool]] = [[False for _ in range(columns)] for _ in range(rows)]

solutions: List[Rect] = []
solution_index: int = -1
current_highlight: Optional[Rect] = None
status_line: str = ""

# OPTIMIZATION: Pre-load and cache templates at startup
templates_cache: Dict[int, np.ndarray] = {}


def load_templates() -> None:
    """Pre-load all digit templates into memory (grayscale)."""
    global templates_cache
    print("Loading templates...")
    for d in range(1, 10):
        template_path = f"./templates/T{d}.png"
        template = cv2.imread(template_path)
        if template is not None:
            templates_cache[d] = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            print(f"  Loaded T{d}.png")
        else:
            print(f"  WARNING: Could not load {template_path}")
    print(f"Templates loaded: {len(templates_cache)}")


def push_overlay_state() -> None:
    controller.state_changed.emit(current_highlight, len(solutions), solution_index, status_line)


def set_status(text: str) -> None:
    global status_line
    status_line = text
    push_overlay_state()


def reset_solver_state() -> None:
    global solutions, solution_index, current_highlight
    solutions = []
    solution_index = -1
    current_highlight = None
    push_overlay_state()


def print_grid() -> None:
    print("Grid:")
    for r in range(rows):
        for c in range(columns):
            val = grid[r][c]
            print(val if val != 0 else ".", end="  ")
        print()


def find_all_rectangles_sum_target(matrix: List[List[int]], target: int) -> List[Rect]:
    """
    O(R^2 * C) rectangle-sum solver:
      - Fix top row r1
      - Accumulate column sums down to r2
      - Reduce to 1D subarray sum == target using prefix sums
    """
    r_count = len(matrix)
    c_count = len(matrix[0]) if r_count > 0 else 0

    found: List[Rect] = []

    for r1 in range(r_count):
        col_sums = [0] * c_count

        for r2 in range(r1, r_count):
            row = matrix[r2]
            for c in range(c_count):
                col_sums[c] += row[c]

            prefix = 0
            seen: Dict[int, List[int]] = {0: [-1]}

            for i, v in enumerate(col_sums):
                prefix += v
                need = prefix - target
                if need in seen:
                    for start_idx in seen[need]:
                        found.append(Rect(r1=r1, c1=start_idx + 1, r2=r2, c2=i))
                seen.setdefault(prefix, []).append(i)

    return found


def rebuild_solutions() -> None:
    global solutions, solution_index, current_highlight

    effective = [[0 for _ in range(columns)] for _ in range(rows)]
    for r in range(rows):
        for c in range(columns):
            effective[r][c] = 0 if eliminated[r][c] else int(grid[r][c])

    solutions = find_all_rectangles_sum_target(effective, TARGET_SUM)
    print(f"Found {len(solutions)} rectangles with sum == {TARGET_SUM}.")

    # IMPORTANT: immediately select the first solution so highlight is non-None.
    if solutions:
        solution_index = 0
        current_highlight = solutions[0]
        _print_current_solution_pixels("Auto-show #1")
    else:
        solution_index = -1
        current_highlight = None

    push_overlay_state()


def _print_current_solution_pixels(prefix: str) -> None:
    if current_highlight is None:
        return

    rect = current_highlight
    px_x = left_start + rect.c1 * offset_x
    px_y = top_start + rect.r1 * offset_y
    px_w = (rect.c2 - rect.c1) * offset_x + tile_w
    px_h = (rect.r2 - rect.r1) * offset_y + tile_h

    print(
        f"{prefix}: r1={rect.r1},c1={rect.c1},r2={rect.r2},c2={rect.c2} "
        f"| px=({px_x},{px_y}) size=({px_w}x{px_h})"
    )


def get_matrix_numbers() -> None:
    """
    OPTIMIZED VERSION:
    1. Single screenshot of entire grid area
    2. Process all cells from that one image
    3. Cached templates (no repeated loading)
    4. Early exit on template matching when confidence is high enough
    """
    global grid, eliminated
    import time

    eliminated = [[False for _ in range(columns)] for _ in range(rows)]
    reset_solver_state()
    set_status("Scanning...")

    start_time = time.time()

    # Calculate the full grid area
    grid_width = (columns - 1) * offset_x + capture_area_w
    grid_height = (rows - 1) * offset_y + capture_area_h
    
    full_area = {
        "top": top_start,
        "left": left_start,
        "width": grid_width,
        "height": grid_height
    }

    new_grid = [[0 for _ in range(columns)] for _ in range(rows)]

    with mss.mss() as sct:
        # OPTIMIZATION 1: Single screenshot of entire grid
        full_image = sct.grab(full_area)
        full_img_np = np.array(full_image)
        full_img_gray = cv2.cvtColor(full_img_np, cv2.COLOR_BGR2GRAY)

        for r in range(rows):
            for c in range(columns):
                # OPTIMIZATION 2: Extract cell from the full screenshot
                cell_x = c * offset_x
                cell_y = r * offset_y
                
                cell_img = full_img_gray[
                    cell_y:cell_y + capture_area_h,
                    cell_x:cell_x + capture_area_w
                ]

                best_score = -1.0
                best_match_digit: Optional[int] = None

                # OPTIMIZATION 3: Use cached templates
                for d, temp_gray in templates_cache.items():
                    res = cv2.matchTemplate(cell_img, temp_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)

                    if max_val > best_score:
                        best_score = max_val
                        best_match_digit = d
                        
                        # OPTIMIZATION 4: Early exit if confidence is very high
                        if max_val > 0.95:
                            break

                # Only accept the match if score is above threshold
                if best_score >= MATCH_THRESHOLD:
                    new_grid[r][c] = int(best_match_digit)
                else:
                    new_grid[r][c] = 0  # Empty cell

    elapsed = time.time() - start_time
    print(f"Scan completed in {elapsed:.3f} seconds")

    grid = new_grid
    print_grid()

    rebuild_solutions()
    set_status(f"Scan OK ({elapsed:.2f}s). SPACE = eliminate+next. F2 = next. F6 = test rect.")


def show_next_solution() -> None:
    global solution_index, current_highlight

    if not solutions:
        rebuild_solutions()

    if not solutions:
        current_highlight = None
        solution_index = -1
        push_overlay_state()
        return

    solution_index = (solution_index + 1) % len(solutions)
    current_highlight = solutions[solution_index]
    _print_current_solution_pixels(f"Show #{solution_index + 1}/{len(solutions)}")
    push_overlay_state()


def eliminate_current_solution() -> None:
    global current_highlight

    if current_highlight is None:
        return

    rect = current_highlight
    for r in range(rect.r1, rect.r2 + 1):
        for c in range(rect.c1, rect.c2 + 1):
            eliminated[r][c] = True


def eliminate_and_next() -> None:
    """
    SPACE: eliminate current highlight and show the next one.
    """
    if current_highlight is not None:
        eliminate_current_solution()

    rebuild_solutions()

    # After rebuild_solutions we auto-highlight #1 if any exist.
    if solutions and solution_index == 0:
        # Move to next so SPACE cycles instead of sticking at #1.
        show_next_solution()


def test_rect() -> None:
    """
    F6: force a visible test rectangle at the top-left of the board.
    If you see this, drawing works and coordinates are correct.
    """
    global current_highlight, solution_index
    current_highlight = Rect(0, 0, 0, 1)
    solution_index = -1
    _print_current_solution_pixels("TEST RECT")
    set_status("TEST RECT active (0,0)-(0,1). If you don't see it, coords/topmost/fullscreen is the issue.")
    push_overlay_state()


def clear_highlight() -> None:
    global current_highlight, solution_index
    current_highlight = None
    solution_index = -1
    push_overlay_state()


def start_logic() -> None:
    print(f"== {VERSION} ==")
    print(f"Template matching threshold: {MATCH_THRESHOLD}")
    
    # Load templates at startup
    load_templates()
    
    print("\nHotkeys:")
    print("  F5     - scan grid + auto-highlight first solution")
    print("  F2     - next solution (no elimination)")
    print("  SPACE  - eliminate current solution, rebuild, show next")
    print("  F6     - FORCE TEST RECT (0,0)-(0,1)")
    print("  F1     - clear highlight")
    print("  ESC    - quit")
    print("")
    print("OPTIMIZATIONS:")
    print("  ✓ Single screenshot of entire grid (was: 160 screenshots)")
    print("  ✓ Pre-loaded templates (was: loaded 9 images × 160 times)")
    print("  ✓ Early exit on high confidence matches")
    print("  → Expected 10-20x speed improvement")
    print("")

    keyboard.add_hotkey('f5', get_matrix_numbers)
    keyboard.add_hotkey('f2', show_next_solution)
    keyboard.add_hotkey('space', eliminate_and_next)
    keyboard.add_hotkey('f6', test_rect)
    keyboard.add_hotkey('f1', clear_highlight)
    keyboard.add_hotkey('esc', lambda: QtWidgets.QApplication.quit())

    keyboard.wait()


if __name__ == "__main__":
    app = QApplication([])

    controller = OverlayController()
    overlay = Overlay()
    controller.state_changed.connect(overlay.set_state)

    # Initial paint
    push_overlay_state()

    logic_thread = threading.Thread(target=start_logic, daemon=True)
    logic_thread.start()

    app.exec_()
