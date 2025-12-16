"""
Optimized rectangle solver with overlay visualization.

This module provides a grid-based number recognition system that finds
rectangular regions summing to a target value using template matching.
"""

import ctypes
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import keyboard
import mss
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication


# Constants
VERSION = "v8-numpy-solver"
TARGET_SUM = 10
MATCH_THRESHOLD = 0.6
TEMPLATE_DIR = Path("./templates")

# Grid configuration
GRID_ROWS = 16
GRID_COLUMNS = 10

# Pixel coordinates and offsets
OFFSET_X = 51
OFFSET_Y = 52
TOP_START = 221
LEFT_START = 708
CAPTURE_WIDTH = 44
CAPTURE_HEIGHT = 45

# Windows API constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010


@dataclass(frozen=True)
class Rectangle:
    """Represents a rectangular region in the grid."""

    r1: int
    c1: int
    r2: int
    c2: int

    def to_pixels(self) -> Tuple[int, int, int, int]:
        """Convert grid coordinates to pixel coordinates (x, y, width, height)."""
        x = LEFT_START + self.c1 * OFFSET_X
        y = TOP_START + self.r1 * OFFSET_Y
        width = (self.c2 - self.c1) * OFFSET_X + CAPTURE_WIDTH
        height = (self.r2 - self.r1) * OFFSET_Y + CAPTURE_HEIGHT
        return x, y, width, height

    def area(self) -> int:
        """Calculate the area of the rectangle."""
        return (self.r2 - self.r1 + 1) * (self.c2 - self.c1 + 1)

    def top_left_position(self) -> int:
        """Calculate position score for sorting (top-left bias)."""
        return self.r1 * GRID_COLUMNS + self.c1


class TemplateManager:
    """Manages loading and caching of digit templates."""

    def __init__(self, template_dir: Path = TEMPLATE_DIR):
        self.template_dir = template_dir
        self._cache: Dict[int, np.ndarray] = {}

    def load_templates(self) -> None:
        """Pre-load all digit templates into memory as grayscale images."""
        print("Loading templates...")
        for digit in range(1, 10):
            template_path = self.template_dir / f"T{digit}.png"
            template = cv2.imread(str(template_path))

            if template is not None:
                self._cache[digit] = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                print(f"  Loaded T{digit}.png")
            else:
                print(f"  WARNING: Could not load {template_path}")

        print(f"Templates loaded: {len(self._cache)}")

    def get_template(self, digit: int) -> Optional[np.ndarray]:
        """Retrieve a cached template."""
        return self._cache.get(digit)

    def get_all_templates(self) -> Dict[int, np.ndarray]:
        """Get all cached templates."""
        return self._cache.copy()


class GridScanner:
    """Handles grid scanning and digit recognition."""

    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager

    def scan_grid(self) -> np.ndarray:
        """
        Scan the entire grid and recognize digits using template matching.

        Returns:
            NumPy array of shape (GRID_ROWS, GRID_COLUMNS) with recognized digits.
        """
        start_time = time.time()
        grid = np.zeros((GRID_ROWS, GRID_COLUMNS), dtype=np.int8)

        # Calculate full grid dimensions
        grid_width = (GRID_COLUMNS - 1) * OFFSET_X + CAPTURE_WIDTH
        grid_height = (GRID_ROWS - 1) * OFFSET_Y + CAPTURE_HEIGHT

        capture_area = {
            "top": TOP_START,
            "left": LEFT_START,
            "width": grid_width,
            "height": grid_height,
        }

        with mss.mss() as sct:
            # Single screenshot of entire grid
            full_image = sct.grab(capture_area)
            full_img_np = np.array(full_image)
            full_img_gray = cv2.cvtColor(full_img_np, cv2.COLOR_BGR2GRAY)

            templates = self.template_manager.get_all_templates()

            for row in range(GRID_ROWS):
                cell_y = row * OFFSET_Y
                for col in range(GRID_COLUMNS):
                    cell_x = col * OFFSET_X

                    cell_img = full_img_gray[
                        cell_y : cell_y + CAPTURE_HEIGHT,
                        cell_x : cell_x + CAPTURE_WIDTH,
                    ]

                    digit = self._recognize_digit(cell_img, templates)
                    grid[row, col] = digit

        elapsed = time.time() - start_time
        print(f"Scan completed in {elapsed:.3f} seconds")
        return grid

    def _recognize_digit(
        self, cell_img: np.ndarray, templates: Dict[int, np.ndarray]
    ) -> int:
        """
        Recognize a digit in a cell image using template matching.

        Returns:
            Recognized digit (1-9) or 0 if no match found.
        """
        best_score = -1.0
        best_digit = 0

        for digit, template in templates.items():
            result = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = float(max_val)
                best_digit = digit

                # Early exit for high confidence matches
                if best_score > 0.95:
                    break

        return best_digit if best_score >= MATCH_THRESHOLD else 0


class RectangleSolver:
    """Finds rectangles in a grid that sum to a target value."""

    @staticmethod
    def find_rectangles(matrix: np.ndarray, target: int) -> List[Rectangle]:
        """
        Find all rectangles in the matrix that sum to the target value.

        Uses O(R^2 * C) algorithm with cumulative sums for efficiency.

        Args:
            matrix: 2D NumPy array of non-negative integers
            target: Target sum to find

        Returns:
            List of Rectangle objects that sum to the target
        """
        if matrix.ndim != 2:
            raise ValueError("Matrix must be 2D")

        rows, cols = matrix.shape
        if rows == 0 or cols == 0:
            return []

        found: List[Rectangle] = []

        # Pre-compute column cumulative sums
        col_sum_matrix = np.cumsum(matrix, axis=0, dtype=np.int32)

        for r1 in range(rows):
            for r2 in range(r1, rows):
                # Get column sums for this row range
                if r1 == 0:
                    col_sums_arr = col_sum_matrix[r2]
                else:
                    col_sums_arr = col_sum_matrix[r2] - col_sum_matrix[r1 - 1]

                # Early pruning: skip if total sum is less than target
                if int(col_sums_arr.sum()) < target:
                    continue

                # Find subarrays that sum to target
                rectangles = RectangleSolver._find_subarrays_with_sum(
                    col_sums_arr.tolist(), target, r1, r2
                )
                found.extend(rectangles)

        return found

    @staticmethod
    def _find_subarrays_with_sum(
        arr: List[int], target: int, r1: int, r2: int
    ) -> List[Rectangle]:
        """Find all subarrays in arr that sum to target."""
        rectangles = []
        prefix_sum = 0
        seen: Dict[int, List[int]] = {0: [-1]}

        for i, value in enumerate(arr):
            prefix_sum += value
            needed = prefix_sum - target

            if needed in seen:
                for start_idx in seen[needed]:
                    rectangles.append(Rectangle(r1=r1, c1=start_idx + 1, r2=r2, c2=i))

            seen.setdefault(prefix_sum, []).append(i)

        return rectangles

    @staticmethod
    def sort_solutions(rectangles: List[Rectangle]) -> List[Rectangle]:
        """Sort rectangles by desirability (area, position, row)."""
        return sorted(rectangles, key=lambda r: (r.area(), r.top_left_position(), r.r1))


class SolverState:
    """Manages the application state."""

    def __init__(self):
        self.grid = np.zeros((GRID_ROWS, GRID_COLUMNS), dtype=np.int8)
        self.eliminated = np.zeros((GRID_ROWS, GRID_COLUMNS), dtype=bool)
        self.solutions: List[Rectangle] = []
        self.solution_index: int = -1
        self.current_highlight: Optional[Rectangle] = None
        self.status_line: str = ""

    def get_effective_grid(self) -> np.ndarray:
        """Get grid with eliminated cells zeroed out."""
        return np.where(self.eliminated, 0, self.grid).astype(np.int16, copy=False)

    def eliminate_rectangle(self, rect: Rectangle) -> None:
        """Mark cells in rectangle as eliminated."""
        self.eliminated[rect.r1 : rect.r2 + 1, rect.c1 : rect.c2 + 1] = True

    def print_grid(self) -> None:
        """Print the current grid to console."""
        print("Grid:")
        for row in range(GRID_ROWS):
            for col in range(GRID_COLUMNS):
                value = int(self.grid[row, col])
                print(value if value != 0 else ".", end="  ")
            print()


class OverlayController(QtCore.QObject):
    """Controller for communicating with the overlay."""

    state_changed = QtCore.pyqtSignal(object, int, int, str)


class Overlay(QtWidgets.QWidget):
    """Transparent overlay window for highlighting solutions."""

    def __init__(self):
        super().__init__()
        self._setup_window()
        self._setup_state()
        self._apply_windows_styles()
        self._setup_topmost_timer()

    def _setup_window(self) -> None:
        """Configure window properties."""
        self.setWindowTitle("AZX Overlay")
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.BypassWindowManagerHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        screen = QApplication.primaryScreen()
        geometry = screen.geometry() if screen else QtCore.QRect(0, 0, 1920, 1080)
        self.setGeometry(geometry)

        self.show()
        self.raise_()

        print(
            f"[overlay] Window: {geometry.x()},{geometry.y()} "
            f"{geometry.width()}x{geometry.height()}"
        )

    def _setup_state(self) -> None:
        """Initialize state variables."""
        self.highlight: Optional[Rectangle] = None
        self.solution_count: int = 0
        self.solution_index: int = -1
        self.status_line: str = ""
        self._hwnd: Optional[int] = None

    def _apply_windows_styles(self) -> None:
        """Apply Windows-specific styling for transparency."""
        try:
            self._hwnd = int(self.winId())
            ex_style = ctypes.windll.user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
            new_style = ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
            ctypes.windll.user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, new_style)
        except Exception as exc:
            print(f"[overlay] WinAPI style application failed: {exc}")

    def _setup_topmost_timer(self) -> None:
        """Setup timer to keep window on top."""
        self._topmost_timer = QtCore.QTimer(self)
        self._topmost_timer.timeout.connect(self._enforce_topmost)
        self._topmost_timer.start(300)

    def _enforce_topmost(self) -> None:
        """Ensure window stays on top of all others."""
        self.raise_()
        if self._hwnd is None:
            return

        try:
            ctypes.windll.user32.SetWindowPos(
                self._hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception:
            pass

    @QtCore.pyqtSlot(object, int, int, str)
    def set_state(
        self,
        highlight_obj: object,
        solution_count: int,
        solution_index: int,
        status_line: str,
    ) -> None:
        """Update overlay state and trigger repaint."""
        self.highlight = highlight_obj if isinstance(highlight_obj, Rectangle) else None
        self.solution_count = solution_count
        self.solution_index = solution_index
        self.status_line = status_line
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the overlay graphics."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        self._draw_status_indicator(painter)
        self._draw_status_text(painter)
        self._draw_highlight(painter)

    def _draw_status_indicator(self, painter: QtGui.QPainter) -> None:
        """Draw green dot indicator."""
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 255, 0, 220))
        painter.drawEllipse(10, 10, 10, 10)

    def _draw_status_text(self, painter: QtGui.QPainter) -> None:
        """Draw status text."""
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 220), 1))
        painter.setFont(QtGui.QFont("Consolas", 10))

        if self.solution_count > 0:
            idx = self.solution_index + 1 if self.solution_index >= 0 else 0
            text = (
                f"OVERLAY ON | {VERSION} | solutions: {self.solution_count} | "
                f"showing: {idx}/{self.solution_count}"
            )
        else:
            text = f"OVERLAY ON | {VERSION} | solutions: 0"

        painter.drawText(28, 20, text)

        if self.status_line:
            painter.drawText(28, 36, self.status_line)

    def _draw_highlight(self, painter: QtGui.QPainter) -> None:
        """Draw highlight rectangle."""
        if self.highlight is None:
            return

        x, y, width, height = self.highlight.to_pixels()

        # Semi-transparent fill
        painter.setBrush(QtGui.QColor(0, 255, 0, 55))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 255), 8))
        painter.drawRect(x, y, width, height)

        # Corner markers
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 255), 10))
        for corner_x, corner_y in [
            (x, y),
            (x + width, y),
            (x, y + height),
            (x + width, y + height),
        ]:
            painter.drawPoint(corner_x, corner_y)


class Application:
    """Main application coordinating all components."""

    def __init__(self):
        self.template_manager = TemplateManager()
        self.scanner = GridScanner(self.template_manager)
        self.solver = RectangleSolver()
        self.state = SolverState()
        self.controller = OverlayController()

    def initialize(self) -> None:
        """Initialize the application."""
        print(f"== {VERSION} ==")
        print(f"Template matching threshold: {MATCH_THRESHOLD}")
        self.template_manager.load_templates()
        self._print_help()

    def _print_help(self) -> None:
        """Print help information."""
        print("\nHotkeys:")
        print("  F5     - Scan grid + auto-highlight first solution")
        print("  F2     - Next solution (no elimination)")
        print("  SPACE  - Eliminate current solution, rebuild, show next")
        print("  F6     - Force test rectangle (0,0)-(0,1)")
        print("  F1     - Clear highlight")
        print("  ESC    - Quit")
        print("\nOptimizations:")
        print("  ✓ Single screenshot of entire grid")
        print("  ✓ Pre-loaded templates")
        print("  ✓ Early exit on high confidence matches")
        print("  ✓ NumPy grid + eliminated mask")
        print("  ✓ NumPy cumulative-sum rectangle solver\n")

    def scan_and_solve(self) -> None:
        """Scan the grid and find solutions."""
        self.state.eliminated.fill(False)
        self._reset_solutions()
        self._update_status("Scanning...")

        # Scan grid
        self.state.grid = self.scanner.scan_grid()
        self.state.print_grid()

        # Find solutions
        self._rebuild_solutions()

        elapsed = time.time()  # Simplified for this context
        self._update_status(
            f"Scan OK. SPACE = eliminate+next. F2 = next. F6 = test rect."
        )

    def show_next_solution(self) -> None:
        """Display the next solution."""
        if not self.state.solutions:
            self._rebuild_solutions()

        if not self.state.solutions:
            self.state.current_highlight = None
            self.state.solution_index = -1
            self._push_state()
            return

        self.state.solution_index = (self.state.solution_index + 1) % len(
            self.state.solutions
        )
        self.state.current_highlight = self.state.solutions[self.state.solution_index]
        self._print_solution_info()
        self._push_state()

    def eliminate_and_next(self) -> None:
        """Eliminate current solution and show next."""
        if self.state.current_highlight is not None:
            self.state.eliminate_rectangle(self.state.current_highlight)

        self._rebuild_solutions()

        if self.state.solutions and self.state.solution_index == 0:
            self.show_next_solution()

    def test_rectangle(self) -> None:
        """Display a test rectangle."""
        self.state.current_highlight = Rectangle(0, 0, 0, 1)
        self.state.solution_index = -1
        self._print_solution_info("TEST RECT")
        self._update_status("TEST RECT active (0,0)-(0,1)")
        self._push_state()

    def clear_highlight(self) -> None:
        """Clear the current highlight."""
        self.state.current_highlight = None
        self.state.solution_index = -1
        self._push_state()

    def _rebuild_solutions(self) -> None:
        """Rebuild solution list with current state."""
        effective_grid = self.state.get_effective_grid()
        self.state.solutions = self.solver.find_rectangles(effective_grid, TARGET_SUM)
        self.state.solutions = self.solver.sort_solutions(self.state.solutions)

        print(f"Found {len(self.state.solutions)} rectangles with sum == {TARGET_SUM}")

        if self.state.solutions:
            self.state.solution_index = 0
            self.state.current_highlight = self.state.solutions[0]
            self._print_solution_info("Auto-show #1")
        else:
            self.state.solution_index = -1
            self.state.current_highlight = None

        self._push_state()

    def _reset_solutions(self) -> None:
        """Reset solution state."""
        self.state.solutions = []
        self.state.solution_index = -1
        self.state.current_highlight = None
        self._push_state()

    def _print_solution_info(self, prefix: str = "Show") -> None:
        """Print information about current solution."""
        if self.state.current_highlight is None:
            return

        rect = self.state.current_highlight
        x, y, width, height = rect.to_pixels()
        index = self.state.solution_index + 1
        total = len(self.state.solutions)

        print(
            f"{prefix} #{index}/{total}: r1={rect.r1},c1={rect.c1},"
            f"r2={rect.r2},c2={rect.c2} | px=({x},{y}) size=({width}x{height})"
        )

    def _update_status(self, text: str) -> None:
        """Update status text."""
        self.state.status_line = text
        self._push_state()

    def _push_state(self) -> None:
        """Push state to overlay."""
        self.controller.state_changed.emit(
            self.state.current_highlight,
            len(self.state.solutions),
            self.state.solution_index,
            self.state.status_line,
        )

    def setup_hotkeys(self) -> None:
        """Register keyboard hotkeys."""
        keyboard.add_hotkey("f5", self.scan_and_solve)
        keyboard.add_hotkey("f2", self.show_next_solution)
        keyboard.add_hotkey("space", self.eliminate_and_next)
        keyboard.add_hotkey("f6", self.test_rectangle)
        keyboard.add_hotkey("f1", self.clear_highlight)
        keyboard.add_hotkey("esc", lambda: QtWidgets.QApplication.quit())


def main() -> None:
    """Main entry point."""
    qt_app = QApplication([])

    # Create and initialize application
    app = Application()
    app.initialize()

    # Create overlay
    overlay = Overlay()
    app.controller.state_changed.connect(overlay.set_state)

    # Initial state push
    app._push_state()

    # Setup hotkeys in separate thread
    def run_hotkeys():
        app.setup_hotkeys()
        keyboard.wait()

    logic_thread = threading.Thread(target=run_hotkeys, daemon=True)
    logic_thread.start()

    # Run Qt event loop
    qt_app.exec_()


if __name__ == "__main__":
    main()
