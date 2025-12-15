import keyboard
import mss
import numpy as np
import cv2
import threading
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import ctypes
from PyQt5.QtWidgets import QApplication

class Overlay(QtWidgets.QWidget):

    def __init__(self, rows, cols, left_start, top_start, offset_x, offset_y, w, h):
        super().__init__()

        self.rows = rows
        self.cols = cols

        self.left_start = left_start
        self.top_start = top_start
        self.offset_x = offset_x
        self.offset_y = offset_y

        self.cell_w = w
        self.cell_h = h

        self.cells = []

        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.setGeometry(0, 0, 1920, 1080)

        self.show()

        #Click-through
        hwnd = self.winId().__int__()
        ctypes.windll.user32.SetWindowLongW(
            hwnd, -20,
            ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            | 0x80000  #WS_EX_LAYERED
            | 0x20     #WS_EX_TRANSPARENT
        )

    def set_cells(self, cell_list):
        self.cells = cell_list
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        for (r, c, color) in self.cells:
            r_px = self.top_start + r * self.offset_y
            c_px = self.left_start + c * self.offset_x

            brush = QtGui.QColor(*color)
            painter.setBrush(brush)
            painter.setPen(QtGui.QPen(QtGui.QColor(255,255,255,180), 2))

            painter.drawRect(c_px, r_px, self.cell_w, self.cell_h)

#rows and columns from the level

rows = 16; columns = 10

#pixels

offset_x = 51; offset_y = 52
top_start = 221; left_start = 708 
capture_area_w = 44; capture_area_h = 45

numbers = []
matrix = []

start_area = {
    "top": top_start,
    "left": left_start,
    "width": capture_area_w,
    "height": capture_area_h
    }

def get_matrix_numbers():

    counter = 0

    with mss.mss() as sct:
        for r in range(rows):
            for c in range(columns):
                counter += 1

                image = sct.grab(start_area)

                img_np = np.array(image)
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

                img_np = np.array(image)

                best_score = -10 
                best_match_digit = -1

                for j in range(1, 10):
                    template_path = f"./templates/T{j}.png"
                    template = cv2.imread(template_path)

                    if template is None:
                        print(f"Couldn't load template: {template_path}")
                        continue

                    temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                    res = cv2.matchTemplate(img_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    current_score = max_val

                    if current_score > best_score:
                        best_score = current_score
                        best_match_digit = j
                
                if best_match_digit != -1:
                    numbers.append(best_match_digit)
                else:
                    print("Not valid match, replacing with whitespace.")
                    numbers.append(" ")

                start_area["left"] += offset_x

            start_area["left"] = left_start
            start_area["top"] += offset_y

    createMatrix()

def createMatrix():

    pos = 0

    for i in range(rows):
        row = []
        for j in range(columns):
            row.append(numbers[pos])
            pos += 1
        matrix.append(row)

    printMatrix()

def printMatrix():
    print("Matrix: ")

    for i in range(rows):
        for j in range(columns):
            print(matrix[i][j], end="  ")
        print()

def checkRight(columns, r, c, matrix):
    sum = 0
    for j in range(columns - c):

        if c + j > columns:
            return False, 0

        if (has_special_char(r, c + j)):
            continue
        else:
            sum += matrix[r][c + j]

        if sum > 10:
            return False, 0
        elif sum == 10:
            return True, j
        
    return False, 0

def sums_right():

    for r in range(rows):
        for c in range(columns):
            
            if has_special_char(r,c):
                continue

            validSum, positions = checkRight(columns, r, c, matrix)

            if validSum:
                for x in range(1, positions + 1):
                    matrix[r][c] = "\u25BA"
                    matrix[r][c + x] = "\u2192"

    printMatrix()
    update_overlay()

def checkDown(rows, r, c, matrix):
    sum = 0

    for j in range(rows - r):

        if r + j > rows:
            return False, 0

        if has_special_char(r + j, c):
            continue
        else:
            sum += matrix[r + j][c]

        if sum > 10:
            return False, 0
        elif sum == 10:
            return True, j
        
    return False, 0

def sums_down():
    positions = 0
    for c in range(columns):
        for r in range(rows):
            
            if has_special_char(r,c):
                continue

            validSum, positions = checkDown(rows, r, c, matrix)

            if validSum:
                for x in range(1, positions + 1):
                    matrix[r][c] = "\u25BC"
                    matrix[x + r][c] = "\u2193"
    
    printMatrix()
    update_overlay()

def checkSquareUp(rows, columns, start_r, start_c, matrix):

    if start_r <= 0 or start_c >= columns:
        return False, 0, 0

    if has_special_char(start_r, start_c):
        return False, 0, 0

    edge_rows = start_r - 1 
    edge_columns = start_c + 1
    area = 1

    while edge_rows >= 0 and edge_columns < columns:

        sum = 0

        for r in range(edge_rows, start_r + 1):
            for c in range(start_c, edge_columns + 1):
                if not has_special_char(r, c):
                    sum += int(matrix[r][c])

        if sum > 10:
            return False, 0, 0

        if sum == 10:
            return True, edge_rows, edge_columns

        if edge_columns >= columns or edge_rows < 0:
             return False, 0, 0

        current_col = edge_columns
        sum_col = sum

        while current_col + 1 < columns:
            current_col += 1

            for r in range(edge_rows, start_r + 1):
                if not has_special_char(r, current_col):
                    sum_col += int(matrix[r][current_col])

            if sum_col > 10:
                break

            if sum_col == 10:
                return True, edge_rows, current_col

        current_row = edge_rows
        sum_row = sum

        while current_row - 1 >= 0:
            current_row -= 1

            for c in range(start_c, edge_columns + 1):
                if not has_special_char(current_row, c):
                    sum_row += int(matrix[current_row][c])

            if sum_row > 10:
                break

            if sum_row == 10:
                return True, current_row, edge_columns

        edge_rows -= 1
        edge_columns += 1
        area += 1

    return False, 0, 0

def checkSquareDown(rows, columns, start_r, start_c, matrix):

    if start_r >= rows or start_c >= columns:
        return False, 0, 0

    if has_special_char(start_r, start_c):
        return False, 0, 0

    edge_rows = start_r + 1; edge_columns = start_c + 1
    area = 1

    while edge_rows < rows and edge_columns < columns:

        sum = 0

        for r in range(start_r, edge_rows + 1):
            for c in range(start_c, edge_columns + 1):
                if not has_special_char(r, c):
                    sum += int(matrix[r][c])

        if sum > 10:
            return False, 0, 0

        if sum == 10:
            return True, edge_rows, edge_columns


        if edge_columns > columns or edge_rows > rows:
            return False, 0, 0

        current_col = edge_columns
        sum_col = sum

        while current_col + 1 < columns:
            current_col += 1

            for r in range(start_r, edge_rows + 1):
                if not has_special_char(r, current_col):
                    sum_col += int(matrix[r][current_col])

            if sum_col > 10:
                break

            if sum_col == 10:
                return True, edge_rows, current_col

        current_row = edge_rows
        sum_row = sum

        while current_row + 1 < rows:
            current_row += 1

            for c in range(start_c, edge_columns + 1):
                if not has_special_char(current_row, c):
                    sum_row += int(matrix[current_row][c])

            if sum_row > 10:
                break

            if sum_row == 10:
                return True, current_row, edge_columns

        edge_rows += 1
        edge_columns += 1
        area += 1

    return False, 0, 0

def sums_square():
    for r in range(rows):
        for c in range(columns):

            if has_special_char(r, c):
                continue
            
            validSumDown, max_r, max_c = checkSquareDown(rows, columns, r, c, matrix)
            
            if validSumDown:
                for j in range(r, max_r + 1):
                    for k in range(c, max_c + 1):
                        matrix[j][k] = "\u25A0"
                matrix[r][c] = "\u25A1"
            else:
                validSumUp, max_r, max_c = checkSquareUp(rows, columns, r, c, matrix)
                if validSumUp:
                    for j in range(r, max_r - 1, -1):
                        for k in range(c, max_c + 1):
                            matrix[j][k] = "\u25A0"
                    matrix[r][c] = "\u25A1"

    printMatrix()
    update_overlay()

def has_special_char(r, c):
    if matrix[r][c] == "\u2192" or matrix[r][c] == "\u2193" or matrix[r][c] == "\u25A0" or matrix[r][c] == "\u25A1" or matrix[r][c] == "\u25BA" or matrix[r][c] == "\u25BC" or matrix[r][c] == " ":
        return True
    return False

def clean_matrix():
    for r in range(rows):
        for c in range(columns):

            if has_special_char(r, c):
                matrix[r][c] = " "
            else:
                continue

    print("Cleaned special characters from matrix: ")
    printMatrix()
    update_overlay()

def update_overlay():
    cell_list = []

    for r in range(rows):
        for c in range(columns):

            value = matrix[r][c]

            if value == "\u25A0":
                cell_list.append((r, c, (255, 0, 0, 70)))
            elif value == "\u25A1":
                cell_list.append((r, c, (255, 0, 0, 200)))
            elif value =="\u2192":
                cell_list.append((r, c, (0, 255, 0, 70)))
            elif value =="\u2193":
                cell_list.append((r, c, (0, 0, 255, 70)))
            elif value =="\u25BA":
                cell_list.append((r, c, (0, 255, 0, 200)))
            elif value =="\u25BC":
                cell_list.append((r, c, (0, 0, 255, 200)))
            elif value ==" ":
                cell_list.append((r, c, (0, 0, 0, 0)))

    overlay.set_cells(cell_list)


def start_logic():
    print("f1. clean matrix")
    print("f2. sums right")
    print("f3. sums down")
    print("f4. sums square")
    print("f5. scan matrix")

    keyboard.add_hotkey('f5', get_matrix_numbers)
    keyboard.add_hotkey('f3', sums_down)
    keyboard.add_hotkey('f2', sums_right)
    keyboard.add_hotkey('f4', sums_square)
    keyboard.add_hotkey('f1', clean_matrix)

    keyboard.add_hotkey("esc", lambda: QtWidgets.QApplication.quit())
    
    keyboard.wait()

if __name__ == "__main__":
    app = QApplication([])

    overlay = Overlay(
        rows, columns,
        left_start, top_start,
        offset_x, offset_y,
        capture_area_w, capture_area_h
    )

    logic_thread = threading.Thread(target=start_logic, daemon=True)
    logic_thread.start()

    app.exec_()