import random

############################################################
# CMPSC/DS 442: Homework 3 
############################################################

student_name = "Dylan Ding"

############################################################
import copy
from collections import deque
############################################################

# Include your imports here, if any are used.


############################################################
# Section 1: Dominoes Game
############################################################

def make_dominoes_game(rows, cols):
    """
    Create a dominoes board

    :param rows: The number of rows for the board
    :type rows: int
    :param cols: The number of columns for the board
    :type cols: int
    :return: A new instance of `DominoesGame` initialized with an empty board
    :rtype: DominoesGame
    """
    board = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(False)
        board.append(row)
    return DominoesGame(board)

class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

    def get_board(self):
        return self.board

    def reset(self):
        new_board = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(False)
            new_board.append(row)
        self.board = new_board

    def is_legal_move(self, row, col, vertical):
        if vertical:
            if row + 1 >= self.rows:
                return False
            if self.board[row][col]:
                return False
            if self.board[row + 1][col]:
                return False
            return True
        else:
            if col + 1 >= self.cols:
                return False
            if self.board[row][col]:
                return False
            if self.board[row][col + 1]:
                return False
            return True


    def legal_moves(self, vertical):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.is_legal_move(row, col, vertical):
                    yield row, col

    def execute_move(self, row, col, vertical):
        if vertical:
            self.board[row][col] = True
            self.board[row + 1][col] = True
        else:
            self.board[row][col] = True
            self.board[row][col + 1] = True

    def game_over(self, vertical):
        for move in self.legal_moves(vertical):
            return False
        return True

    def copy(self):
        new_board = []

        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(self.board[r][c])
            new_board.append(row)

        return DominoesGame(new_board)

    def successors(self, vertical):
        for move in self.legal_moves(vertical):
            new_game = self.copy()
            row, col = move
            new_game.execute_move(row, col, vertical)
            yield move, new_game

    def get_random_move(self, vertical):
        moves = []
        for move in self.legal_moves(vertical):
            moves.append(move)

        if len(moves) == 0:
            return None

        return random.choice(moves)

    def evaluate(self, vertical):
        return (len(list(self.legal_moves(vertical))) -
                len(list(self.legal_moves(not vertical))))

    def get_best_move(self, limit, vertical):

        def alphabeta(game, depth, alpha, beta, maximizing):
            if depth == 0 or game.game_over(vertical if maximizing else not vertical):
                return game.evaluate(vertical), None, 1

            total_leaves = 0

            if maximizing:
                best_val = float("-inf")
                best_move = None

                for move, succ in game.successors(vertical):
                    val, _, leaves = alphabeta(succ, depth - 1, alpha, beta, False)
                    total_leaves += leaves

                    if val > best_val:
                        best_val = val
                        best_move = move

                    alpha = max(alpha, best_val)
                    if alpha >= beta:
                        break

                return best_val, best_move, total_leaves

            else:
                best_val = float("inf")
                best_move = None

                for move, succ in game.successors(not vertical):
                    val, _, leaves = alphabeta(succ, depth - 1, alpha, beta, True)
                    total_leaves += leaves

                    if val < best_val:
                        best_val = val
                        best_move = move

                    beta = min(beta, best_val)
                    if alpha >= beta:
                        break

                return best_val, best_move, total_leaves

        value, move, leaves = alphabeta(self, limit, float("-inf"), float("inf"), True)
        return move, value, leaves


############################################################
# Section 2: Sudoku
############################################################

def sudoku_cells():
    cells = []

    for row in range(9):
        for col in range(9):
            cells.append((row, col))

    return cells

def sudoku_arcs():
    arcs = []
    for r in range(9):
        for c in range(9):
            cell1 = (r, c)

            for i in range(9):
                if i != c:
                    arcs.append((cell1, (r, i)))
                if i != r:
                    arcs.append((cell1, (i, c)))

            br, bc = (r // 3) * 3, (c // 3) * 3
            for i in range(br, br + 3):
                for j in range(bc, bc + 3):
                    if (i, j) != cell1:
                        arcs.append((cell1, (i, j)))

    return arcs

def read_board(path):
    board = {}

    with open(path, "r") as f:
        row = 0
        for line in f:
            line = line.strip()
            for col in range(9):
                ch = line[col]
                cell = (row, col)

                if ch == "*":
                    board[cell] = set()
                    for value in range(1, 10):
                        board[cell].add(value)
                else:
                    board[cell] = {int(ch)}
            row += 1

    return board

class Sudoku(object):

    CELLS = sudoku_cells()
    ARCS = sudoku_arcs()

    def __init__(self, board):
        self.board = {}
        for cell in board:
            self.board[cell] = set(board[cell])

    def get_values(self, cell):
        return self.board[cell]

    def remove_inconsistent_values(self, cell1, cell2):
        removed = False

        # For Sudoku inequality constraints, a value v in cell1 is inconsistent
        # only if cell2 is forced to be exactly {v}.
        if len(self.board[cell2]) == 1:
            value2 = next(iter(self.board[cell2]))
            if value2 in self.board[cell1] and len(self.board[cell1]) > 1:
                self.board[cell1].remove(value2)
                removed = True

        return removed

    def infer_ac3(self):
        queue = deque(self.ARCS)

        while queue:
            cell1, cell2 = queue.popleft()

            if self.remove_inconsistent_values(cell1, cell2):
                if len(self.board[cell1]) == 0:
                    return False

                for x, y in self.ARCS:
                    if y == cell1 and x != cell2:
                        queue.append((x, cell1))

        return True

    def infer_improved(self):
        changed = True

        while changed:
            changed = False

            before = {}
            for cell in self.CELLS:
                before[cell] = set(self.board[cell])

            if not self.infer_ac3():
                return False

            # hidden singles in rows
            for row in range(9):
                for value in range(1, 10):
                    places = []
                    for col in range(9):
                        cell = (row, col)
                        if value in self.board[cell]:
                            places.append(cell)
                    if len(places) == 1 and len(self.board[places[0]]) > 1:
                        self.board[places[0]] = {value}

            # hidden singles in columns
            for col in range(9):
                for value in range(1, 10):
                    places = []
                    for row in range(9):
                        cell = (row, col)
                        if value in self.board[cell]:
                            places.append(cell)
                    if len(places) == 1 and len(self.board[places[0]]) > 1:
                        self.board[places[0]] = {value}

            # hidden singles in 3x3 blocks
            for block_row in range(0, 9, 3):
                for block_col in range(0, 9, 3):
                    for value in range(1, 10):
                        places = []
                        for row in range(block_row, block_row + 3):
                            for col in range(block_col, block_col + 3):
                                cell = (row, col)
                                if value in self.board[cell]:
                                    places.append(cell)
                        if len(places) == 1 and len(self.board[places[0]]) > 1:
                            self.board[places[0]] = {value}

            for cell in self.CELLS:
                if len(self.board[cell]) == 0:
                    return False

            for cell in self.CELLS:
                if self.board[cell] != before[cell]:
                    changed = True
                    break

        return True

    def is_valid_solution(self):
        # check rows
        for row in range(9):
            values = []
            for col in range(9):
                cell = (row, col)
                if len(self.board[cell]) != 1:
                    return False
                values.append(next(iter(self.board[cell])))
            if len(set(values)) != 9:
                return False

        # check columns
        for col in range(9):
            values = []
            for row in range(9):
                cell = (row, col)
                if len(self.board[cell]) != 1:
                    return False
                values.append(next(iter(self.board[cell])))
            if len(set(values)) != 9:
                return False

        # check 3x3 blocks
        for start_row in range(0, 9, 3):
            for start_col in range(0, 9, 3):
                values = []
                for row in range(start_row, start_row + 3):
                    for col in range(start_col, start_col + 3):
                        cell = (row, col)
                        if len(self.board[cell]) != 1:
                            return False
                        values.append(next(iter(self.board[cell])))
                if len(set(values)) != 9:
                    return False

        return True

    def infer_with_guessing(self):
        if not self.infer_improved():
            return False

        # contradiction: empty domain
        for cell in self.CELLS:
            if len(self.board[cell]) == 0:
                return False

        # if all singletons, only succeed if the whole board is valid
        all_singletons = True
        for cell in self.CELLS:
            if len(self.board[cell]) != 1:
                all_singletons = False
                break

        if all_singletons:
            return self.is_valid_solution()

        # choose cell with fewest possibilities > 1
        guess_cell = None
        guess_size = None
        for cell in self.CELLS:
            size = len(self.board[cell])
            if size > 1:
                if guess_cell is None or size < guess_size:
                    guess_cell = cell
                    guess_size = size

        # try guesses with backtracking
        for guess in list(self.board[guess_cell]):
            new_board = {}
            for cell in self.CELLS:
                new_board[cell] = set(self.board[cell])

            new_board[guess_cell] = {guess}
            new_sudoku = Sudoku(new_board)

            if new_sudoku.infer_with_guessing():
                for cell in self.CELLS:
                    self.board[cell] = set(new_sudoku.board[cell])
                return True

        return False