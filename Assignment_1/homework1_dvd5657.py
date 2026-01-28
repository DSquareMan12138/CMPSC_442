############################################################
# CMPSC/DS 442: Uninformed Search
############################################################

student_name = "Dylan Ding"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import math
import random


############################################################
# Section 1: N-Queens
############################################################

def num_placements_entire(n):
    """

    :param n: int
    :return: number of combination that I can put n queens into a nxn board
    """

    if not isinstance(n, int):
        return None

    return math.comb(n * n, n)

def num_placements_one_in_row(n):
    """

    :param n: int
    :return: number of combination that I can put n queens into a nxn board if only one queen is in one row
    """

    if not isinstance(n, int):
        return None

    return n ** n


def n_queens_valid(board):
    """

    :param board: list
    :return: True if queen are not attacking each other. False otherwise
    """

    if not isinstance(board, list):
        return None

    # check column
    if len(board) != len(set(board)):
        return False

    # check diagonal
    for i in range(len(board) - 1):
        diagonal_calculation = 1

        for j in range(i+1, len(board)):

            if board[j] == board[i] + diagonal_calculation or board[j] == board[i] - diagonal_calculation:
                return False

            diagonal_calculation += 1

    return True



def n_queens_solutions(n):
    """

    :param n:
    :return: no return it is a generator
    """

    def n_queens_helper(n, board):

        # base case
        if len(board) == n:
            yield board.copy()
            return

        # add in stack if valid
        for i in range(n):
            board.append(i)
            if n_queens_valid(board):
                # freeze the state and delegate to a child branch
                yield from n_queens_helper(n, board)
            # pop if invalid
            board.pop()
    yield from n_queens_helper(n, [])



############################################################
# Section 2: Lights Out
############################################################

class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board.deepcopy()

        # dimension of the board
        self.row = len(board)
        self.col = len(board[0])

    def get_board(self):
        return self.board

    def perform_move(self, row, col):
        self.board[row][col] = not self.board[row][col]

        # neighbors
        if row + 1 < self.row:
            self.board[row+1][col] = not self.board[row+1][col]
        if row - 1 >= 0:
            self.board[row-1][col] = not self.board[row-1][col]
        if col + 1 < self.col:
            self.board[row][col+1] = not self.board[row][col+1]
        if col - 1 >= 0:
            self.board[row][col-1] = not self.board[row][col-1]


    def scramble(self):

        for row in range(self.row):
            for col in range(self.col):

                if random.random() < 0.5:
                    self.perform_move(row, col)


    def is_solved(self):
        pass

    def copy(self):
        pass

    def successors(self):
        pass

    def find_solution(self):
        pass

def make_puzzle(rows, cols):
    puzzle = []
    for row in range(rows):
        puzzle.append([False] * cols)
    return puzzle

############################################################
# Section 3: Linear Disk Movement
############################################################

def solve_identical_disks(length, n):
    pass

def solve_distinct_disks(length, n):
    pass

