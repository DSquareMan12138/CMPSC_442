############################################################
# CMPSC/DS 442: Uninformed Search
############################################################
from collections import deque

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
        self.board = [row[:] for row in board]

        # dimension of the board
        self.row = len(self.board)
        self.col = len(self.board[0])

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
        """
        scramble the board
        """

        for row in range(self.row):
            for col in range(self.col):

                if random.random() < 0.5:
                    self.perform_move(row, col)


    def is_solved(self):
        """
        :return: Boolean indicating if the puzzle is solved
        """
        for row in self.board:
            if any(row):
                return False
        return True

    def copy(self):
        return LightsOutPuzzle([row[:] for row in self.board])

    def successors(self):
        """
        give all possible valid move and its outcome
        """
        for row in range(self.row):
            for col in range(self.col):
                new_board = self.copy()
                new_board.perform_move(row, col)
                yield (row, col), new_board

    def find_solution(self):
        """
        use BFS to find the optimal (shortest) solution path
        :return: the solution path

        """
        # check if it is solved
        if self.is_solved():
            return []

        def board_to_tuple(board_obj):
            return tuple(tuple(row) for row in board_obj.get_board())

        # make the board operable
        start_board = board_to_tuple(self)

        # make a queue for BFS
        q = deque()
        q.append((self.copy(), []))

        # make a visited memory set
        visited = {start_board}

        while q:
            # get current state
            board, path = q.popleft()

            if board.is_solved():
                return path

            for move, outcome_board in board.successors():
                state = board_to_tuple(outcome_board)

                if state in visited:
                    continue

                visited.add(state)
                q.append((outcome_board, path + [move]))

        return None

def make_puzzle(rows, cols):
    puzzle = []
    for row in range(rows):
        puzzle.append([False] * cols)
    return LightsOutPuzzle(puzzle)

############################################################
# Section 3: Linear Disk Movement
############################################################

def solve_identical_disks(length, n):

    disk = (1,) * n + (0,) * (length - n)
    success_case = (0,) * (length - n) + (1,) * n

    def swap(state, old_index, new_index):
        s = list(state)
        s[old_index], s[new_index] = s[new_index], s[old_index]
        return tuple(s)

    def move(state):
        for i, value in enumerate(state):
            if value != 1:
                continue

            # move right for distance == 1
            j = i + 1
            if j < length and state[j] == 0:
                yield (i,j), swap(state, i, j)


            # move right for distance == 2
            j = j + 2
            if j < length and state[j] == 0 and state[i + 1] == 0:
                yield (i,j), swap(state, i, j)

        q = deque([(disk, [])])
        visited = {disk}

        while q:
            state, path = q.popleft()

            # success case
            if state == success_case:
                return path

            for new_move, new_state in move(state):
                if new_state not in visited:
                    visited.add(new_state)
                    q.append((new_state, path + [new_move]))

        return None




    return

def solve_distinct_disks(length, n):

    disk = tuple(range(1, n + 1)) + (0,) * (length - n)
    success_case = (0,) * (length - n) + tuple(range(n, 0, -1))

    def swap(state, old_index, new_index):
        s = list(state)
        s[old_index], s[new_index] = s[new_index], s[old_index]
        return tuple(s)

    def move(state):
        for i, value in enumerate(state):
            if value == 0:
                continue

            # move left or right for distance == 1
            for direction in (-1, 1):
                j = i + direction
                if length > j >= 0 == state[j]:
                    yield (i, j), swap(state, i, j)


            # move left or right for distance == 2
            for direction in (-2, 2):
                j = i + direction
                mid = i + (direction - 1)
                if length > j >= 0 == state[j] and state[mid] != 0:
                    yield (i, j), swap(state, i, j)

        q = deque([(disk, [])])
        visited = {disk}

        while q:
            state, path = q.popleft()

            # success case
            if state == success_case:
                return path

            for new_move, new_state in move(state):
                if new_state not in visited:
                    visited.add(new_state)
                    q.append((new_state, path + [new_move]))

        return None

