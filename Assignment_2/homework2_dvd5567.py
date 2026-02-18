############################################################
# CMPSC/DS 442: Informed Search
############################################################

student_name = "Dylan Ding"

############################################################
# Imports
############################################################

from heapq import heappush, heappop
import random
import math


############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    """
    :param rows: int
    :param cols: int

    Create a tile puzzle with the given number of rows and columns.
    The puzzle is initialized with tiles in row-major order, with the blank tile (0) at the top left corner.
    """
    board = []
    val = 0

    # 2d array
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(val)
            val += 1
        board.append(row)
    return TilePuzzle(board)


class TilePuzzle(object):
    """
    Represents a tile puzzle with a 2D board and a blank tile.
    """
    # Required
    def __init__(self, board):

        self.board = [row[:] for row in board]

        self.rows = len(self.board)
        self.cols = len(self.board[0]) if self.rows > 0 else 0

        self.blank_position = None
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == 0:
                    self.blank_position = (r, c)
                    return

    def get_board(self):
        return [row[:] for row in self.board]

    def perform_move(self, direction):
        """

        :param direction: "up", "down", "left", "right"
        :return: True if the move was successful, False otherwise
        """
        moves = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        if direction not in moves or self.blank_position is None:
            return False

        blank_row, blank_col = self.blank_position
        delta_r, delta_c = moves[direction]
        new_row, new_col = blank_row + delta_r, blank_col + delta_c

        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            self.board[blank_row][blank_col], self.board[new_row][new_col] = self.board[new_row][new_col], self.board[blank_row][blank_col]
            self.blank_position = (new_row, new_col)
            return True
        return False

    def scramble(self, num_moves):
        """

        :param num_moves: Number of random moves to scramble the puzzle
        :return:
        """
        directions = ["up", "down", "left", "right"]
        for _ in range(num_moves):
            self.perform_move(random.choice(directions))

    def is_solved(self):
        expected = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] != expected:
                    return False
                expected += 1
        return True

    def copy(self):
        return TilePuzzle(self.get_board())

    def successors(self):
        """
        make it a spg
        :return:
        """
        for direction in ["up", "down", "left", "right"]:
            p = self.copy()
            if p.perform_move(direction):
                yield (direction, p)

    # Required
    def get_solutions_iddfs(self):
        """
        Iterative Deepening DFS that returns all shortest solutions.
        Uses path cycle checking + depth pruning to avoid infinite search.
        """
        depth = 0
        start_state = self._state_tuple()

        while True:
            found_solution = False

            # best remaining depth seen in this iteration
            depth_seen = {start_state: depth}

            for sol in self._iddfs_helper(
                remaining_depth=depth,
                path=[],
                path_set={start_state},
                depth_seen=depth_seen
            ):
                found_solution = True
                yield sol

            if found_solution:
                return

            depth += 1


    def _iddfs_helper(self, remaining_depth, path, path_set, depth_seen):
        """
        Depth-limited DFS used by IDDFS
        """

        if self.is_solved():
            yield path
            return

        if remaining_depth == 0:
            return

        for move, next_puzzle in self.successors():

            state_key = next_puzzle._state_tuple()

            # prevent undo loops (cycle in current path)
            if state_key in path_set:
                continue

            prev_depth = depth_seen.get(state_key)
            if prev_depth is not None and prev_depth >= remaining_depth - 1:
                continue

            depth_seen[state_key] = remaining_depth - 1

            path_set.add(state_key)
            yield from next_puzzle._iddfs_helper(
                remaining_depth - 1,
                path + [move],
                path_set,
                depth_seen
            )
            path_set.remove(state_key)


    def _state_tuple(self):
        return tuple(tuple(row) for row in self.board)


    # Required
    def get_solution_A_star(self):
        """
        use astar to find the optimal solution
        :return:
        """
        def manhattan(state):
            dist = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    val = state[r][c]
                    if val == 0:
                        continue
                    goal_row, goal_col = divmod(val, self.cols)
                    dist += abs(goal_row - r) + abs(goal_col - c)
            return dist

        priority_q = []
        start = self._state_tuple()

        heappush(priority_q, (manhattan(start), 0, start, []))
        visited = set()

        while priority_q:
            f, g, state, path = heappop(priority_q)
            if state in visited:
                continue
            visited.add(state)

            # when found
            if self._is_solved_state(state):
                return path

            puzzle = TilePuzzle([list(row) for row in state])
            for move, nxt in puzzle.successors():
                nxt_state = nxt._state_tuple()
                if nxt_state in visited:
                    continue
                new_g = g + 1
                heappush(priority_q, (new_g + manhattan(nxt_state), new_g, nxt_state, path + [move]))

        return []

    def _state_tuple(self):
        """"""
        return tuple(tuple(row) for row in self.board)

    def _is_solved_state(self, state):
        """
        return true if the state is solved
        :param state:
        :return:
        """
        expected = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if state[r][c] != expected:
                    return False
                expected += 1
        return True


############################################################
# Section 2: Grid Navigation
############################################################

def get_neighbors(node, scene):
    """
    return all neighbors of a node
    :param node: tuple
    :param scene: matrix
    :return:
    """
    row, col = node
    height, width = len(scene), len(scene[0])
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    neighbors = []
    for delta_row, delta_col in directions:
        new_row, new_col = row + delta_row, col + delta_col
        if 0 <= new_row < height and 0 <= new_col < width and not scene[new_row][new_col]:
            neighbors.append((new_row, new_col))
    return neighbors


def find_shortest_path(start, goal, scene):
    """
    find shortest path from start to goal
    :param start: tuple
    :param goal: tuple
    :param scene: matrix
    :return: list of tuples
    """
    if scene[start[0]][start[1]] or scene[goal[0]][goal[1]]:
        return None

    def heu(a, b):
        return math.dist(a, b)


    priority_q = []
    heappush(priority_q, (heu(start, goal), 0.0, start, [start]))
    visited = set()

    while priority_q:
        f, g, node, path = heappop(priority_q)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path

        for neighbor in get_neighbors(node, scene):
            if neighbor in visited:
                continue
            step = math.dist(node, neighbor)  # 1 (cardinal) or sqrt(2) (diagonal)
            new_g = g + step
            heappush(priority_q, (new_g + heu(neighbor, goal), new_g, neighbor, path + [neighbor]))

    return None


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

def heuristic(state, goal):

    goal_pos = {v: i for i, v in enumerate(goal)}
    heu_val = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        d = abs(i - goal_pos[v])
        heu_val += math.ceil(d / 2)
    return heu_val


def solve_distinct_disks_v2(length, n):
    start = tuple(list(range(1, n + 1)) + [0] * (length - n))
    goal = tuple([0] * (length - n) + list(range(n, 0, -1)))

    def neighbors(state):
        s = list(state)
        for i, v in enumerate(s):
            if v == 0:
                continue

            # move 1 step
            for d in (-1, 1):
                j = i + d
                if 0 <= j < length and s[j] == 0:
                    t = s[:]
                    t[i], t[j] = t[j], t[i]
                    yield (i, j), tuple(t)

            # jump 2 steps over a disk
            for d in (-2, 2):
                j = i + d
                mid = i + (d // 2)
                if length > j >= 0 == s[j] and s[mid] != 0:
                    t = s[:]
                    t[i], t[j] = t[j], t[i]
                    yield (i, j), tuple(t)

    priority_q = []
    heappush(priority_q, (heuristic(start, goal), 0, start, []))
    visited = set()

    while priority_q:
        f, g, state, path = heappop(priority_q)
        if state in visited:
            continue
        visited.add(state)

        if state == goal:
            return path

        for move, nxt in neighbors(state):
            if nxt in visited:
                continue
            new_g = g + 1
            heappush(priority_q, (new_g + heuristic(nxt, goal), new_g, nxt, path + [move]))

    return []
############################################################
