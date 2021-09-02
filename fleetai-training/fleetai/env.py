from typing import Any

import numpy as np
import gym

class _Ship(object):
    def __init__(self, size, row, col, dr, dc):
        self.size = size
        self.row = row
        self.col = col
        self.dr = dr
        self.dc = dc


class BattleshipEnv(gym.Env):
    def __init__(self, observation_space="flat-ships", action_space="coords", board_width=10, board_height=10,
                 ships=(1, 2, 3, 4, 5), latent_var_precision=8):
        self.board = np.zeros((board_height, board_width), dtype=bool)
        self.ship_lens = sorted(ships)
        self.shots = np.zeros_like(self.board, dtype=bool)
        self.observation_space_type = observation_space
        self.action_space_type = action_space
        self.latent_var_precision = latent_var_precision
        self.window = None
        self.ships = None

        if action_space == "coords":
            self.action_space = gym.spaces.MultiDiscrete((board_height, board_width))
        elif action_space == "flat":
            self.action_space = gym.spaces.Discrete(board_height * board_width)
        else:
            raise Exception(f"Unrecognized action space {action_space}")

        allowed_parts = {"flat", "ships", "latent"}
        parts = observation_space.split("-")
        assert len(set(parts)) == len(parts), "No repeated parts in observation space!"
        assert all(p in allowed_parts for p in parts), "Unrecognized part in " + observation_space
        parts = set(parts)
        state_dim = 0
        if "flat" in parts:
            state_dim += 2 * board_height * board_width
        if "ships" in parts:
            state_dim += len(ships)
        if "latent" in parts:
            state_dim += latent_var_precision
        self.observation_space = gym.spaces.MultiBinary(state_dim)
        self.reset()

    def _is_sunk(self, ship: _Ship):
        row, col = ship.row, ship.col
        return all(self.shots[row + i * ship.dr, col + i * ship.dc] for i in range(ship.size))

    def _observe(self):
        hits = (self.board & self.shots).astype(np.int8)
        misses = (~self.board & self.shots).astype(np.int8)
        obs = np.empty(0, dtype=np.int8)
        if "flat" in self.observation_space_type:
            misses_hits = np.hstack((misses.flatten(), hits.flatten()))
            obs = np.hstack((obs, misses_hits))
        if "ships" in self.observation_space_type:
            ship_obs = np.array([self._is_sunk(ship) for ship in self.ships], dtype=np.int8)
            obs = np.hstack((obs, ship_obs))
        if "latent" in self.observation_space_type:
            bits = (np.random.random(self.latent_var_precision) >= 0.5).astype(np.int8)
            obs = np.hstack((obs, bits))
        return obs

    def _done(self):
        return (self.board == (self.board & self.shots)).all()

    def step(self, action: Any):
        if self.action_space_type == "coords":
            row, col = action
        elif self.action_space_type == "flat":
            row = action // self.board.shape[0]
            col = action % self.board.shape[1]
        else:
            raise AssertionError
        if self.shots[row, col]:
            reward = -1
        else:
            reward = 1 if self.board[row, col] else -0.15
            self.shots[row, col] = True
        return self._observe(), reward, self._done(), {}

    def can_move(self, action: Any):
        if self.action_space_type == "coords":
            row, col = action
        elif self.action_space_type == "flat":
            row = action // self.board.shape[0]
            col = action % self.board.shape[1]
        else:
            raise AssertionError
        return not self.shots[row, col]

    def _render_graphics(self, width, height):
        from graphics import GraphWin, Rectangle, Point, Circle
        if self.window is None:
            self.window = GraphWin(width=width, height=height, autoflush=False)
            self.window.setCoords(0, height, width, 0)
        sq_height = min(width, height) // self.board.shape[0]
        sq_width = min(width, height) // self.board.shape[1]
        if width <= height:
            x_start = 0
            y_start = (height - width) // 2
        else:
            x_start = (width - height) // 2
            y_start = 0
        for item in self.window.items[:]:
            item.undraw()
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                p1 = Point(x_start + col * sq_width, y_start + row * sq_height)
                p2 = Point(x_start + (col + 1) * sq_width, y_start + (row + 1) * sq_height)
                rect = Rectangle(p1, p2)
                rect.setFill("blue")
                rect.setOutline("black")
                rect.draw(self.window)
        for ship in self.ships:
            row_start = ship.row
            row_end = ship.row + ship.size * ship.dr - ship.dr
            col_start = ship.col
            col_end = ship.col + ship.size * ship.dc - ship.dc
            if ship.dr > 0 or ship.dc > 0: # right or down
                p1 = Point(x_start + col_start * sq_width + sq_width // 4, y_start + row_start * sq_height + sq_height // 4)
                p2 = Point(x_start + col_end * sq_width + 3 * sq_width // 4, y_start + row_end * sq_height + 3 * sq_height // 4)
            else: # left or up
                p1 = Point(x_start + col_start * sq_width + 3 * sq_width // 4, y_start + row_start * sq_height + 3 * sq_height // 4)
                p2 = Point(x_start + col_end * sq_width + sq_width // 4, y_start + row_end * sq_height + sq_height // 4)
            s_rect = Rectangle(p1, p2)
            s_rect.setFill("black")
            s_rect.draw(self.window)
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                if self.shots[row, col]:
                    fill = "red" if self.board[row, col] else "gray"
                    center = Point(x_start + col * sq_width + sq_width//2, y_start + row * sq_height + sq_height//2)
                    circle = Circle(center, min(sq_height, sq_height) // 8)
                    circle.setFill(fill)
                    circle.draw(self.window)
        self.window.flush()

    def render(self, mode="human", width=720, height=480):
        if mode == "ansi":
            rendered = "\u250E" + ("\u2500" * self.board.shape[1]) + "\u2512\n"
            for shot_row, board_row in zip(self.shots, self.board):
                rendered += "\u2503"
                rendered += "".join(
                    (" " if not s else ("\u2591" if s and not b else "\u2588")) for s, b in zip(shot_row, board_row))
                rendered += "\u2503"
                rendered += "\n"
            rendered += "\u2516" + ("\u2500" * self.board.shape[1]) + "\u251A"
            return rendered
        elif mode == "human":
            self._render_graphics(width, height)
        else:
            raise NotImplementedError

    def _can_place(self, size, row, col, dr, dc):
        for _ in range(size):
            if row < 0 or row >= self.board.shape[0] or col < 0 or col >= self.board.shape[1]:
                return False
            if self.board[row, col]:
                return False
            row += dr
            col += dc
        return True

    def reset(self):
        self.board[:] = False
        self.shots[:] = False
        self.ships = []
        for ship in reversed(self.ship_lens):
            placed = False
            while not placed:
                row = np.random.randint(0, self.board.shape[0])
                col = np.random.randint(0, self.board.shape[1])
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                dr, dc = dirs[np.random.randint(0, len(dirs))]
                if self._can_place(ship, row, col, dr, dc):
                    placed = True
                    self.ships.append(_Ship(ship, row, col, dr, dc))
                    r, c = row, col
                    for _ in range(ship):
                        self.board[r, c] = True
                        r += dr
                        c += dc
        self.ships = self.ships[::-1]
        return self._observe()

    def close(self):
        if self.window is not None:
            self.window.close()
        self.window = None

if __name__ == "__main__":
    be = BattleshipEnv("flat-ships")
    be.shots = be.board.astype(bool)
    # be.shots = np.random.randn(*be.shots.shape) >= 0
    print(be.render("ansi"))
    print(be.board)
    state = be._observe()
    print(f"Dim: {state.shape}")
    print(state[-5:])
    print(be._done())
