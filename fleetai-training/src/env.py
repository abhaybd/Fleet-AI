from typing import Any

import numpy as np
import gym


class BattleshipEnv(gym.Env):
    def __init__(self, observation_space="flat-ships", action_space="coords", size=10, ships=(1, 2, 3, 4, 5)):
        self.board = np.zeros((size, size), dtype=np.int8)
        self.ship_lens = sorted(ships)
        self.shots = np.zeros_like(self.board, dtype=bool)
        self.observation_space_type = observation_space
        self.action_space_type = action_space
        self.window = None

        if action_space == "coords":
            self.action_space = gym.spaces.MultiDiscrete((size, size))
        elif action_space == "flat":
            self.action_space = gym.spaces.Discrete(size * size)
        else:
            raise Exception(f"Unrecognized action space {action_space}")

        if observation_space == "flat":
            self.observation_space = gym.spaces.MultiBinary(2 * size * size)
        elif observation_space == "flat-ships":
            self.observation_space = gym.spaces.MultiBinary(2 * size * size + len(ships))
        else:
            raise Exception(f"Unrecognized observation space {observation_space}")
        self.reset()

    def _observe(self):
        board = self.board.astype(bool)
        hits = (board & self.shots).astype(np.int8)
        misses = (~board & self.shots).astype(np.int8)
        obs = np.hstack((misses.flatten(), hits.flatten()))
        if "ships" in self.observation_space_type:
            ship_obs = np.empty((len(self.ship_lens),), dtype=np.int8)
            ship_hits = self.board[board & self.shots]
            for i, ship in enumerate(self.ship_lens):
                sunk = (ship_hits == ship).sum() == ship
                ship_obs[i] = sunk
            obs = np.hstack((obs, ship_obs))
        return obs

    def _done(self):
        board = self.board.astype(bool)
        return (board == (board & self.shots)).all()

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
        from graphics import GraphWin, Rectangle, Point, Circle, Text
        if self.window is None:
            self.window = GraphWin(width=width, height=height, autoflush=False)
            self.window.setCoords(0, 0, width, height)
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
        for row in reversed(range(self.board.shape[0])):
            for col in range(self.board.shape[1]):
                p1 = Point(x_start + col * sq_width, y_start + row * sq_height)
                p2 = Point(x_start + (col + 1) * sq_width, y_start + (row + 1) * sq_height)
                rect = Rectangle(p1, p2)
                rect.setFill("blue")
                rect.setOutline("black")
                rect.draw(self.window)
                if self.board[row, col]:
                    label = Text(Point(x_start + col * sq_width + 10, y_start + row * sq_height + 10),
                                 str(self.board[row, col]))
                    label.setFill("black")
                    label.draw(self.window)
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
        self.board[:] = 0
        self.shots[:] = False
        ships = reversed(sorted(self.ship_lens))
        for ship in ships:
            placed = False
            while not placed:
                row = np.random.randint(0, self.board.shape[0])
                col = np.random.randint(0, self.board.shape[1])
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                dr, dc = dirs[np.random.randint(0, len(dirs))]
                if self._can_place(ship, row, col, dr, dc):
                    placed = True
                    r, c = row, col
                    for _ in range(ship):
                        self.board[r, c] = ship
                        r += dr
                        c += dc
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
