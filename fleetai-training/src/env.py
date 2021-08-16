import numpy as np
import gym

class BattleshipEnv(gym.Env):
    def __init__(self, size=10, ships=(1,2,3,4,5)):
        self.board = np.zeros((size, size), dtype=bool)
        self.ship_lens = ships
        self.shots = np.zeros_like(self.board, dtype=bool)
        self.action_space = gym.spaces.MultiDiscrete((size, size))
        self.observation_space = gym.spaces.MultiBinary((2, size, size)) # 0=misses,1=hits
        self.reset()

    def _observe(self):
        hits = (self.board & self.shots).astype(np.int8)
        misses = (~self.board & self.shots).astype(np.int8)
        return np.stack((misses, hits))

    def _done(self):
        return (self.board == (self.board & self.shots)).all()

    def step(self, action):
        row, col = action
        if self.shots[row, col]:
            reward = -1
        else:
            reward = 1 if self.board[row, col] else 0
            self.shots[row, col] = True
        return self._observe(), reward, self._done(), {}

    def render(self, mode='human'):
        if mode == "ansi":
            rendered = "\u250E" + ("\u2500" * self.board.shape[1]) + "\u2512\n"
            for shot_row, board_row in zip(self.shots, self.board):
                rendered += "\u2503"
                rendered += "".join((" " if not s else ("\u2591" if s and not b else "\u2588")) for s,b in zip(shot_row, board_row))
                rendered += "\u2503"
                rendered += "\n"
            rendered += "\u2516" + ("\u2500" * self.board.shape[1]) + "\u251A"
            return rendered
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
        ships = reversed(sorted(self.ship_lens))
        for ship in ships:
            placed = False
            while not placed:
                row = np.random.randint(0, self.board.shape[0])
                col = np.random.randint(0, self.board.shape[1])
                dirs = [(0,1),(0,-1),(1,0),(-1,0)]
                dr,dc = dirs[np.random.randint(0,len(dirs))]
                if self._can_place(ship, row, col, dr, dc):
                    placed = True
                    r,c = row, col
                    for _ in range(ship):
                        self.board[r,c] = True
                        r += dr
                        c += dc

if __name__ == "__main__":
    be = BattleshipEnv()
    be.shots = be.board
    # be.shots = np.random.randn(*be.shots.shape) >= 0
    print(be.render("ansi"))