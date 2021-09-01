import torch
import numpy as np

class PPOBuffer(object):
    def __init__(self):
        self.trajectories = []

    def create_traj(self):
        traj_id = len(self.trajectories)
        self.trajectories.append(_Traj())
        return traj_id

    def put_single_data(self, traj_id, state, action, log_prob, reward):
        if isinstance(state, (float, int)):
            state = np.array([state])
        if isinstance(action, (float, int)):
            action = np.array([action])
        traj = self.trajectories[traj_id]
        if traj.last_val is not None:
            raise Exception("Cannot add data to finished trajectory!")
        traj.states.append(state)
        traj.actions.append(action)
        traj.log_probs.append([log_prob])
        traj.rewards.append([reward])

    def put_data(self, data):
        if len(data) != len(self.trajectories):
            raise Exception(f"Required {len(data)} data points, found {len(self.trajectories)}")
        for d in data:
            self.put_single_data(*d)

    def finish_traj(self, traj_id, last_val):
        """
        If the trajectory terminated (done=True) then last_val=0.
        If it stopped early due to the horizon, last_val should be the last value predicted by the critic.
        This allows for bootstrapping past the episode horizon.
        """
        self.trajectories[traj_id].last_val = last_val

    def get(self, device):
        if any(traj.last_val is None for traj in self.trajectories):
            raise Exception("Not all trajectories have been finished!")

        to_tensor = lambda x: torch.from_numpy(np.array(x, dtype=np.float32)).to(device)

        states = [to_tensor(traj.states) for traj in self.trajectories]
        actions = [to_tensor(traj.actions) for traj in self.trajectories]
        log_probs = [to_tensor(traj.log_probs) for traj in self.trajectories]
        rewards = [to_tensor(traj.rewards) for traj in self.trajectories]
        last_vals = [to_tensor([traj.last_val]) for traj in self.trajectories]
        self.clear()
        return states, actions, log_probs, rewards, last_vals

    def clear(self):
        self.trajectories.clear()

    def size(self):
        return sum(map(len, self.trajectories))

class _Traj(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.last_val = None

    def __len__(self):
        assert all(len(x) == len(self.states) for x in [self.states, self.actions, self.log_probs, self.rewards])
        return len(self.states)
