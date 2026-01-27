from typing import Optional, List, Dict, Any, Union
from json import load, dumps
import numpy as np

class TrajectoryManager:
    def __init__(self,  actions: List[float], filtering: bool = True):
        self.trajectories = []
        self.actions = sorted(actions)
        self.action_mapping = {a:i for i,a in enumerate(self.actions)}
        if filtering: self.filter_duplicates()
        self.point_num = len(self.trajectories)
        self.episode_num = sum(len(point) for point in self.trajectories)
        self.reward_dim = 0 #todo

    def load_file(self, filepath: str):
        with open(filepath, "r") as f: self.trajectories = load(f)
        self.point_num = len(self.trajectories)
        self.episode_num = sum(len(point) for point in self.trajectories)

    def load_pareto(self, pareto_set: List[Any]):
        self.trajectories = [traj for _, _, traj in pareto_set]
        self.point_num = len(self.trajectories)
        self.episode_num = sum(len(point) for point in self.trajectories)

    def append(self, point: List[Dict[str, Any]]):
        self.trajectories.append(point)
        self.point_num = len(self.trajectories)
        self.episode_num = sum(len(point) for point in self.trajectories)


    def filter_duplicates(self):
        filtered_trajectories = []

        for point in self.trajectories:
            filtered_point, seen = [], set()

            for episode in point:
                key = dumps(episode, sort_keys=True)

                if key not in seen:
                    seen.add(key)
                    filtered_point.append(episode)

            filtered_trajectories.append(filtered_point)

        self.trajectories = filtered_trajectories


    def rewards_ep(self):
        rewards = []
        for point in self.trajectories:
            for episode in point:
                r = np.asarray(episode["rewards"], dtype=np.float32)
                rewards.append(r.sum(axis=0))

        return np.array(rewards)

    def rewards_sc(self, gamma: float = 0.99):
        # todo make this more clear
        rewards = []
        for point in self.trajectories:
            point_return = []
            for episode in point:
                r = np.asarray(episode["rewards"], dtype=np.float32)
                discounts = gamma ** np.arange(r.shape[0], dtype=np.float32)
                discounted_r = (r * discounts[:, None]).sum(axis=0)

                point_return.append(discounted_r)
            rewards.append(np.mean(point_return, axis=0))

        return rewards


    def sequence(self, key: str = "actions", padding: Optional[int] = -1):
        seq = [episode[key] for point in self.trajectories for episode in point]
        if padding is None: return seq

        max_len = max(len(a) for a in seq)
        return np.array([a + [padding] * (max_len - len(a)) for a in seq])


    def distribution(self, key: str = "actions", normalize: bool = True):
        mapping = self.action_mapping  # todo state mapping
        dists = []

        for point in self.trajectories:
            for episode in point:
                counts = np.zeros(len(mapping))
                for v in episode[key]: counts[mapping[v]] += 1
                if normalize: counts /= counts.sum()
                dists.append(counts)
        return np.array(dists)


    def save(self, filepath: str):
        with open(filepath, "w") as f: f.write(dumps(self.trajectories, indent=2))