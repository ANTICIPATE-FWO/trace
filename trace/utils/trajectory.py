from typing import List, Dict, Any
from json import load, dumps
import numpy as np
from numbers import Number

class TrajectoryManager:
    def __init__(self, actions:list):
        self.trajectories, self.point_num, self.episode_num = [], 0, 0
        self.actions = sorted(actions)
        # does it make sense to use the env name and act as data verification as well?
        # maybe if it runs into many errors this way
        # if this happens turn the utils into core and move the initialization and enc data inside it
        self.action_mapping = {a: i for i, a in enumerate(self.actions)}

    def load(self, source, filtering=False):
        if isinstance(source, str): self.trajectories = load(open(source, "rb"))
        elif isinstance(source, list): self.trajectories = [traj for _, _, traj in source]
        else: raise ValueError(f"Unknown source type: {type(source)}")

        if filtering: self._filter_duplicates()

        self.point_num = len(self.trajectories)
        self.episode_num = sum(len(point) for point in self.trajectories)
        return self

    def _filter_duplicates(self):
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

    def append(self, point: List[Dict[str, Any]]):
        self.trajectories.append(point)
        self.point_num = len(self.trajectories)
        self.episode_num = sum(len(point) for point in self.trajectories)

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


    def sequence(self, key: str = "actions", pad: int|None = -1):
        seq = [episode[key] for point in self.trajectories for episode in point]
        if pad is None: return seq

        max_len = max(len(a) for a in seq)

        padding = pad if isinstance(seq[0][0], Number) else [pad] * len(seq[0][0])
        return np.array([s + [padding] * (max_len - len(s)) for s in seq])


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

