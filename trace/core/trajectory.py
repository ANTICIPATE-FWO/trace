from typing import List, Dict, Any
from json import load, dumps
import numpy as np
from numbers import Number

from trace.core import discount

class TrajectoryManager:
    def __init__(self, env_id:str):
        self.trajectories  = []
        self.env_id = env_id
        self.point_num, self.episode_num = 0, 0

        from trace.core import env_metadata
        self.metadata = env_metadata[env_id]
        self.action_mapping = {a: i for i, a in enumerate(self.metadata['actions'].keys())}

    def load(self, source, filtering=False):
        if isinstance(source, str): self.trajectories = load(open(source, "rb"))
        elif isinstance(source, list): self.trajectories = source
        else: raise ValueError(f"Unknown source type: {type(source)}")

        if filtering: self._filter_duplicates()

        self.point_num = len(self.trajectories)
        self.episode_num = sum(len(point) for point in self.trajectories)
        self._verify_data()
        return self

    def _verify_data(self):
        # todo include shape assertations
        action_seq = self.sequence(key='actions', pad=None)
        assert all(a in self.metadata["actions"] for ep in action_seq for a in ep)


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
                rewards.append(r if r.ndim==1 else r.sum(axis=0))

        return np.array(rewards)

    def accrued_reward(self, gamma: float = 0.99):
        return [
            np.mean([
                discount(episode["rewards"], gamma) for episode in point
            ], axis=0)
            for point in self.trajectories
        ]

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

