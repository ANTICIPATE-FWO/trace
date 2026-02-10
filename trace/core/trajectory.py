from typing import List, Dict, Any
from json import load, dumps
import numpy as np

from trace.core import discount

class TrajectoryManager:
    def __init__(self, env_id:str):
        self.trajectories  = []
        self.env_id = env_id
        self.point_num, self.episode_num = 0, 0

        from trace.core import env_metadata
        self.metadata = env_metadata[env_id]
        self.num_actions = len(self.metadata['actions'])
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
        action_seq = self.sequence(key='actions', flatten=True, pad=None)
        assert all(a in self.metadata["actions"] for ep in action_seq for a in ep)

    def __len__(self):
        return len(self.trajectories)

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
        return np.array([
            np.mean([
                discount(episode["rewards"], gamma) for episode in point
            ], axis=0)
            for point in self.trajectories
        ])

    def sequence(self, key: str = "actions", pad: int|None = -1, flatten: bool = False):
        seq = [[episode[key] for episode in point] for point in self.trajectories]
        if flatten: seq = [episode for point in seq for episode in point]
        if pad: return np.array(homogenize(seq))
        return seq

    def distribution(self, key: str = "actions", normalize: bool = True, flatten: bool = False):
        dists = []

        for point in self.trajectories:
            dists.append([])
            for episode in point:
                counts = np.zeros(self.num_actions)
                for v in episode[key]: counts[self.action_mapping[v]] += 1
                if normalize: counts /= counts.sum()
                dists[-1].append(counts)
        if flatten: return np.stack([d for ep in dists for d in ep])
        return dists

    def policy_data(self, pad: int|None = -1, flatten: bool = False):
        obs = self.sequence(key="observations", pad=pad, flatten=flatten)
        acs = self.sequence(key="actions", pad=pad, flatten=flatten)
        return obs, acs

    def save(self, filepath: str):
        with open(filepath, "w") as f: f.write(dumps(self.trajectories, indent=2))


def homogenize(array, pad=-1):
    def shape(x):
        if not isinstance(x, list): return ()
        sub = [shape(i) for i in x]
        return (len(x),) + tuple(
            max((s[d] if d < len(s) else 0) for s in sub)
            for d in range(max(map(len, sub), default=0))
        )

    def fill(x, shp):
        if not shp: return x
        x = x if isinstance(x, list) else []
        return [
            fill(x[i] if i < len(x) else pad, shp[1:])
            for i in range(shp[0])
        ]

    return fill(array, shape(array))
