from json import load, dumps
from yaml import safe_load
import numpy as np

from trace.core.maths import discount, pareto_filter


class TrajectoryManager:
    def __init__(self, metadata: dict|str):
        self.metadata = metadata if isinstance(metadata, dict) \
            else safe_load(open(f'trace/configs/{metadata}.yaml'))
        self.env_id = self.metadata['env_id']
        self.num_actions = len(self.metadata['actions'])
        self.gamma = self.metadata['gamma']

        self.trajectories = []

    def load(self, source, pareto:bool=False, duplicates:bool=False, split:bool=False):
        if isinstance(source, str): self.trajectories = load(open(source, 'rb'))
        elif isinstance(source, list): self.trajectories = source
        else: raise ValueError(f'Unknown source type: {type(source)}')

        if split: self.trajectories = [[traj] for point in self.trajectories for traj in point]
        if duplicates: self.trajectories = filter_duplicates(self.trajectories)
        if pareto: self.trajectories = [self.trajectories[ind] for ind in pareto_filter(self.accrue())]

        self._verify_data()
        return self

    def subset(self, labels: list|np.ndarray):
        new_trajectories = [point for point, l in zip(self.trajectories, labels) if l]
        return TrajectoryManager(metadata=self.metadata).load(new_trajectories)

    def _verify_data(self):
        # todo include shape assertations
        action_seq = self.sequence(key='actions', per_point=False, pad=None)
        assert all(a in self.metadata['actions'] for ep in action_seq for a in ep)

    def __len__(self):
        return len(self.trajectories)

    def accrue(self, key: str='rewards', gamma: float|None=None):
        if gamma is None: gamma = self.gamma
        return np.array([
            np.mean([
                discount(trajectory[key], gamma) for trajectory in point
            ], axis=0)
            for point in self.trajectories
        ])

    def sequence(self, key:str='actions', pad:int|None=None, per_point:bool=False):
        if per_point:
            seq = [[trajectory[key] for trajectory in point] for point in self.trajectories]
        else:
            seq = [trajectory[key] for point in self.trajectories for trajectory in point]

        if pad: return np.array(homogenize(seq))
        return seq

    def conditioning_features(self, pad: int|None=None, per_point:bool=False, gamma:float|None=None, ):
        obs = self.sequence(key='observations', pad=pad, per_point=per_point)
        acs = self.sequence(key='actions', pad=pad, per_point=per_point)
        rew = self.accrue(key='rewards', gamma=gamma)
        return obs, acs, rew

    def save(self, filepath: str):
        with open(filepath, 'w') as f: f.write(dumps(self.trajectories, indent=2))


def homogenize(array:list, pad:int=-1):
    def homogenous_shape(x):
        if not isinstance(x, list): return ()
        sub_array = [homogenous_shape(i) for i in x]
        return (len(x),) + tuple(
            max((s[d] if d < len(s) else 0) for s in sub_array)
            for d in range(max(map(len, sub_array), default=0))
        )

    def fill(x, shp:tuple):
        if not shp: return x
        x = x if isinstance(x, list) else []
        return [
            fill(x[i] if i < len(x) else pad, shp[1:])
            for i in range(shp[0])
        ]

    return fill(array, homogenous_shape(array))


def filter_duplicates(array:list, sort:bool=True):
    filtered, seen = [], set()
    for element in array:
        if sort:
            if isinstance(element, dict): element = dict(sorted(element.items()))
            else: element = sorted(element)
        if (element_str := str(element)) not in seen:
            seen.add(element_str)
            filtered.append(element)
    return filtered
