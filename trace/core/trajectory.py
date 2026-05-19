from json import load, dumps
import numpy as np

from core import pareto_filter
from trace.core.maths import discount, all_ints


class TrajectoryManager:
    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.env_id = metadata['env_id']
        self.num_actions = len(self.metadata['actions'])
        self.gamma = metadata['gamma']

        self.trajectories = []

    def load(self, source, filtering:bool=False, flat:bool=False):
        if isinstance(source, str): self.trajectories = load(open(source, 'rb'))
        elif isinstance(source, list): self.trajectories = source
        else: raise ValueError(f'Unknown source type: {type(source)}')

        if flat:
            self.trajectories = [[traj] for point in self.trajectories for traj in point]

        if filtering:
            #self.trajectories = [filter_duplicates(traj) for traj in self.trajectories]
            self.trajectories = [[self.trajectories[i][0]] for i in pareto_filter(self.accrue())]


        self._verify_data()
        return self

    def subset(self, labels:list):
        new_trajectories = [point for point, l in zip(self.trajectories, labels) if l]
        return TrajectoryManager(metadata=self.metadata).load(new_trajectories)

    def _verify_data(self):
        # todo include shape assertations
        action_seq = self.sequence(key='actions', flatten=True, pad=None)
        assert all(a in self.metadata['actions'] for ep in action_seq for a in ep)

    def __len__(self):
        return len(self.trajectories)

    def accrue(self, key: str='rewards', gamma: float|None=None):
        if not gamma: gamma = self.gamma
        return np.array([
            np.mean([
                discount(trajectory[key], gamma) for trajectory in point
            ],axis=0)
            for point in self.trajectories
        ])

    def sequence(self, key: str='actions', pad: int|None=None, flatten: bool=False):
        seq = [[episode[key] for episode in point] for point in self.trajectories]
        if flatten: seq = [episode for point in seq for episode in point]
        if pad: return np.array(homogenize(seq))
        return seq

    def distribution(self, key: str='actions', normalize: bool=True, flatten: bool=False):
        assert all_ints(self.metadata['actions'].keys()), 'Cannot split to distribution for non-int actions'

        dists = []
        for point in self.trajectories:
            dists.append([])
            for episode in point:
                counts = np.zeros(self.num_actions)
                for v in episode[key]: counts[v] += 1
                if normalize: counts /= np.sum(counts)
                dists[-1].append(counts)
        if flatten: return np.stack([d for ep in dists for d in ep])
        return dists

    def conditioning_features(self, pad: int|None=None, flatten: bool=False, gamma: float|None=None):
        obs = self.sequence(key='observations', pad=pad, flatten=flatten)
        acs = self.sequence(key='actions', pad=pad, flatten=flatten)
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
