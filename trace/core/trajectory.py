from json import load, dumps
import numpy as np

from trace.core.auxiliary import homogenize
from core import discount, all_ints


class TrajectoryManager:
    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.env_id = metadata['env_id']
        self.num_actions = len(self.metadata['actions'])
        self.gamma = metadata['gamma']

        self.trajectories = []
        self.point_num, self.episode_num = 0, 0
    

    def load(self, source, filtering:bool=False):
        if isinstance(source, str): self.trajectories = load(open(source, 'rb'))
        elif isinstance(source, list): self.trajectories = source
        else: raise ValueError(f'Unknown source type: {type(source)}')

        if filtering: self._filter_duplicates()

        self.point_num = len(self.trajectories)
        self.episode_num = sum(len(point) for point in self.trajectories)
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

    def _filter_duplicates(self):
        # todo auxiliary and out of class
        filtered_trajectories = []

        for point in self.trajectories:
            filtered_point, seen_episodes = [], set()
            for episode in point:
                if (episode_str := dumps(episode, sort_keys=True)) not in seen_episodes:
                    seen_episodes.add(episode_str)
                    filtered_point.append(episode)
            filtered_trajectories.append(filtered_point)

        self.trajectories = filtered_trajectories

    def accrue(self, key: str='rewards', gamma: float|None=None):
        if not gamma: gamma = self.gamma
        return np.array([
            np.mean([
                discount(episode[key], gamma) for episode in point
            ], axis=0)
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

    def policy_data(self, pad: int|None=None, flatten: bool=False):
        obs = self.sequence(key='observations', pad=pad, flatten=flatten)
        acs = self.sequence(key='actions', pad=pad, flatten=flatten)
        rew = self.accrue(key='rewards')
        return obs, acs, rew

    def save(self, filepath: str):
        with open(filepath, 'w') as f: f.write(dumps(self.trajectories, indent=2))

def traj_dict(obs, acs, rew):
    return [
        [{
        'observations': obs_episode,
        'actions': acs_episode,
        'rewards': rew_episode,
        } for obs_episode, acs_episode, rew_episode in zip(obs_point, acs_point, rew_point)]
        for obs_point, acs_point, rew_point in zip(obs, acs, rew)
    ]