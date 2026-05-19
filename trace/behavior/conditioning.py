from collections import defaultdict
import numpy as np

from trace.core import all_ints, discretize


class EmpiricalDistribution:
    def __init__(self, metadata: dict, alpha: float=1.0, step: float=0.1):
        self.metadata, self.alpha = metadata, alpha
        self.env_id, self.actions = metadata['env_id'], metadata['actions']

        highs, lows = metadata['observations_high'], metadata['observations_low']
        self.step = 1 if all_ints(highs) and all_ints(lows) else step
        self.obs_space = [np.arange(l, h+1, self.step) for l,h in zip(lows, highs)]
        self.counts = defaultdict(lambda: np.zeros(len(self.actions)))

    def update(self, obs, action):
        key = discretize(obs, step=self.step)
        self.counts[key][action] += 1

    def obs_shape(self):
        return [dim.shape[0] for dim in self.obs_space]

    def get_action_dims(self):
        return len(self.actions)

    def get_actions(self):
        return list(self.actions.values())

    def get_visited(self):
        return list(self.counts.keys())

    def fit(self, obs_seq, act_seq):
        if not isinstance(obs_seq[0], (list, np.ndarray)): obs_seq = [obs_seq]
        if not isinstance(act_seq[0], (list, np.ndarray)): act_seq = [act_seq]

        for episode_obs, episode_acts in zip(obs_seq, act_seq):
            for obs, acts in zip(episode_obs, episode_acts): self.update(obs, int(acts))
        return self

    def action_probs(self, obs):
        key = discretize(obs, step=self.step)
        probs = (self.counts[key] + self.alpha)
        return probs / np.sum(probs)

    def act(self, obs, deterministic=True):
        probs = self.action_probs(obs)

        if deterministic: return int(probs.argmax())
        return int(np.random.choice(len(self.actions), p=probs))
