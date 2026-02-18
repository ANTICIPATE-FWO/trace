from collections import defaultdict
import numpy as np

def discretize_obs(obs, precision=2):
    return tuple(round(float(x), precision) for x in obs)

def all_ints(lst):
    return all(isinstance(x, int) for x in lst)

class BayesianPolicy:
    def __init__(self, env_id: str = 'deep-sea-treasure-v0', alpha=1.0, precision=1):
        self.env_id, self.alpha, self.precision = env_id, alpha, precision

        from trace.core import env_metadata
        self.num_actions = len(env_metadata[env_id]['actions'])

        highs = env_metadata[env_id]['observations_high']
        lows = env_metadata[env_id]['observations_low']

        step = 1 if all_ints(highs) and all_ints(lows) else 10**(-self.precision)
        self.obs_space = [np.arange(l, h+1, step) for l,h in zip(lows, highs)]
        self.counts = defaultdict(lambda: np.zeros(self.num_actions))


    def update(self, obs, action):
        key = discretize_obs(obs, precision=self.precision)
        self.counts[key][action] += 1

    def obs_shape(self):
        return [dim.shape[0] for dim in self.obs_space]

    def get_action_dims(self):
        return self.num_actions

    def get_visited(self):
        return list(self.counts.keys())

    def fit(self, obs_seq, act_seq):
        if not isinstance(obs_seq[0], (list, np.ndarray)): obs_seq = [obs_seq]
        if not isinstance(act_seq[0], (list, np.ndarray)): act_seq = [act_seq]

        for episode_obs, episode_acts in zip(obs_seq, act_seq):
            for obs, acts in zip(episode_obs, episode_acts): self.update(obs, int(acts))
        return self

    def action_probs(self, obs):
        key = discretize_obs(obs, precision=self.precision)
        probs = (self.counts[key] + self.alpha)
        return probs / probs.sum()

    def act(self, obs, deterministic=True):
        probs = self.action_probs(obs)

        if deterministic: return int(probs.argmax())
        return int(np.random.choice(self.num_actions, p=probs))
