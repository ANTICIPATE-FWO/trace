from collections import defaultdict
import numpy as np

def discretize_obs(obs, precision=2):
    return tuple(round(float(x), precision) for x in obs)

class BayesianDSTPolicy:
    def __init__(self, num_actions, alpha=1.0):
        self.num_actions = num_actions
        self.alpha = alpha

        # state → action counts
        self.counts = defaultdict(lambda: np.zeros(num_actions))

    def update(self, obs, action):
        key = discretize_obs(obs)
        self.counts[key][action] += 1

    def fit(self, obs_seq, act_seq):
        if act_seq.ndim == 3: act_seq = act_seq.argmax(axis=-1)

        for traj_obs, traj_act in zip(obs_seq, act_seq):
            for o, a in zip(traj_obs, traj_act):
                self.update(o, int(a))

    def action_probs(self, obs):
        key = discretize_obs(obs)
        counts = self.counts[key]

        probs = (counts + self.alpha)
        return probs / probs.sum()

    def act(self, obs, deterministic=False):
        probs = self.action_probs(obs)

        if deterministic: return int(probs.argmax())
        return int(np.random.choice(self.num_actions, p=probs))
