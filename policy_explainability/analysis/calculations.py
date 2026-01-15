import numpy as np

def scalarized_reward(point, gamma=0.99):
    returns = []

    for ep in point:
        rewards = np.asarray(ep["rewards"], dtype=np.float32)
        timesteps = rewards.shape[0]

        discounts = gamma ** np.arange(timesteps, dtype=np.float32)
        discounted_return = (rewards * discounts[:, None]).sum(axis=0)

        returns.append(discounted_return)

    return np.mean(returns, axis=0)

def episode_reward(episode):
    rewards = np.asarray(episode["rewards"], dtype=np.float32)
    return rewards.sum(axis=0)