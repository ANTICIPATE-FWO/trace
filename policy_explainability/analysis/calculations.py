import numpy as np

def scalarized_reward(trajectory, gamma=0.99):
    returns = []

    for traj in trajectory:
        rewards = np.asarray(traj["rewards"], dtype=np.float32)
        timesteps = rewards.shape[0]

        discounts = gamma ** np.arange(timesteps, dtype=np.float32)
        discounted_return = (rewards * discounts[:, None]).sum(axis=0)

        returns.append(discounted_return)

    return np.mean(returns, axis=0)
