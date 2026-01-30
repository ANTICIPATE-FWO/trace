from json import load
import numpy as np
from trace.analysis import filter_traj, rewards_per_episode
from trace.core import action_distribution


def main():
    filepath = "../data/38_dst_ipro.json"
    with open(filepath, "r") as f:
        trajectories = load(f)

    filtered_trajectories = filter_traj(trajectories)
    rewards_ep = np.array([
        rewards_per_episode(episode)
        for point in filtered_trajectories for episode in point
    ])
    distributions_ep = action_distribution(filtered_trajectories, normalize=True)
    print()

    #tsne_visualization(rewards_ep, distributions_ep)

    import matplotlib.pyplot as plt
    reward_dim = rewards_ep.shape[1]
    for i in range(reward_dim):
        sc = plt.scatter(distributions_ep[:, 1], distributions_ep[:, 3], c=rewards_ep[:,i])
        plt.colorbar(sc)
        plt.title(f"Reward {i}")
        plt.show()
    return



if __name__ == "__main__":
    main()
