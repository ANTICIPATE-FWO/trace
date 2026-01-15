from json import load
import numpy as np
from policy_explainability.analysis import filter_traj, pretty_print, visualize_pareto, episode_reward, tsne_visualization
from policy_explainability.utils import action_distribution




def main():
    filepath = "../data/38_dst_ipro.json"
    with open(filepath, "r") as f:
        trajectories = load(f)

    filtered_trajectories = filter_traj(trajectories)
    rewards_ep = np.array([
        episode_reward(episode)
        for point in filtered_trajectories for episode in point
    ])
    distributions_ep = action_distribution(filtered_trajectories, normalize=True)

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
