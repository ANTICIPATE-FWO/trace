import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from sklearn.manifold import TSNE


def visualize_pareto(pareto_front:np.ndarray, ground_truth:Optional[np.ndarray]=None, title:str="Pareto Front"):
    if pareto_front.shape[1] != 2:
        print(f"Visualization not possible for {pareto_front.shape[1]} dimensions")
        return
    plt.scatter( pareto_front[:, 0], pareto_front[:, 1], s=40, label="Pareto Front")
    if ground_truth is not None:
        plt.scatter(ground_truth[:, 0], ground_truth[:, 1], s=40, label="Ground Truth")
        plt.legend()
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(title)
    plt.grid(True)
    plt.show()


def visualize_dst_map(env):
    sea_map = env.unwrapped.sea_map
    print(sea_map)
    plt.figure(figsize=(6, 6))
    plt.imshow(sea_map, cmap="viridis", origin="upper")

    for i in range(sea_map.shape[0]):
        for j in range(sea_map.shape[1]):
            value = sea_map[i, j]
            if value > 0:
                plt.text(
                    j, i,
                    f"{value:g}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.title("Deep Sea Treasure Map")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()


def pretty_print(rewards_sc, filtered_trajectories):
    print("\nExpected return\t\t\t  |\tUnique trajectories")
    print("-" * 60)

    for r_sc, f_traj in zip(rewards_sc, filtered_trajectories):
        r_str = np.array2string(r_sc, formatter={'float_kind': lambda x: f"{x:+.3f}"})
        print(f"{r_str:<25} | {len(f_traj):>3}")


# TODO decide what to do about tsne
def tsne_visualization(rewards, action_distributions, perplexity: int = 30, random_state: int = 0):
    reward_dim = rewards.shape[1]

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=random_state
    )
    z = tsne.fit_transform(action_distributions)

    fig, axes = plt.subplots(
        1,
        reward_dim,
        figsize=(5 * reward_dim, 4),
        squeeze=False
    )

    for i in range(reward_dim):
        ax = axes[0, i]
        sc = ax.scatter(
            z[:, 0],
            z[:, 1],
            c=rewards[:, i],
            s=20,
            alpha=0.8
        )
        ax.set_title(f"Reward dimension {i}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.colorbar(sc, ax=ax, label=f"Reward dim {i}")

    plt.tight_layout()
    plt.show()