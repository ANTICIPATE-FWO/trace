import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

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
