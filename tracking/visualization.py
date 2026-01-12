import matplotlib.pyplot as plt

def visualize_pareto(pareto_front, title:str="Pareto Front"):
    if pareto_front.shape[1] != 2:
        print(f"Visualization not possible for {pareto_front.shape[1]} dimensions")
        return
    plt.scatter( pareto_front[:, 0], pareto_front[:, 1], s=40,)
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(title)
    plt.grid(True)
    plt.show()

#print(f'Ground truth: {np.sum(eval_env.unwrapped.sea_map > 0)} points')

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
