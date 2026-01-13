from json import load
import numpy as np
from analysis import scalarized_reward, filter_traj, pretty_print, visualize_pareto

def main():
    filepath = "../data/38_dst_ipro.json"
    with open(filepath, "r") as f:
        trajectories = load(f)

    rewards_sc = [scalarized_reward(pareto_point) for pareto_point in trajectories]
    filtered_trajectories = filter_traj(trajectories)


    print("Pareto points: ", len(rewards_sc))
    print("Action lengths: ")
    print([len(episode['actions']) for point in filtered_trajectories for episode in point])


    visualize_pareto(np.array(rewards_sc))
    pretty_print(rewards_sc, filtered_trajectories)



if __name__ == "__main__":
    main()




