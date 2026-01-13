from json import load
import numpy as np
np.set_printoptions(precision=3)

from analysis import scalarized_reward, filter_traj

def main():
    filepath = "../data/mc_ipro.json"

    with open(filepath, "r") as f:
        trajectories = load(f)

    filtered_trajectories = filter_traj(trajectories)
    rewards_sc = [scalarized_reward(pareto_point) for pareto_point in trajectories]

    print("Pareto points:")
    for r_sc, f_traj in zip(rewards_sc, filtered_trajectories):
        print(f'Expected return: {r_sc} \t\tUnique trajectories: {len(f_traj)}')


if __name__ == "__main__":
    main()




