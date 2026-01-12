from json import dumps

def save_traj(pareto_set, path:str="data/data.json"):

    all_trajectories = [traj for _, _, traj in pareto_set]
    #implement filtering function after forking

    with open(path, "w") as f:
        f.write(dumps(all_trajectories, indent=2))

    #visualize_pareto(pareto_front)


def filter_traj():
    pass