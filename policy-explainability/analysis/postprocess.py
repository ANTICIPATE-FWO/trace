from json import dumps
from typing import Any, Dict, List

def save_traj(pareto_set, path:str="data/dst_ipro.json"):
    all_trajectories = [traj for _, _, traj in pareto_set]

    with open(path, "w") as f:
        f.write(dumps(all_trajectories, indent=2))

    return

def filter_traj(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered_trajectories = []

    for point in trajectories:
        filtered_point, seen = [], set()

        for episode in point:
            key = dumps(episode, sort_keys=True)

            if key not in seen:
                seen.add(key)
                filtered_point.append(episode)

        filtered_trajectories.append(filtered_point)

    return filtered_trajectories