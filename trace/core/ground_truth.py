import numpy as np

from itertools import combinations
from collections import deque
from random import choices
from math import ceil

from trace.core.maths import point_dist

def shortest_distances(sea_map: np.ndarray, start: tuple, action_mapping: dict):
    h, w = sea_map.shape
    dist = np.full((h, w), np.inf)
    dist[start] = 0
    q = deque([start])

    while q:
        r, c = q.popleft()
        for dr, dc in action_mapping.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if sea_map[nr, nc] != -10. :
                    if dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))
    return dist


def enumerate_shortest_paths(sea_map: np.ndarray, dist: np.ndarray, start: tuple, target: tuple, action_mapping: dict):
    h, w = sea_map.shape
    paths = []

    def dfs(pos, observations, actions):
        if pos == target:
            paths.append((observations.copy(), actions.copy()))
            return

        r, c = pos
        for a, (dr, dc) in action_mapping.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if sea_map[nr, nc] != -10.:
                    if dist[nr, nc] == dist[r, c] + 1:
                        observations.append((nr, nc))
                        actions.append(a)
                        dfs((nr, nc), observations, actions)
                        observations.pop()
                        actions.pop()

    dfs(start, [start], [])
    return paths


def dst_ground_truth(sea_map: np.ndarray, action_mapping: dict, start: tuple = (0,0)):
    dist = shortest_distances(sea_map, start, action_mapping)
    treasure_cells = np.argwhere(sea_map > 0)
    ground_truth = []

    for r, c in treasure_cells:
        if np.isinf(dist[r, c]): continue
        paths = enumerate_shortest_paths(sea_map, dist, start, (r, c), action_mapping)
        for observations, actions in paths:
            ground_truth.append([{
                "observations": observations,
                "actions": actions,
                "rewards": [[0, -1] for _ in range(int(dist[r, c]) - 1)] + [[sea_map[r, c], -1]],
            }])
    return ground_truth


def synthetic_stochastic_points(ground_truth: list, num: int=100, length: int=10):
    episodes = [episode for point in ground_truth for episode in point]
    return [choices(episodes, k=length) for _ in range(num)]


def path_combinations(mines: list|np.ndarray, base: list|np.ndarray):
    mines, base = [list(mine) for mine in mines], list(base)
    all_paths = []

    for m in range(1, len(mines) + 1):
        for combo in combinations(mines, m):
            path = [base, *combo, base]
            all_paths.append(best_visit_order(path))

    return all_paths


def best_visit_order(path: list):
    # todo make deterministic instead of greedy
    remaining = path[:-1]
    best_path = [remaining.pop(0)]

    while remaining:
        next_node = min(remaining, key=lambda node: point_dist(best_path[-1], node))
        remaining.remove(next_node)
        best_path.append(next_node)

    best_path.append(best_path[0])
    return best_path


def best_acc_steps(trail_length:float, speed_acc:float=0.0075, fuel_acc:float=0.025, fuel_idle:float=0.005):
    best_fuel, best_steps = float('inf'), 0

    for acc_steps in range(1, ceil(trail_length / speed_acc)):
        vel = acc_steps * speed_acc
        d_acc = speed_acc * acc_steps * (acc_steps + 1) / 2

        if d_acc >= trail_length: # end reached while accelerating
            if acc_steps * (fuel_acc + fuel_idle) < best_fuel:
                return acc_steps

        d_rem = trail_length - d_acc
        idle_steps = ceil(d_rem / vel)
        fuel = acc_steps * (fuel_acc + fuel_idle) + idle_steps * fuel_idle

        if fuel < best_fuel:
            best_fuel, best_steps = fuel, acc_steps

    return best_steps

def main():
    print(best_acc_steps(0.339999))

if __name__ == '__main__':
    main()