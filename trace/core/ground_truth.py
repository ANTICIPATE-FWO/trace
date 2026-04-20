import numpy as np

from collections import deque
from random import choices


def shortest_distances(sea_map: np.ndarray, start: tuple, action_mapping: dict):
    h, w = sea_map.shape
    dist = np.full((h, w), np.inf)
    dist[start] = 0
    q = deque([start])

    while q:
        r, c = q.popleft()
        for dr, dc in action_mapping:
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