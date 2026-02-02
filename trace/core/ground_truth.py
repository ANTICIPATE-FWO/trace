from collections import deque
import numpy as np

from trace.core import env_metadata
ACTIONS = env_metadata['deep-sea-treasure-v0']['actions']


def shortest_distances(sea_map, start):
    h, w = sea_map.shape
    dist = np.full((h, w), np.inf)
    dist[start] = 0

    q = deque([start])

    while q:
        r, c = q.popleft()
        for dr, dc in ACTIONS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if sea_map[nr, nc] != -10. :
                    if dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))
    return dist


def enumerate_shortest_paths(sea_map, dist, start, target):
    h, w = sea_map.shape
    paths = []

    def dfs(pos, observations, actions):
        if pos == target:
            paths.append((observations.copy(), actions.copy()))
            return

        r, c = pos
        for a, (dr, dc) in ACTIONS.items():
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



def dst_ground_truth(sea_map, start: tuple = (0,0)):
    dist = shortest_distances(sea_map, start)
    treasure_cells = np.argwhere(sea_map > 0)
    ground_truth = []

    for r, c in treasure_cells:
        if np.isinf(dist[r, c]): continue
        paths = enumerate_shortest_paths(sea_map, dist, start, (r, c))

        ground_truth += [{
            "observations": observations,
            "actions": actions,
            "rewards": [sea_map[r, c], -int(dist[r, c])],
            } for observations, actions in paths]
    return ground_truth
