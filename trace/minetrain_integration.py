import numpy as np
np.set_printoptions(suppress=True, precision=2)

from trace.core import initialize_setting, path_combinations, seg_angle, point_dist, trail_strategy
from trace.visuals import minecart_trajectories

import mo_gymnasium.envs.minecart.minecart as minecart

def path_playout(env, path, eps:float=0.1):
    #todo cleanup
    observations, actions, rewards = [], [], []
    terminated, truncated = False, False
    obs, _ = env.reset()
    observations.append(obs)

    for e in range(len(path) - 1):
        start, end = path[e], path[e + 1]
        trail_angle = seg_angle(start, end)
        rot_action = minecart.ACT_LEFT if env.get_angle() - trail_angle > 0 else minecart.ACT_RIGHT
        angle_diff = (env.get_angle() - trail_angle) % 360
        while not abs(angle_diff) < 30:
            obs, reward, terminated, truncated, _ = env.step(rot_action)
            observations.append(obs)
            actions.append(rot_action)
            rewards.append(reward)
            angle_diff = (env.get_angle() - trail_angle) % 360

            if terminated or truncated: break
        if terminated or truncated: break
        acc_steps = trail_strategy(point_dist(start, end))

        for i in range(acc_steps):
            obs, reward, terminated, truncated, _ = env.step(minecart.ACT_ACCEL)
            observations.append(obs)
            actions.append(minecart.ACT_ACCEL)
            rewards.append(reward)

            if terminated or truncated: break
        if terminated or truncated: break

        while not env.spin_allowed():
            obs, reward, terminated, truncated, _ = env.step(minecart.ACT_NONE)
            observations.append(obs)
            actions.append(minecart.ACT_NONE)
            rewards.append(reward)

            if terminated or truncated: break
        if terminated or truncated: break

        obs, reward, terminated, truncated, _ = env.step(minecart.ACT_BRAKE)
        observations.append(obs)
        actions.append(minecart.ACT_BRAKE)
        rewards.append(reward)
        if terminated or truncated: break

        obs, reward, terminated, truncated, _ = env.step(minecart.ACT_MINE)
        observations.append(obs)
        actions.append(minecart.ACT_MINE)
        rewards.append(reward)
        if terminated or truncated: break

    return observations, actions, rewards



def main():
    env_id = "minecart-v0"
    env, eval_env = initialize_setting(env_id, minetrain=True)

    paths = path_combinations(eval_env.mines, eval_env.base)
    example = paths[30]
    print('path: ', example)
    observations, actions, rewards = path_playout(eval_env, example)
    for o in observations: print(o)
    fig = minecart_trajectories([observations], [actions], mines=eval_env.mines, alpha=0.8, color='brown')
    fig.show()

if __name__ == '__main__':
    main()