import numpy as np
np.set_printoptions(suppress=True, precision=2)

from trace.core import initialize_setting, path_combinations, seg_angle, point_dist, best_acc_steps, same_point, angle_subtraction
from trace.visuals import minecart_trajectories

import mo_gymnasium.envs.minecart.minecart as minecart

def perform_action(env, action, episode_dict):
    episode_dict['actions'].append(action)
    observations, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated: return env, True

    episode_dict['observations'].append(observations)
    episode_dict['rewards'].append(reward)
    return env, False


def path_playout(env, path):
    obs, _ = env.reset()
    episode_dict, terminated = {'observations': [obs], 'actions': [], 'rewards': []}, False

    for start, end in zip(path, path[1:]):
        trail_angle = seg_angle(start, end)
        rot_action = minecart.ACT_LEFT if env.get_angle() - trail_angle > 0 else minecart.ACT_RIGHT
        while not -10 < abs(angle_subtraction(env.get_angle(), trail_angle)) < 10:
            env, terminated = perform_action(env, rot_action, episode_dict)
            if terminated: break
        if terminated: break

        for _ in range(best_acc_steps(point_dist(start, end))):
            env, terminated = perform_action(env, minecart.ACT_ACCEL, episode_dict)
            if terminated or same_point(env.get_pos(), end): break
        if terminated: break

        while not same_point(env.get_pos(), end):
            env, terminated = perform_action(env, minecart.ACT_NONE, episode_dict)
            if terminated: break
        if terminated: break
        env, terminated = perform_action(env, minecart.ACT_BRAKE, episode_dict)
        if terminated: break
        env, terminated = perform_action(env, minecart.ACT_MINE, episode_dict)
        if terminated: break

    print('final length: ', len(episode_dict['observations']))
    return episode_dict


def main():
    env_id = "minecart-v0"
    env, eval_env = initialize_setting(env_id, minetrain=True)
    paths = path_combinations(eval_env.mines, eval_env.base)

    example = paths[27]
    print('path: ', example, '\n')

    episode_dict = path_playout(eval_env, example)
    observations, actions = episode_dict['observations'], episode_dict['actions']
    for a in actions:
        if a == 0 : print(a)
    fig = minecart_trajectories([observations], [actions], mines=eval_env.mines, alpha=0.8, color='brown')
    fig.show()

if __name__ == '__main__':
    main()