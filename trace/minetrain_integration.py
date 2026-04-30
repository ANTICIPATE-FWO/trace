import numpy as np
np.set_printoptions(suppress=True, precision=2)

from trace.core import initialize_setting
from trace.visuals import minecart_trajectories


def main():
    env_id = "minecart-v0"
    env, eval_env = initialize_setting(env_id, minetrain=True)
    observations, actions = [], []
    obs, _ = eval_env.reset()
    observations.append(obs)

    for i in range(1000):
        if i % 100 == 0: print(f'step {i}')
        action = eval_env.action_space.sample()
        actions.append(action)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        observations.append(obs)

        if terminated or truncated:
            print(f'terminated at step {i} final position: {obs[:2]}')
            break

    observations = np.array(observations)

    mines = [tuple(mine.pos) for mine in eval_env.unwrapped.mines]

    fig = minecart_trajectories([observations], [actions], mines=mines, alpha=0.8, color='brown')
    fig.show()

if __name__ == '__main__':
    main()