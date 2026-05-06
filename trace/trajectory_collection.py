import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.chdir('..')

from yaml import safe_load

from trace.core import initialize_setting, dst_ground_truth, TrajectoryManager
from trace.morl_baselines.multi_policy.ipro.ipro import IPRO

def main():
    config = "minetrain"
    iter_total_timesteps = 1_000_000
    metadata = safe_load(open(f"trace/configs/{config}.yaml", "r"))

    env_id = metadata['env_id']
    ref_point = metadata['ref_point']
    file_prefix = metadata['file_prefix']
    gamma = metadata['gamma']
    num_steps = metadata['num_steps']
    tolerance = float(metadata['tolerance'])
    eval_episodes = metadata['eval_episodes']

    env, eval_env = initialize_setting(env_id=env_id, minetrain='minetrain' in env_id)
    """
    method = 'ground_truth'
    filepath =f"data/{file_prefix}_{method}.json"
    TrajectoryManager(metadata).load(dst_ground_truth(eval_env.unwrapped.sea_map, metadata['action_mapping'])).save(filepath)
    print(f'Saved ground truth trajectories in {filepath}.')
    
    return
    """

    method = "ipro"
    filepath = f"data/{file_prefix}_{method}.json"
    ipro = IPRO(
        env=env,
        direction="maximize",
        tolerance=tolerance,
        max_iterations=None,
        iter_total_timesteps=iter_total_timesteps,
        num_steps=num_steps,
        device="cpu",
        log=False,
        seed=0,
        gamma=gamma
    )
    print('Initialized environment and algorithm.')
    pareto_set = ipro.train(
        eval_env=eval_env,
        ref_point=ref_point,
        deterministic=eval_episodes==1,
        eval_episodes=eval_episodes,
    )
    manager = TrajectoryManager(metadata).load([traj for _, _, traj in pareto_set])

    manager.save(filepath)
    print(f'Saved Pareto trajectories in {filepath}.')
    print(f'\nIPRO Pareto front points:')

    for i, point in enumerate(ipro.get_pareto_front()):
        print(f'{i}: {point}')



if __name__ == "__main__":
    main()