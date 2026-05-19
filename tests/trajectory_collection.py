import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.chdir('..')

from trace.core import TrajectoryManager
from trace.policies import initialize_setting, dst_gt, minetrain_gt, IPRO


def ground_truth(metadata:dict, save:bool=True):
    env_id, file_prefix = metadata['env_id'], metadata['file_prefix']

    if 'minetrain' in file_prefix:
        _, env = initialize_setting(env_id=env_id)
        manager = TrajectoryManager(metadata).load(minetrain_gt(env))

    elif 'deep-sea-treasure' in file_prefix:
        _, env = initialize_setting(env_id=env_id, minetrain=True)
        sea_map, action_mapping = env.unwrapped.sea_map, metadata['action_mapping']
        manager = TrajectoryManager(metadata).load(dst_gt(sea_map, action_mapping))

    else:
        raise NotImplementedError(f'Ground truth not implemented for {env_id}')

    if save: manager.save(f"data/{file_prefix}_ground_truth.json")
    return manager


def ipro_samples(metadata:dict, iter_steps:int, save:bool=True, verbose:bool=True):
    env, eval_env = initialize_setting(env_id=metadata['env_id'], minetrain='minetrain' in metadata['env_id'])

    ipro = IPRO(
        env=env,
        direction="maximize",
        tolerance=float(metadata['tolerance']),
        max_iterations=None,
        iter_total_timesteps=iter_steps,
        num_steps=metadata['num_steps'],
        log=False,
        gamma=metadata['gamma'])

    pareto_set = ipro.train(
        eval_env=eval_env,
        ref_point=metadata['ref_point'],
        deterministic=metadata['eval_episodes'] == 1,
        eval_episodes=metadata['eval_episodes'],
        verbose=verbose)

    manager = TrajectoryManager(metadata).load([traj for _, _, traj in pareto_set])
    if save: manager.save(f"data/{metadata['file_prefix']}_ipro.json")
    return manager


