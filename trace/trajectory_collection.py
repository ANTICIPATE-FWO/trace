import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.chdir('..')

from yaml import safe_load

from trace.core import initialize_setting, dst_ground_truth
from trace.morl_baselines.multi_policy.ipro.ipro import IPRO
from trace.core import TrajectoryManager

def main():
    env_id = "deep-sea-treasure-concave-v0"
    metadata = safe_load(open("trace/configs/environments.yaml", "r"))[env_id]

    env, eval_env = initialize_setting(env_id=env_id)
    ref_point, file_prefix = metadata['ref_point'], metadata['file_prefix']

    """
    method = 'ground_truth'
    filepath =f"data/{file_prefix}_{method}.json"
    TrajectoryManager(env_id).load(dst_ground_truth(eval_env.unwrapped.sea_map, metadata['action_mapping'])).save(filepath)
    print(f'Saved ground truth trajectories in {filepath}.')
    
    return
    """

    method = "ipro"
    filepath = f"data/{file_prefix}_{method}.json"
    ipro = IPRO(
        env=env,
        direction="maximize",
        tolerance=1e-15,
        max_iterations=100,
        iter_total_timesteps=1_000_000,
        num_steps=50,
        learning_rate=2.5e-4,
        device="cpu",
        log=False,
        seed=0,
        gamma=1.0
    )
    print('Initialized environment and algorithm.')

    pareto_set = ipro.train(
        eval_env=eval_env,
        ref_point=ref_point,
        deterministic=False,
    )
    TrajectoryManager(metadata).load([traj for _, _, traj in pareto_set]).save(filepath)
    print(f'Saved Pareto trajectories in {filepath}.')
    print(f'\nIPRO Pareto front points:')

    for i, point in enumerate(ipro.get_pareto_front()):
        print(f'{i}: {point}')



if __name__ == "__main__":
    main()